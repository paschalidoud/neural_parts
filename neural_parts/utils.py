
from math import ceil, cos, pi, sin, sqrt

import torch

try:
    import trimesh
except ImportError:
    trimesh = None


def sphere_mesh(n=512, radius=1.0, device="cpu"):
    """Return a unit sphere mesh discretization with *at least* n vertices.
    
    Initially this returns a standard uv sphere but in the future we could
    change it to return a more sophisticated discretization.

    Code adapted from github.com/caosdoar/spheres/blob/master/src/spheres.cpp .
    """
    # Select the subdivisions such that we get at least n vertices
    parallels = ceil(sqrt(n/2))+2
    meridians = 2*parallels

    # Create all the vertices
    vertices = [[0, 1., 0]]
    for i in range(1, parallels):
        polar = pi * i / parallels
        sp = sin(polar)
        cp = cos(polar)
        for j in range(meridians):
            azimuth = 2 * pi * j / meridians
            sa = sin(azimuth)
            ca = cos(azimuth)
            vertices.append([sp * ca, cp, sp * sa])
    vertices.append([0, -1., 0])

    # Create the triangles
    faces = []
    for j in range(meridians):
        faces.append([0, (j+1) % meridians + 1, j+1])
    for i in range(parallels-2):
        a_start = i*meridians + 1
        b_start = (i+1)*meridians + 1
        for j in range(meridians):
            a1 = a_start + j
            a2 = a_start + (j+1) % meridians
            b1 = b_start + j
            b2 = b_start + (j+1) % meridians
            faces.append([a1, a2, b1])
            faces.append([b1, a2, b2])
    for j in range(meridians):
        a = j + meridians * (parallels-2) + 1
        b = (j+1) % meridians + meridians * (parallels-2) + 1
        faces.append([len(vertices)-1, a, b])

    vertices = torch.tensor(vertices, dtype=torch.float, device=device)
    faces = torch.tensor(faces, dtype=torch.long, device=device)

    return vertices * radius, faces


def sphere_edges(divisions=2, radius=1.0, device="cpu"):
    """Compute a sphere mesh using trimesh and return the sphere and the edges
    between points."""
    sphere = trimesh.creation.icosphere(subdivisions=divisions, radius=radius)
    edges = set()
    for face in sphere.faces:
        edges.add((face[0], face[1]))
        edges.add((face[0], face[2]))
        edges.add((face[1], face[2]))
    edges = torch.tensor(
        [[a, b] for a, b in edges],
        dtype=torch.long,
        device=device
    )
    vertices = torch.tensor(sphere.vertices, dtype=torch.float, device=device)

    return vertices, edges


def quaternions_to_rotation_matrices(quaternions):
    """
    Arguments:
    ---------
        quaternions: Tensor with size ...x4, where ... denotes any shape of
                     quaternions to be translated to rotation matrices

    Returns:
    -------
        rotation_matrices: Tensor with size ...x3x3, that contains the computed
                           rotation matrices
    """
    # Allocate memory for a Tensor of size ...x3x3 that will hold the rotation
    # matrix along the x-axis
    shape = quaternions.shape[:-1] + (3, 3)
    R = quaternions.new_zeros(shape)

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[..., 1]**2
    yy = quaternions[..., 2]**2
    zz = quaternions[..., 3]**2
    ww = quaternions[..., 0]**2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = torch.zeros_like(n)
    s[n != 0] = 2 / n[n != 0]

    xy = s[..., 0] * quaternions[..., 1] * quaternions[..., 2]
    xz = s[..., 0] * quaternions[..., 1] * quaternions[..., 3]
    yz = s[..., 0] * quaternions[..., 2] * quaternions[..., 3]
    xw = s[..., 0] * quaternions[..., 1] * quaternions[..., 0]
    yw = s[..., 0] * quaternions[..., 2] * quaternions[..., 0]
    zw = s[..., 0] * quaternions[..., 3] * quaternions[..., 0]

    xx = s[..., 0] * xx
    yy = s[..., 0] * yy
    zz = s[..., 0] * zz

    R[..., 0, 0] = 1 - yy - zz
    R[..., 0, 1] = xy - zw
    R[..., 0, 2] = xz + yw

    R[..., 1, 0] = xy + zw
    R[..., 1, 1] = 1 - xx - zz
    R[..., 1, 2] = yz - xw

    R[..., 2, 0] = xz - yw
    R[..., 2, 1] = yz + xw
    R[..., 2, 2] = 1 - xx - yy

    return R


def transform_to_primitives_centric_system(X, translations, rotation_angles):
    """
    Arguments:
    ----------
        X: Tensor with size BxNx3, containing the 3D points, where B is the
           batch size and N is the number of points
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx4 containing the 4 quaternion
                         values for the M primitives

    Returns:
    --------
        X_transformed: Tensor with size BxNxMx3 containing the N points
                       transformed in the M primitive centric coordinate
                       systems.
    """
    # Make sure that all tensors have the right shape
    assert X.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Subtract the translation and get X_transformed with size BxNxMx3
    X_transformed = X.unsqueeze(2) - translations.unsqueeze(1)

    # R = euler_angles_to_rotation_matrices(rotation_angles.view(-1, 3)).view(
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )

    # Let as denote a point x_p in the primitive-centric coordinate system and
    # its corresponding point in the world coordinate system x_w. We denote the
    # transformation from the point in the world coordinate system to a point
    # in the primitive-centric coordinate system as x_p = R(x_w - t)
    X_transformed = R.unsqueeze(1).matmul(X_transformed.unsqueeze(-1))

    X_signs = (X_transformed > 0).float() * 2 - 1
    X_abs = X_transformed.abs()
    X_transformed = X_signs * torch.max(X_abs, X_abs.new_tensor(1e-5))

    return X_transformed.squeeze(-1)


def transform_to_world_coordinates_system(X_P, translations, rotation_angles):
    """
    Arguments:
    ----------
        X_P: Tensor with size BxMxSx3, containing the 3D points, where B is
              the batch size, M is the number of primitives and S is the number
              of points on each primitive-centric system
        translations: Tensor with size BxMx3, containing the translation
                      vectors for the M primitives
        rotation_angles: Tensor with size BxMx4 containing the 4 quaternion
                         values for the M primitives

    Returns:
    --------
        X_P_w: Tensor with size BxMxSx3 containing the N points
                transformed in the M primitive centric coordinate
                systems.
    """
    # Make sure that all tensors have the right shape
    assert X_P.shape[0] == translations.shape[0]
    assert translations.shape[0] == rotation_angles.shape[0]
    assert translations.shape[1] == rotation_angles.shape[1]
    assert X_P.shape[1] == translations.shape[1]
    assert X_P.shape[-1] == 3
    assert translations.shape[-1] == 3
    assert rotation_angles.shape[-1] == 4

    # Compute the rotation matrices to every primitive centric coordinate
    # system (R has size BxMx3x3)
    R = quaternions_to_rotation_matrices(rotation_angles.view(-1, 4)).view(
        rotation_angles.shape[0], rotation_angles.shape[1], 3, 3
    )
    # We need the R.T to get the rotation matrix from the primitive-centric
    # coordinate system to the world coordinate system.
    R_t = torch.einsum("...ij->...ji", (R,))
    assert R.shape == R_t.shape

    X_Pw = R_t.unsqueeze(2).matmul(X_P.unsqueeze(-1))
    X_Pw = X_Pw.squeeze(-1) + translations.unsqueeze(2)

    return X_Pw
