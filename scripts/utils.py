import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from neural_parts.utils import sphere_mesh


def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=Loader)
    return config


def points_on_sphere(N, n_primitives=1, device="cpu"):
    """Get uniform points on a sphere that will be deformed with the IN.

    Arguments:
    ----------
        N: int, number of points to be sampled
        n_primitives: int, number of primitives
    """
    vertices, faces = sphere_mesh(n=N, device=device)
    vertices = vertices.view(1, -1, 1, 3)
    vertices = vertices.expand((1, -1, n_primitives, 3))
    vertices = vertices.contiguous()
    return vertices, faces
