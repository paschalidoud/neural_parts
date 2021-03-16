import torch

from ..stats_logger import StatsLogger


def chamfer_loss(predictions, targets, config):
    """Compute the bidirectional Chamfer loss between the target and the
    predicted shape.

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points on the target surface
                 and their corresponding normals
        config: A dictionary containing the loss parameters
    """
    # Extract the target surface points and some shapes into local variables
    targets = targets[-1]
    B, N2, _ = targets.shape
    points = targets[:, :, :3]

    # Extract the predicted surface.
    y_prim = predictions["y_prim"]
    _, M, N1, _ = y_prim.shape

    diff = y_prim[:, :, :, None] - points[:, None, None]
    dists = torch.square(diff).sum(-1)
    assert dists.shape == (B, M, N1, N2)

    target_to_prim = dists.min(1)[0].min(1)[0].mean()
    prim_to_target = dists.min(3)[0].mean()
    loss = target_to_prim + prim_to_target

    # log the chamfer loss
    StatsLogger.instance()["chamfer"].value = loss.item()
    StatsLogger.instance()["target_to_prim"].value = target_to_prim.item()
    StatsLogger.instance()["prim_to_target"].value = prim_to_target.item()

    return loss


def true_chamfer_loss(predictions, targets, config):
    """Compute the chamfer distance between the predicted surface points
    y_prim[on_surface] and targets[-1].

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points on the target surface
                 and their corresponding normals
        config: A dictionary containing the loss parameters
    """
    # A tensor of shape (B, S, M, 3) that contains points that are in
    # the surface of each primitive
    y_prim = predictions["y_prim"]
    B, S, M, D = y_prim.shape
    # A bool tensor of shape (B, S, M) that indicates which of the y_prim
    # points are actually on the surface of the predicted object.
    on_surface = predictions["on_surface"]

    # Extract the ground truth points
    points = targets[-1][:, :, :D]

    # We don't need the on_surface points but the internal points which is all
    # the rest (the not on surface). Also we reshape it to be broadcastable
    # with the dists computed later on.
    internal = ~on_surface[:, :, :, None]

    # Compute the pairwise distances and put a large number in points that
    # should be ignored
    dists = ((y_prim[:, :, :, None] - points[:, None, None])**2)
    dists = torch.sqrt(dists.sum(-1))

    # Compute the target to primitive distances by setting to inf all the
    # distances that should be ignored in order for the min() operation to
    # ignore them
    target_dists = dists.masked_fill(internal, float("inf"))
    target_to_prim = dists.min(1)[0].min(1)[0].mean()

    # Compute the primitive to target distances by setting to 0 all the
    # distances that should be ignored and accounting for them in the averaging
    prim_dists = dists.masked_fill(internal, 0)
    prim_to_target = prim_dists.min(3)[0].mean()
    prim_to_target = prim_to_target / on_surface.float().mean()

    target_to_prim_w = config.get("target_to_prim_weight", 1.0)
    prim_to_target_w = config.get("prim_to_target_weight", 1.0)
    loss = (
        target_to_prim_w * target_to_prim +
        prim_to_target_w * prim_to_target
    )

    StatsLogger.instance()["chamfer"].value = loss.item()
    StatsLogger.instance()["target_to_prim"].value = target_to_prim.item()
    StatsLogger.instance()["prim_to_target"].value = prim_to_target.item()

    return loss


def normal_consistency_loss(predictions, targets, config):
    """
    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points on the target surface
                 and their corresponding normals
        config: A dictionary containing the loss parameters
    """
    phi_surface = predictions["phi_surface"]
    B, N, M = phi_surface.shape

    targets = targets[-1]
    B, N, D = targets.shape
    points = targets[:, :, :D//2]
    normals = targets[:, :, D//2:]

    # Select the primitives that "own" the surface points
    batch_indices = torch.arange(B, device=points.device)[:, None]
    point_indices = torch.arange(N, device=points.device)[None, :]
    prim_indices = (phi_surface**2).min(2)[1]
    phi_surface = phi_surface[
        batch_indices,
        point_indices,
        prim_indices
    ]

    # Compute the prediction normals by computing the gradient of the implicit
    # surface w.r.t. to the target points
    predicted_normals = torch.autograd.grad(
        phi_surface.sum(),
        targets,
        retain_graph=True,
        create_graph=True
    )[0][:, :, :3]
    predicted_normals_norm = torch.sqrt(
        torch.square(predicted_normals).sum(-1, keepdim=True)
    )
    predicted_normals = predicted_normals / predicted_normals_norm

    # Now compute the cosine distance between the ground truth and predicted
    # normals
    loss = (1 - torch.einsum("bne,bne->bn", normals, predicted_normals)).mean()

    StatsLogger.instance()["normals"].value = loss.item()

    return loss
