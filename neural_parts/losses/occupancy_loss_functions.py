import torch

from ..stats_logger import StatsLogger


def implicit_surface_loss(predictions, targets, config):
    """Uses the implicit surface values as predictions of inside/outside per
    point and computes the binary cross entropy loss for those predictions.

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
        config: A dictionary containing the loss parameters
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]
    B, N, _ = gt_labels.shape

    # Tensor of shape (B, N, M) containing the implicit surface values for
    # points in space
    phi_volume = predictions["phi_volume"]
    assert len(phi_volume.shape) == 3

    # Compute the inside/outside function based on these distances
    sharpness = config.get("sharpness", 1.0)
    primitive_inside_outside = sharpness * (-phi_volume)

    if config.get("smooth_max", False):
        union_inside_outside = torch.logsumexp(
            primitive_inside_outside, dim=-1, keepdims=True
        )
    else:
        union_inside_outside = primitive_inside_outside.max(-1, keepdims=True)[0]

    # Compute the binary classification loss
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        union_inside_outside, gt_labels, reduction="none"
    )
    loss = gt_weights * loss
    loss = loss.mean()

    # log the implicit occupancy loss
    StatsLogger.instance()["impl_occ"].value = loss.item()

    return loss


def conservative_implicit_surface_loss(predictions, targets, config):
    """
    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
        config: A dictionary containing the loss parameters
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]
    B, N, _ = gt_labels.shape

    # Tensor of shape (B, N, M) containing the implicit surface values for
    # points in space
    phi_volume = predictions["phi_volume"]
    assert len(phi_volume.shape) == 3

    # Take the smallest distance primitive-wise
    if config.get("smooth_max", False):
        phi_volume = -torch.logsumexp(-phi_volume, dim=2, keepdim=True)
    else:
        phi_volume = phi_volume.min(2, keepdim=True)[0]

    inside_but_out = (phi_volume.relu() * gt_weights)[gt_labels == 1].sum()
    outside_but_in = ((-phi_volume).relu() * gt_weights)[gt_labels == 0].sum()
    loss = (inside_but_out + outside_but_in) / (B*N)

    StatsLogger.instance()["inside_but_out"].value = inside_but_out.item()
    StatsLogger.instance()["outside_but_in"].value = outside_but_in.item()
    StatsLogger.instance()["cons_occ"].value = loss.item()

    return loss


def min_points_loss(predictions, targets, config):
    """"Uses the implicit surface values to make sure that every primitive is
    "responsible" for a specific number of points from the target shape.

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
        config: A dictionary containing the loss parameters
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]
    B, N, _ = gt_labels.shape

    # Tensor of shape (B, N, M) containing the implicit surface values for
    # points in space
    phi_volume = predictions["phi_volume"]
    assert len(phi_volume.shape) == 3
    assert phi_volume.shape[0] == B
    assert phi_volume.shape[1] == N

    # The minimum number of points closest to a primitive
    k = config.get("min_n_points", 10)
    sharpness = config.get("sharpness", 1.0)

    # Mask all the points that are not internal to the target shape
    implicit = phi_volume + ((1-gt_labels) * 1e6)

    # Get the the top k points per primitive
    implicit = torch.topk(implicit, k, dim=1, largest=False, sorted=False)[0]

    # Make sure that they are inside
    occupancy = torch.sigmoid((-sharpness * implicit))
    loss = -torch.log(occupancy + 1e-6).mean()

    StatsLogger.instance()["min_k"].value = loss.item()

    return loss
