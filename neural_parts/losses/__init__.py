import torch

from ..stats_logger import StatsLogger
from .chamfer_loss_functions import chamfer_loss, \
    true_chamfer_loss, normal_consistency_loss
from .occupancy_loss_functions import implicit_surface_loss, \
    conservative_implicit_surface_loss, min_points_loss


def get_loss(loss_types, config):
    losses = {
        "implicit_surface_loss": implicit_surface_loss,
        "chamfer_loss": chamfer_loss,
        "conservative_implicit_surface_loss": conservative_implicit_surface_loss,
        "true_chamfer_loss": true_chamfer_loss,
        "normal_consistency_loss": normal_consistency_loss,
        "min_points_loss": min_points_loss,
        "size_regularizer": size_regularizer,
        "non_overlapping_regularizer": non_overlapping_regularizer,
    }

    return weighted_sum_of_losses(
        *[losses[t] for t in loss_types],
        weights=config["weights"]
    )


def weighted_sum_of_losses(*functions, weights=None):
    if weights is None:
        weights = [1.0]*len(functions)
    def inner(*args, **kwargs):
        return sum(
            w*loss_fn(*args, **kwargs)
            for w, loss_fn in zip(weights, functions)
        )
    return inner


def size_regularizer(predictions, target, config):
    prim_points = predictions["y_prim"]

    # Compute the pairwise square dists between points on a primitive
    loss = ((prim_points[:, :, None] - prim_points[:, None])**2).sum(-1).mean()

    StatsLogger.instance()["size"].value = loss.item()

    return loss


def non_overlapping_regularizer(predictions, target, config):
    max_shared_points = config.get("max_shared_points", 2)
    sharpness = config.get("sharpness", 1.0)

    phi_volume = predictions["phi_volume"]
    # Compute the inside/outside function based on these distances
    primitive_inside_outside = torch.sigmoid(sharpness * (-phi_volume))
    loss = (primitive_inside_outside.sum(-1) - max_shared_points).relu().mean()

    StatsLogger.instance()["overlapping"].value = loss.item()

    return loss
