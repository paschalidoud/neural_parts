import torch

from ..stats_logger import StatsLogger


def get_metrics(keys):
    metrics = {
        "iou": iou_metric,
        "iou_occnet": iou_occnet_metric,
        "classification_metrics": classification_metrics
    }

    if keys is None:
        keys = []

    return list_of_metrics(*[metrics[k] for k in keys])


def list_of_metrics(*functions):
    def inner(*args, **kwargs):
        return [
            metric_fn(*args, **kwargs) for metric_fn in functions
        ]
    return inner


def classification_metrics(predictions, targets):
    """Compute various classification metrics between the prediction of the
    neural network and the ground truth.

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]
    B, N, _ = gt_labels.shape

    phi_volume = predictions["phi_volume"]
    assert len(phi_volume.shape) == 3

    # Take the distance to the closest primitive
    phi_volume = phi_volume.min(-1, keepdim=True)[0]
    assert phi_volume.shape == (B, N, 1)
    # Points that are inside a primitive should have a negative value, whereas
    # points with positive values are outside the predicted shape
    occ_pred = (phi_volume <= 0).float()

    true_positive = occ_pred[gt_labels == 1].sum()
    false_negative = (1 - occ_pred)[gt_labels == 1].sum()
    false_positive = occ_pred[gt_labels == 0].sum()

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = ((occ_pred == gt_labels).float() * gt_weights).mean()

    # log the accuracy, the precision and the recall
    StatsLogger.instance()["acc"].value = accuracy
    StatsLogger.instance()["prec"].value = precision
    StatsLogger.instance()["rec"].value = recall

    return accuracy, precision, recall


def iou_metric(predictions, targets):
    """Compute the IOU from the prediction of the neural network and the ground
    truth.

    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]

    phi_volume = predictions["phi_volume"]
    assert len(phi_volume.shape) == 3

    # Compute the minimum distance per point in order to get the occupancy
    # predictions
    occ_pred = (phi_volume.min(-1)[0] <= 0).float()
    iou_metric = iou(occ_pred, gt_labels.squeeze(-1), gt_weights.squeeze(-1))

    # log the iou metric
    StatsLogger.instance()["iou"].value = iou_metric
    return iou_metric


def iou_occnet_metric(predictions, targets):
    """
    Arguments:
    ----------
        predictions: A dictionary containing the network predictions
        targets: A list of tensors containing the points, the occupancy
                 labels and the weights.
    """
    # Extract some shapes and the targets into local variables
    gt_points = targets[0]
    gt_labels = targets[1]
    gt_weights = targets[2]
    B, N, _ = gt_labels.shape

    y_volume_latent = predictions["phi_volume"]
    occ_pred = torch.sigmoid(y_volume_latent.max(-1)[0])
    iou_metric = iou(occ_pred, gt_labels.squeeze(-1), gt_weights.squeeze(-1))
    StatsLogger.instance()["iou"].value = iou_metric

    return iou_metric


def iou(occ1, occ2, weights=None, average=True):
    """Compute the intersection over union (IoU) for two sets of occupancy
    values.

    Arguments:
    ----------
        occ1: Tensor of size BxN containing the first set of occupancy values
        occ2: Tensor of size BxN containing the first set of occupancy values

    Returns:
    -------
        the IoU
    """
    if not torch.is_tensor(occ1):
        occ1 = torch.tensor(occ1)
        occ2 = torch.tensor(occ2)

    if weights is None:
        weights = occ1.new_ones(occ1.shape)

    assert len(occ1.shape) == 2
    assert occ1.shape == occ2.shape

    # Convert them to boolean
    occ1 = occ1 >= 0.5
    occ2 = occ2 >= 0.5

    # Compute IoU
    area_union = (occ1 | occ2).float()
    area_union = (weights * area_union).sum(dim=-1)
    area_union = torch.max(area_union.new_tensor(1.0), area_union)
    area_intersect = (occ1 & occ2).float()
    area_intersect = (weights * area_intersect).sum(dim=-1)
    iou = (area_intersect / area_union)

    if average:
        return iou.mean().item()
    else:
        return iou
