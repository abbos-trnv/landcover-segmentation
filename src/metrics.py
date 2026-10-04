"""Метрики semantic segmentation: IoU и Dice без padding."""

import numpy as np


def confusion_matrix_from_masks(prediction, target, num_classes: int = 5, ignore_index: int = 255):
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = target != ignore_index
    codes = num_classes * target[valid].astype(int) + prediction[valid].astype(int)
    return np.bincount(codes, minlength=num_classes**2).reshape(num_classes, num_classes)


def segmentation_scores(confusion_matrix, class_names=None):
    cm = np.asarray(confusion_matrix, dtype=np.float64)
    true_positive = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - true_positive
    dice_denominator = cm.sum(axis=1) + cm.sum(axis=0)
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    dice = np.divide(2 * true_positive, dice_denominator, out=np.full_like(true_positive, np.nan), where=dice_denominator > 0)
    names = class_names or ["background", "buildings", "woodland", "water", "roads"]
    result = {f"iou_{name}": float(value) for name, value in zip(names, iou)}
    result.update({f"dice_{name}": float(value) for name, value in zip(names, dice)})
    result["foreground_miou"] = float(np.nanmean(iou[1:]))
    result["macro_dice"] = float(np.nanmean(dice[1:]))
    result["pixel_accuracy"] = float(true_positive.sum() / cm.sum()) if cm.sum() else float("nan")
    return result
