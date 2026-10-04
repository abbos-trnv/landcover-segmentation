"""Функции потерь с корректной обработкой padding-пикселей."""

import torch
import torch.nn.functional as F


def multiclass_dice_loss(logits, target, num_classes: int, ignore_index: int = 255):
    """Macro Dice loss без padding и без искусственной замены его фоном."""
    valid = target != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    probs = torch.softmax(logits, dim=1)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    valid = valid.unsqueeze(1)

    probs = probs * valid
    one_hot = one_hot * valid
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    cardinality = (probs + one_hot).sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
    return 1.0 - dice.mean()


def ce_dice_loss(logits, target, num_classes: int, ignore_index: int = 255, ce_weight=None):
    """Базовый loss: Cross-Entropy + Dice с равными коэффициентами."""
    ce = F.cross_entropy(logits, target, weight=ce_weight, ignore_index=ignore_index)
    dice = multiclass_dice_loss(logits, target, num_classes, ignore_index)
    return ce + dice, {"cross_entropy": ce.detach(), "dice": dice.detach()}
