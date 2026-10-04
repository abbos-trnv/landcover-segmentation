"""Общий training/evaluation loop для всех сравниваемых decoder-архитектур."""

from time import perf_counter

import numpy as np
import torch

from .losses import ce_dice_loss
from .metrics import confusion_matrix_from_masks, segmentation_scores


def _to_device(batch, device):
    return (
        batch["image"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )


def train_one_epoch(model, loader, optimizer, device, num_classes=5, ignore_index=255, ce_weight=None):
    model.train()
    total_loss = total_ce = total_dice = 0.0
    started_at = perf_counter()

    for batch in loader:
        images, masks = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss, parts = ce_dice_loss(logits, masks, num_classes, ignore_index, ce_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += parts["cross_entropy"].item()
        total_dice += parts["dice"].item()

    divisor = max(len(loader), 1)
    return {
        "loss": total_loss / divisor,
        "cross_entropy": total_ce / divisor,
        "dice_loss": total_dice / divisor,
        "seconds": perf_counter() - started_at,
    }


@torch.no_grad()
def evaluate(model, loader, device, num_classes=5, ignore_index=255, ce_weight=None):
    """Возвращает общие и per-scene метрики; padding исключён из всех расчётов."""
    model.eval()
    total_loss = 0.0
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    scene_cms = {}

    for batch in loader:
        images, masks = _to_device(batch, device)
        logits = model(images)
        loss, _ = ce_dice_loss(logits, masks, num_classes, ignore_index, ce_weight)
        total_loss += loss.item()

        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        for prediction, target, scene in zip(predictions, targets, batch["scene"]):
            cm = confusion_matrix_from_masks(prediction, target, num_classes, ignore_index)
            total_cm += cm
            scene_cms.setdefault(scene, np.zeros_like(total_cm))
            scene_cms[scene] += cm

    scores = segmentation_scores(total_cm)
    scores["loss"] = total_loss / max(len(loader), 1)
    per_scene = []
    for scene, cm in sorted(scene_cms.items()):
        row = {"scene": scene, **segmentation_scores(cm)}
        per_scene.append(row)
    return scores, per_scene
