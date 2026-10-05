"""Автоматический экспорт графиков, примеров inference и одного run-архива."""

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch


CLASS_COLORS = ["#222222", "#e53935", "#2e9b4b", "#2f7ed8", "#f6d32d"]


def export_run_artifacts(model, test_dataset, device, run_dir, example_count: int = 6):
    """Создаёт figures/ и один ZIP-архив завершённого experiment run.

    Требует уже сохранённых history.csv, test_metrics.csv и
    test_per_scene_metrics.csv. Предсказания делаются лучшим checkpoint,
    который к моменту вызова должен быть загружен в model.
    """
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    history = pd.read_csv(run_dir / "history.csv")
    best_epoch = history.loc[history["val_foreground_miou"].idxmax()]

    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Learning curves")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_foreground_miou"], marker="o", markersize=3)
    axes[1].axvline(best_epoch["epoch"], color="tab:red", linestyle="--")
    axes[1].set(xlabel="Epoch", ylabel="foreground mIoU", title="Validation foreground mIoU")
    figure.tight_layout()
    figure.savefig(figures_dir / "training_curves.png", dpi=180)
    plt.close(figure)

    metrics = pd.read_csv(run_dir / "test_metrics.csv").iloc[0]
    per_scene = pd.read_csv(run_dir / "test_per_scene_metrics.csv").sort_values("foreground_miou")
    class_names = ["Buildings", "Woodland", "Water", "Roads"]
    class_values = [metrics[f"iou_{name.lower()}"] for name in class_names]

    figure, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].bar(class_names, class_values, color=CLASS_COLORS[1:])
    axes[0].set(ylim=(0, 1), ylabel="IoU", title="Test IoU by class")
    axes[1].barh(per_scene["scene"], per_scene["foreground_miou"], color="tab:blue")
    axes[1].set(xlim=(0, 1), xlabel="foreground mIoU", title="Test quality by scene")
    figure.tight_layout()
    figure.savefig(figures_dir / "test_metrics.png", dpi=180)
    plt.close(figure)

    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad(color="white")
    # Для иллюстраций берём непограничные тайлы: у них нет искусственного
    # padding, поэтому зритель не примет предсказания вне сцены за ошибку.
    candidate_indices = [
        index for index, is_edge in enumerate(test_dataset.tile_index["is_edge_tile"])
        if not is_edge
    ]
    sample_indices = np.linspace(
        0, len(candidate_indices) - 1, min(example_count, len(candidate_indices)), dtype=int
    )
    sample_indices = [candidate_indices[index] for index in sample_indices]
    model.eval()
    for number, index in enumerate(sample_indices, start=1):
        sample = test_dataset[index]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(image).argmax(dim=1)[0].cpu().numpy()

        figure, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sample["image"].permute(1, 2, 0).numpy())
        axes[0].set_title(f"Image: {sample['scene']}")
        target = sample["mask"].numpy()
        padding = target == 255
        axes[1].imshow(np.ma.masked_where(padding, target), cmap=cmap, vmin=0, vmax=4)
        axes[1].set_title("Ground truth")
        axes[2].imshow(np.ma.masked_where(padding, prediction), cmap=cmap, vmin=0, vmax=4)
        axes[2].set_title("Prediction")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(figures_dir / f"prediction_{number:02d}.png", dpi=180)
        plt.close(figure)

    archive = shutil.make_archive(
        str(run_dir.parent / f"{run_dir.name}_export"),
        "zip",
        root_dir=run_dir.parent,
        base_dir=run_dir.name,
    )
    return {"figures_dir": str(figures_dir), "archive": archive}
