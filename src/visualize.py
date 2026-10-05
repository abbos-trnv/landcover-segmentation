"""Сохранение одинаковых визуальных примеров для сравнения моделей."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


CLASS_COLORS = ["#222222", "#e53935", "#2e9b4b", "#2f7ed8", "#f6d32d"]
CLASS_NAMES = ["Фон", "Здания", "Лес", "Вода", "Дороги"]


def save_prediction_figure(image, target, prediction, output_path, ignore_index=255):
    """Сохраняет RGB-тайл, true mask и model prediction в одном рисунке."""
    image = np.asarray(image)
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    target_array = np.asarray(target)
    padding = target_array == ignore_index
    target = np.ma.masked_where(padding, target_array)
    prediction = np.ma.masked_where(padding, np.asarray(prediction))

    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad(color="white")
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Аэрофотоснимок")
    axes[1].imshow(target, cmap=cmap, vmin=0, vmax=4)
    axes[1].set_title("Истинная маска")
    axes[2].imshow(prediction, cmap=cmap, vmin=0, vmax=4)
    axes[2].set_title("Предсказание")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
