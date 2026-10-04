"""Построение воспроизводимого индекса тайлов по исходным сценам."""

from pathlib import Path

import pandas as pd
import rasterio


def build_tile_index(image_dir, scene_split: dict, tile_size: int = 512):
    """Нарезает сцены сеткой без перекрытия, не сохраняя копии GeoTIFF на диск."""
    rows = []
    for split, scenes in scene_split.items():
        if split not in {"train", "val", "test"}:
            continue
        for scene in scenes:
            with rasterio.open(Path(image_dir) / f"{scene}.tif") as source:
                height, width = source.height, source.width
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    rows.append(
                        {
                            "split": split,
                            "scene": scene,
                            "x": x,
                            "y": y,
                            "is_edge_tile": x + tile_size > width or y + tile_size > height,
                        }
                    )
    return pd.DataFrame(rows)
