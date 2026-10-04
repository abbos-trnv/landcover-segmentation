"""Загрузка тайлов LandCover.ai напрямую из исходных GeoTIFF-сцен."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torch.utils.data import Dataset


IGNORE_INDEX = 255


class LandCoverTileDataset(Dataset):
    """Пары RGB-тайл/маска по строкам tile_index.csv.

    Крайние тайлы дополняются нулями у изображения и IGNORE_INDEX у маски.
    Поэтому padding не создаёт искусственный класс "фон" при обучении.
    """

    def __init__(
        self,
        tile_index,
        image_dir: str | Path,
        mask_dir: str | Path,
        tile_size: int = 512,
        transform=None,
    ):
        self.tile_index = tile_index.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_size = tile_size
        self.transform = transform

    def __len__(self):
        return len(self.tile_index)

    def __getitem__(self, index: int):
        row = self.tile_index.iloc[index]
        scene = row["scene"]
        x, y = int(row["x"]), int(row["y"])
        window = Window(x, y, self.tile_size, self.tile_size)

        with rasterio.open(self.image_dir / f"{scene}.tif") as src:
            image = src.read(
                window=window,
                boundless=True,
                fill_value=0,
            )

        with rasterio.open(self.mask_dir / f"{scene}.tif") as src:
            mask = src.read(
                1,
                window=window,
                boundless=True,
                fill_value=IGNORE_INDEX,
            )

        image = torch.from_numpy(image.astype(np.float32) / 255.0)
        mask = torch.from_numpy(mask.astype(np.int64))

        if self.transform is not None:
            transformed = self.transform(
                image=image.permute(1, 2, 0).numpy(),
                mask=mask.numpy(),
            )
            image = transformed["image"]
            mask = transformed["mask"].long()

        return {
            "image": image,
            "mask": mask,
            "scene": scene,
            "x": x,
            "y": y,
            "is_edge_tile": bool(row.get("is_edge_tile", False)),
        }
