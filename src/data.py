"""Загрузка тайлов LandCover.ai напрямую из исходных GeoTIFF-сцен."""

from pathlib import Path
from collections import OrderedDict

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
        max_open_scenes: int = 41,
    ):
        self.tile_index = tile_index.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.tile_size = tile_size
        self.transform = transform
        # Внутри worker-процесса кэшируем открытые GeoTIFF. Без кэша для
        # каждого из тысяч тайлов заново открывались image и mask файлы.
        self.max_open_scenes = max_open_scenes
        self._sources = OrderedDict()

    def __len__(self):
        return len(self.tile_index)

    def _scene_sources(self, scene):
        if scene in self._sources:
            self._sources.move_to_end(scene)
            return self._sources[scene]

        pair = (
            rasterio.open(self.image_dir / f"{scene}.tif"),
            rasterio.open(self.mask_dir / f"{scene}.tif"),
        )
        self._sources[scene] = pair
        while len(self._sources) > self.max_open_scenes:
            _, old_pair = self._sources.popitem(last=False)
            old_pair[0].close()
            old_pair[1].close()
        return pair

    def __getstate__(self):
        """Не переносить открытые файловые дескрипторы в DataLoader workers."""
        state = self.__dict__.copy()
        state["_sources"] = OrderedDict()
        return state

    def close(self):
        for image_source, mask_source in self._sources.values():
            image_source.close()
            mask_source.close()
        self._sources.clear()

    def __getitem__(self, index: int):
        row = self.tile_index.iloc[index]
        scene = row["scene"]
        x, y = int(row["x"]), int(row["y"])
        window = Window(x, y, self.tile_size, self.tile_size)

        image_source, mask_source = self._scene_sources(scene)
        image = image_source.read(
            window=window,
            boundless=True,
            fill_value=0,
        )
        mask = mask_source.read(
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
