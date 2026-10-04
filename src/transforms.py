"""Одинаковые аугментации для всех архитектур основного сравнения."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def training_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
    ])


def validation_transform():
    return A.Compose([ToTensorV2()])
