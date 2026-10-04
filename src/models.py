"""Фабрика моделей для контролируемого decoder-сравнения."""

import segmentation_models_pytorch as smp


SUPPORTED_ARCHITECTURES = {
    "unet": smp.Unet,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def create_model(
    architecture: str,
    encoder_name: str = "resnet18",
    encoder_weights=None,
    num_classes: int = 5,
):
    """Создаёт модель; в основном эксперименте encoder_weights всегда None."""
    key = architecture.lower()
    if key not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return SUPPORTED_ARCHITECTURES[key](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
