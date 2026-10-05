"""Фабрика моделей для controlled CNN и CNN-vs-Transformer экспериментов."""

import torch.nn as nn
import torch.nn.functional as functional
import segmentation_models_pytorch as smp


SUPPORTED_ARCHITECTURES = {
    "unet": smp.Unet,
    "fpn": smp.FPN,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


class SegFormerB0Scratch(nn.Module):
    """SegFormer-B0, инициализированный без внешних весов.

    Hugging Face возвращает logits на масштабе 1/4 от входа. Этот адаптер
    восстанавливает их до размера тайла, чтобы loss и метрики были теми же,
    что у CNN-моделей.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as error:
            raise ImportError(
                "SegFormer requires `transformers`. Install it with "
                "`pip install transformers`."
            ) from error

        config = SegformerConfig(
            num_channels=3,
            num_labels=num_classes,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            num_attention_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            decoder_hidden_size=256,
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images):
        logits = self.model(pixel_values=images).logits
        return functional.interpolate(
            logits,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def create_model(
    architecture: str,
    encoder_name: str = "resnet18",
    encoder_weights=None,
    num_classes: int = 5,
):
    """Создаёт модель; в основном эксперименте encoder_weights всегда None."""
    key = architecture.lower()
    if key == "segformer_b0":
        if encoder_weights is not None:
            raise ValueError("SegFormer experiment must use scratch initialization.")
        return SegFormerB0Scratch(num_classes)
    if key not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return SUPPORTED_ARCHITECTURES[key](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
