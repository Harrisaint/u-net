"""
U-Net model setup (segmentation_models_pytorch) and inference utilities.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from device import DEVICE


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    out_classes: int = 1,
) -> nn.Module:
    """Instantiate a U-Net with a pre-trained encoder."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=out_classes,
        activation=None,
    )
    return model


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

@torch.no_grad()
def predict(
    model: nn.Module,
    image_tensor: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a single image through the model.

    Parameters
    ----------
    model : nn.Module
        Trained U-Net (will be moved to ``DEVICE`` automatically).
    image_tensor : torch.Tensor
        Shape ``(3, H, W)`` or ``(1, 3, H, W)``, float32 in [0, 1].
    threshold : float
        Probability cutoff for the binary mask.

    Returns
    -------
    prob_map : torch.Tensor
        Continuous probability map in [0, 1], shape ``(1, H, W)``.
        Always returned on CPU.
    binary_mask : torch.Tensor
        Thresholded mask (0 / 1), same shape.  Always returned on CPU.
    """
    model.eval()
    model.to(DEVICE)

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)
    logits = model(image_tensor)
    prob_map = torch.sigmoid(logits).squeeze(0)
    binary_mask = (prob_map >= threshold).float()

    return prob_map.cpu(), binary_mask.cpu()
