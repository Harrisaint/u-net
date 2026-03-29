"""
Heatmap generation: overlay a U-Net probability map onto the original image.
"""

import cv2
import numpy as np
import torch


def generate_heatmap_overlay(
    prob_map: torch.Tensor,
    image_tensor: torch.Tensor,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a probability heatmap with the original image.

    Parameters
    ----------
    prob_map : torch.Tensor
        Probability values in [0, 1], shape ``(1, H, W)`` or ``(H, W)``.
    image_tensor : torch.Tensor
        Original image, shape ``(3, H, W)``, float32 in [0, 1].
    alpha : float
        Opacity of the heatmap layer (0 = invisible, 1 = fully opaque).
    colormap : int
        OpenCV colormap constant.

    Returns
    -------
    overlay : np.ndarray
        Blended image, uint8 RGB, shape ``(H, W, 3)``.
    """
    if prob_map.dim() == 3:
        prob_map = prob_map.squeeze(0)

    prob_np = prob_map.numpy()
    heat_uint8 = (prob_np * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    image_np = (
        image_tensor.permute(1, 2, 0).numpy() * 255
    ).astype(np.uint8)

    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap_rgb, alpha, 0)
    return overlay
