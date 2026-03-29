"""
Extract quantitative metadata from a binary segmentation mask.
"""

from typing import Dict

import cv2
import numpy as np
import torch


def extract_mask_metadata(binary_mask: torch.Tensor) -> Dict:
    """Compute region-level statistics from a binary mask.

    Parameters
    ----------
    binary_mask : torch.Tensor
        Shape ``(1, H, W)`` or ``(H, W)``, values 0 or 1.

    Returns
    -------
    dict with keys:
        - ``area_ratio``   : float — fraction of pixels belonging to the lesion.
        - ``centroid``      : tuple[int, int] — ``(x, y)`` centre of mass, or
          ``None`` if no lesion is present.
        - ``bounding_box``  : dict with ``x, y, width, height``, or ``None``.
        - ``lesion_present``: bool
    """
    if binary_mask.dim() == 3:
        binary_mask = binary_mask.squeeze(0)

    mask_np = binary_mask.numpy().astype(np.uint8)
    total_pixels = mask_np.size
    lesion_pixels = int(mask_np.sum())
    area_ratio = lesion_pixels / total_pixels

    result: Dict = {
        "area_ratio": round(area_ratio, 6),
        "centroid": None,
        "bounding_box": None,
        "lesion_present": lesion_pixels > 0,
    }

    if lesion_pixels == 0:
        return result

    contours, _ = cv2.findContours(
        mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest = max(contours, key=cv2.contourArea)

    M = cv2.moments(largest)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        result["centroid"] = (cx, cy)

    x, y, w, h = cv2.boundingRect(largest)
    result["bounding_box"] = {"x": x, "y": y, "width": w, "height": h}

    return result
