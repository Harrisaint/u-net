"""
Custom PyTorch Dataset for breast ultrasound images with binary masks.

Diagnostic focus: **Benign** vs **Malignant** only (normal cases excluded).

Expected directory layout:
    data/
    ├── benign/
    │   ├── benign (1).png
    │   ├── benign (1)_mask.png
    │   └── ...
    └── malignant/
        ├── malignant (1).png
        ├── malignant (1)_mask.png
        └── ...
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ["benign", "malignant"]
IMG_SIZE = 256


# ------------------------------------------------------------------
# Augmentation presets
# ------------------------------------------------------------------

def get_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Augmentation pipeline for training (helps prevent overfitting)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5,
        ),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Deterministic resize only — no stochastic augmentation."""
    return A.Compose([
        A.Resize(img_size, img_size),
    ])


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class UltrasoundDataset(Dataset):
    """Pairs each original image with its corresponding ``_mask.png`` file.

    Parameters
    ----------
    root_dir : str | Path
        Path to the top-level ``data/`` folder.
    img_size : int
        Spatial size to resize every image/mask to (square).
    classes : list[str] | None
        Subdirectories to include.  Defaults to ``["benign", "malignant"]``.
    transform : albumentations.Compose | None
        Albumentations pipeline applied to both image and mask.
        If ``None``, images/masks are simply resized to ``img_size``.
    """

    def __init__(
        self,
        root_dir: str | Path,
        img_size: int = IMG_SIZE,
        classes: List[str] | None = None,
        transform: A.Compose | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.classes = classes or CLASSES
        self.transform = transform
        self.samples: List[Dict] = []

        self._discover_samples()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _discover_samples(self) -> None:
        """Walk each class subdirectory and pair images <-> masks."""
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.is_dir():
                continue

            mask_files = {p.name: p for p in cls_dir.glob("*_mask.png")}

            for mask_name, mask_path in sorted(mask_files.items()):
                img_name = mask_name.replace("_mask.png", ".png")
                img_path = cls_dir / img_name

                if not img_path.exists():
                    continue

                self.samples.append(
                    {
                        "image_path": str(img_path),
                        "mask_path": str(mask_path),
                        "class": cls,
                    }
                )

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        """Read an image as RGB (no resize — handled by transform)."""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _load_mask(path: str) -> np.ndarray:
        """Read a mask as grayscale and binarize (0 / 1)."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        return (mask > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Return ``(image_tensor, mask_tensor, class_label)``.

        * ``image_tensor``  – shape ``(3, H, W)``, float32 in ``[0, 1]``
        * ``mask_tensor``   – shape ``(1, H, W)``, float32 binary
        * ``class_label``   – ``"benign"`` or ``"malignant"``
        """
        sample = self.samples[idx]

        image = self._load_image(sample["image_path"])
        mask = self._load_mask(sample["mask_path"])

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = cv2.resize(
                image, (self.img_size, self.img_size),
                interpolation=cv2.INTER_LINEAR,
            )
            mask = cv2.resize(
                mask, (self.img_size, self.img_size),
                interpolation=cv2.INTER_NEAREST,
            )

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor, sample["class"]


def build_dataloader(
    root_dir: str | Path,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    img_size: int = IMG_SIZE,
    pin_memory: bool = False,
    transform: A.Compose | None = None,
) -> torch.utils.data.DataLoader:
    """Convenience wrapper that returns a ready-to-use DataLoader."""
    ds = UltrasoundDataset(
        root_dir=root_dir, img_size=img_size, transform=transform,
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
