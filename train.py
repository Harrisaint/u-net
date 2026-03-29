"""
Training script for the U-Net segmentation model.

Usage:
    python train.py                          # defaults
    python train.py --epochs 50 --lr 3e-4    # override hyper-params
    python train.py --resume checkpoints/best.pth
"""

import argparse
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, random_split

from dataset import UltrasoundDataset, IMG_SIZE
from device import DEVICE
from model import build_unet

CHECKPOINT_DIR = Path("checkpoints")


# ------------------------------------------------------------------
# DataLoader helpers
# ------------------------------------------------------------------

def _build_loaders(
    data_dir: str,
    batch_size: int,
    val_split: float,
    img_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Create train / val DataLoaders with a random split."""
    full_ds = UltrasoundDataset(root_dir=data_dir, img_size=img_size)
    n_val = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    pin = DEVICE.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin,
    )
    return train_loader, val_loader


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------

def _dice_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute mean Dice coefficient for a batch (from raw logits)."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ------------------------------------------------------------------
# Training & validation loops
# ------------------------------------------------------------------

def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Run one training epoch.  Returns ``(avg_loss, avg_dice)``."""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    n_samples = 0

    for images, masks, _ in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_dice += _dice_score(logits, masks) * bs
        n_samples += bs

    return running_loss / n_samples, running_dice / n_samples


@torch.no_grad()
def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
) -> tuple[float, float]:
    """Run one validation pass.  Returns ``(avg_loss, avg_dice)``."""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    n_samples = 0

    for images, masks, _ in loader:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, masks)

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_dice += _dice_score(logits, masks) * bs
        n_samples += bs

    return running_loss / n_samples, running_dice / n_samples


# ------------------------------------------------------------------
# Main training driver
# ------------------------------------------------------------------

def train(
    data_dir: str = "data",
    epochs: int = 25,
    batch_size: int = 8,
    lr: float = 1e-4,
    val_split: float = 0.2,
    img_size: int = IMG_SIZE,
    resume: str | None = None,
    patience: int = 7,
) -> None:
    """End-to-end training routine with validation, checkpointing, and
    early stopping."""

    print(f"[device] Using {DEVICE} for training")

    # ---- data ---------------------------------------------------
    train_loader, val_loader = _build_loaders(
        data_dir, batch_size, val_split, img_size
    )
    print(
        f"[data]   {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val samples"
    )

    # ---- model --------------------------------------------------
    model = build_unet()
    if resume:
        state = torch.load(resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
        print(f"[model]  Resumed weights from {resume}")
    model.to(DEVICE)

    # ---- optimiser & loss ---------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = smp.losses.DiceLoss(mode="binary", from_logits=True)

    # ---- training loop ------------------------------------------
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_dice = _train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_dice = _validate(model, val_loader, criterion)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"train_loss={train_loss:.4f}  train_dice={train_dice:.4f}  |  "
            f"val_loss={val_loss:.4f}  val_dice={val_dice:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # ---- checkpointing & early stopping ---------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pth")
            print(f"         ✓ saved best checkpoint (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping — no improvement for {patience} epochs.")
                break

    torch.save(model.state_dict(), CHECKPOINT_DIR / "last.pth")
    print(f"Training complete.  Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {CHECKPOINT_DIR.resolve()}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net on ultrasound data")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to .pth checkpoint to resume from")
    p.add_argument("--patience", type=int, default=7,
                   help="Early-stopping patience (epochs)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        img_size=args.img_size,
        resume=args.resume,
        patience=args.patience,
    )
