"""
train/train_embedding.py
------------------------
Main training script for the vehicle embedding model.

Usage
-----
python train/train_embedding.py
python train/train_embedding.py --backbone resnet50 --loss arcface
python train/train_embedding.py --config configs/config.yaml --epochs 50
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# ── Ensure repo root is on the path ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.dataset import StanfordCarsDataset, holdout_split
from train.losses import build_loss
from train.model import EmbeddingModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. Run from the repo root."
        )
    with open(p) as f:
        return yaml.safe_load(f)


def get_device(cfg_device: str) -> torch.device:
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    logger.warning("GPU not available — falling back to CPU.")
    return torch.device("cpu")


# ── Training loop ─────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    # CLI overrides
    if args.backbone:
        cfg["model"]["backbone"] = args.backbone
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs

    device = get_device(cfg["detector"]["device"])
    logger.info("Device: %s", device)

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_path = cfg["training"]["dataset_path"]
    if not Path(dataset_path).exists():
        logger.error(
            "Dataset not found at '%s'.\n"
            "Download Stanford Cars and place it there:\n"
            "  data/stanford_cars/cars_train/\n"
            "  data/stanford_cars/devkit/",
            dataset_path,
        )
        sys.exit(1)

    full_dataset = StanfordCarsDataset(dataset_path, split="train")
    train_subset, _ = holdout_split(
        full_dataset,
        n_holdout_classes=cfg["training"]["holdout_classes"],
        save_path="eval/holdout_classes.json",
    )
    num_classes = len(full_dataset.class_names)
    logger.info("Training on %d samples, %d classes total", len(train_subset), num_classes)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = EmbeddingModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["training"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)
    logger.info("%s", model)

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_fn, _ = build_loss(args.loss, cfg, num_classes=num_classes)
    loss_fn = loss_fn.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────
    params = list(model.parameters())
    if hasattr(loss_fn, "parameters"):
        params += list(loss_fn.parameters())

    optimizer = optim.AdamW(
        params,
        lr=cfg["training"]["learning_rate"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["epochs"]
    )

    # ── Save directory ────────────────────────────────────────────────────
    save_path = Path(cfg["training"]["save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        lr_now   = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d | loss=%.4f | lr=%.2e | time=%.1fs",
            epoch, cfg["training"]["epochs"], avg_loss, lr_now, elapsed,
        )

        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            logger.info("  ✓ Best model saved → %s", save_path)

    # Save final model
    final_path = save_path.parent / (save_path.stem + "_final.pth")
    torch.save(model.state_dict(), final_path)
    logger.info("Final model saved → %s", final_path)

    # ── Auto-build database after training ────────────────────────────────
    logger.info("Building reference database from training embeddings …")
    subprocess.run(
        [sys.executable, "database/build_db.py", "--mode", "build"],
        check=True,
    )
    logger.info("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train vehicle embedding model"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)"
    )
    parser.add_argument(
        "--loss", default="triplet", choices=["triplet", "arcface"],
        help="Metric learning loss (default: triplet)"
    )
    parser.add_argument(
        "--backbone", default=None,
        choices=["dinov2_vitb14", "resnet50", "efficientnet_b3"],
        help="Override backbone from config"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of epochs from config"
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
