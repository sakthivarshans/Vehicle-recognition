"""
eval/evaluate.py
----------------
Evaluate recognition accuracy on the Stanford Cars test or holdout split.

Usage
-----
python eval/evaluate.py
python eval/evaluate.py --split holdout
python eval/evaluate.py --split test --config configs/config.yaml
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found at {p}. Run from repo root.")
    with open(p) as f:
        return yaml.safe_load(f)


def evaluate(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    from database.db_manager import VehicleDatabase
    from pipeline.embed import EmbeddingExtractor
    from train.dataset import StanfordCarsDataset, holdout_split
    from torchvision import transforms

    device_str = cfg["detector"]["device"]
    device = torch.device(
        "cuda" if device_str == "cuda" and torch.cuda.is_available() else "cpu"
    )

    # ── Embedder ──────────────────────────────────────────────────────────────
    embedder = EmbeddingExtractor(cfg)

    # ── Database ──────────────────────────────────────────────────────────────
    db = VehicleDatabase(
        index_path=cfg["database"]["index_path"],
        metadata_path=cfg["database"]["metadata_path"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )
    if len(db) == 0:
        logger.error(
            "Database is empty. Run first:\n"
            "  python database/build_db.py --mode build"
        )
        sys.exit(1)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_path = cfg["training"]["dataset_path"]
    if not Path(dataset_path).exists():
        logger.error("Dataset not found at '%s'.", dataset_path)
        sys.exit(1)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if args.split == "test":
        dataset = StanfordCarsDataset(
            dataset_path, split="test", transform=val_transform
        )
        samples = dataset.samples
        get_name = dataset.get_class_name
        logger.info("Evaluating on test split: %d samples", len(samples))

    else:  # holdout
        holdout_path = Path("eval/holdout_classes.json")
        if not holdout_path.exists():
            logger.error(
                "eval/holdout_classes.json not found.\n"
                "Run training first: python train/train_embedding.py"
            )
            sys.exit(1)
        with open(holdout_path) as f:
            holdout_info = json.load(f)
        holdout_class_ids = {int(k) for k in holdout_info.keys()}

        full_ds = StanfordCarsDataset(
            dataset_path, split="train", transform=val_transform
        )
        samples  = [
            (p, l) for p, l in full_ds.samples if l in holdout_class_ids
        ]
        get_name = full_ds.get_class_name
        logger.info(
            "Evaluating on holdout split: %d samples, %d classes",
            len(samples), len(holdout_class_ids),
        )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    top_k_values = cfg["evaluation"]["top_k_values"]
    max_k        = max(top_k_values)
    threshold    = cfg["database"]["similarity_threshold"]

    correct = {k: 0 for k in top_k_values}
    total   = 0
    rows    = []

    t0 = time.time()

    for img_path, true_label in tqdm(samples, desc="Evaluating"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning("Could not open %s: %s", img_path, e)
            continue

        emb = embedder.extract_single(pil_img)
        matches = db.query(emb, top_k=max_k, threshold=0.0)  # use 0 to get all k

        true_name = get_name(true_label)
        # Parse class name → make + model
        parts = true_name.split(" ", 1)
        true_make  = parts[0] if len(parts) > 1 else ""
        true_model = parts[1] if len(parts) > 1 else true_name

        pred_makes  = [m["make"]  for m in matches]
        pred_models = [m["model"] for m in matches]

        def _match(mk, mo):
            return mk.lower() in true_name.lower() or mo.lower() in true_name.lower()

        c_top = {k: False for k in top_k_values}
        for k in top_k_values:
            for mk, mo in zip(pred_makes[:k], pred_models[:k]):
                if _match(mk, mo):
                    c_top[k] = True
                    break

        for k in top_k_values:
            if c_top[k]:
                correct[k] += 1

        total += 1
        rows.append({
            "image_path":   str(img_path),
            "true_make":    true_make,
            "true_model":   true_model,
            "pred_make":    pred_makes[0] if pred_makes else "unknown",
            "pred_model":   pred_models[0] if pred_models else "unknown",
            "score":        matches[0]["score"] if matches else 0.0,
            "correct_top1": int(c_top[1]) if 1 in c_top else 0,
            "correct_top5": int(c_top[5]) if 5 in c_top else 0,
        })

    elapsed = time.time() - t0

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_path = Path(cfg["evaluation"]["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Per-image results saved to %s", results_path)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 42)
    print("  Evaluation Results")
    print("─" * 42)
    print(f"  Split          : {args.split}")
    print(f"  Total queries  : {total}")
    for k in top_k_values:
        acc = 100.0 * correct[k] / total if total > 0 else 0.0
        print(f"  Top-{k} accuracy : {acc:.1f}%")
    print(f"  Elapsed time   : {elapsed:.1f}s")
    print("═" * 42 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vehicle recognition accuracy")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--split", default="test", choices=["test", "holdout"],
        help="Which split to evaluate on (default: test)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
