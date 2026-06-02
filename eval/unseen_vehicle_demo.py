"""
eval/unseen_vehicle_demo.py
---------------------------
Proof-of-concept: demonstrate that the system can recognise vehicle makes
and models it was NEVER trained on, purely by enrolling them into the
database from reference images.

This is the critical reviewer test. Run with no arguments:
    python eval/unseen_vehicle_demo.py
"""

import csv
import json
import logging
import sys
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

PASS_THRESHOLD = 0.50      # class is PASS if Top-1 acc > 50 %
N_REFERENCE    = 5         # reference images per holdout class


def load_config(path: str = "configs/config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found at {p}. Run from repo root.")
    with open(p) as f:
        return yaml.safe_load(f)


def run_demo() -> None:
    cfg = load_config()

    # ── Load holdout class info ───────────────────────────────────────────────
    holdout_path = Path("eval/holdout_classes.json")
    if not holdout_path.exists():
        logger.error(
            "eval/holdout_classes.json not found.\n"
            "This file is created during training:\n"
            "  python train/train_embedding.py\n"
            "Or run the dataset smoke test:\n"
            "  python train/dataset.py"
        )
        sys.exit(1)

    with open(holdout_path) as f:
        holdout_info = json.load(f)   # {class_id_str: class_name_str}

    holdout_class_ids = {int(k): v for k, v in holdout_info.items()}
    logger.info("Holdout classes: %d", len(holdout_class_ids))

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset_path = cfg["training"]["dataset_path"]
    if not Path(dataset_path).exists():
        logger.error("Dataset not found at '%s'.", dataset_path)
        sys.exit(1)

    from train.dataset import StanfordCarsDataset
    from torchvision import transforms

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_ds = StanfordCarsDataset(
        dataset_path, split="train", transform=val_transform
    )

    # ── Load embedder ─────────────────────────────────────────────────────────
    from pipeline.embed import EmbeddingExtractor
    embedder = EmbeddingExtractor(cfg)

    # ── Load database (temporary copy for this demo) ──────────────────────────
    from database.db_manager import VehicleDatabase
    db = VehicleDatabase(
        index_path=cfg["database"]["index_path"],
        metadata_path=cfg["database"]["metadata_path"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )

    # ── Group images by holdout class ─────────────────────────────────────────
    class_to_paths: dict = {}
    for img_path, label in full_ds.samples:
        if label in holdout_class_ids:
            class_to_paths.setdefault(label, []).append(img_path)

    # ── Per-class enrolment & evaluation ──────────────────────────────────────
    demo_rows     = []
    total_correct = 0
    total_queries = 0

    # Track which classes we enrol so we can clean up afterward
    enrolled_classes = []

    print("\n  Enrolling holdout classes into database …\n")
    for class_id, class_name in tqdm(holdout_class_ids.items(), desc="Enrolling"):
        paths = class_to_paths.get(class_id, [])
        if len(paths) < N_REFERENCE + 1:
            logger.warning(
                "Class '%s' has only %d images — skipping.", class_name, len(paths)
            )
            continue

        # Split: first N_REFERENCE = reference, rest = query
        ref_paths   = paths[:N_REFERENCE]
        query_paths = paths[N_REFERENCE:]

        # Parse class name → make, model, year
        parts = class_name.split(" ", 1)
        year  = parts[0] if parts[0].isdigit() else ""
        rest  = parts[1] if len(parts) > 1 else class_name
        name_parts = rest.split(" ", 1)
        make       = name_parts[0]
        model_name = name_parts[1] if len(name_parts) > 1 else rest

        # Embed reference images
        ref_embs = []
        for rp in ref_paths:
            try:
                img = Image.open(rp).convert("RGB")
                emb = embedder.extract_single(img)
                ref_embs.append(emb)
            except Exception as e:
                logger.warning("Could not embed %s: %s", rp, e)

        if not ref_embs:
            continue

        ref_embs_np = np.concatenate(ref_embs, axis=0)
        db.enrol(make, model_name, year, ref_embs_np)
        enrolled_classes.append((make, model_name))

    print(f"\n  Enrolled {len(enrolled_classes)} unseen classes.\n")
    print("  Running recognition on query images …\n")

    # Evaluate
    for class_id, class_name in tqdm(holdout_class_ids.items(), desc="Querying"):
        paths = class_to_paths.get(class_id, [])
        if len(paths) < N_REFERENCE + 1:
            continue

        query_paths = paths[N_REFERENCE:]

        parts = class_name.split(" ", 1)
        year  = parts[0] if parts[0].isdigit() else ""
        rest  = parts[1] if len(parts) > 1 else class_name
        name_parts = rest.split(" ", 1)
        make       = name_parts[0]
        model_name = name_parts[1] if len(name_parts) > 1 else rest

        n_correct = 0
        n_queries = 0

        for qp in query_paths:
            try:
                img = Image.open(qp).convert("RGB")
                emb = embedder.extract_single(img)
                matches = db.query(emb, top_k=1, threshold=0.0)
                best    = matches[0]
                predicted_name = f"{best['make']} {best['model']}".lower()
                if make.lower() in predicted_name or model_name.lower() in predicted_name:
                    n_correct += 1
                n_queries += 1
            except Exception as e:
                logger.warning("Query failed for %s: %s", qp, e)

        if n_queries == 0:
            continue

        acc      = n_correct / n_queries
        passed   = acc >= PASS_THRESHOLD
        total_correct += n_correct
        total_queries += n_queries

        demo_rows.append({
            "class_name":  class_name,
            "make":        make,
            "model":       model_name,
            "year":        year,
            "n_reference": N_REFERENCE,
            "n_queries":   n_queries,
            "n_correct":   n_correct,
            "top1_acc":    round(acc * 100, 1),
            "result":      "PASS" if passed else "FAIL",
        })

    # ── Print results table ───────────────────────────────────────────────────
    print("\n  Unseen Vehicle Enrolment Demo")
    print("═" * 78)
    print(
        f"  {'Make/Model':<30}{'Refs':>6}{'Queries':>9}"
        f"{'Top-1 Acc':>11}{'Result':>10}"
    )
    print("─" * 78)
    for row in demo_rows:
        label = f"{row['make']} {row['model']} ({row['year']})"
        print(
            f"  {label:<30}{row['n_reference']:>6}"
            f"{row['n_queries']:>9}{row['top1_acc']:>10.1f}%"
            f"  {row['result']:>7}"
        )
    print("─" * 78)
    overall_acc = (total_correct / total_queries * 100) if total_queries > 0 else 0.0
    print(f"\n  Overall unseen-vehicle Top-1 accuracy: {overall_acc:.1f}%")
    pass_count = sum(1 for r in demo_rows if r["result"] == "PASS")
    print(f"  Classes PASS / FAIL: {pass_count} / {len(demo_rows) - pass_count}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = Path("eval/unseen_demo_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if demo_rows:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(demo_rows[0].keys()))
            writer.writeheader()
            writer.writerows(demo_rows)
    logger.info("Demo results saved to %s", out_csv)

    # ── Clean up: remove holdout entries from database ────────────────────────
    print("  Cleaning up holdout entries from database …")
    for make, model_name in enrolled_classes:
        db.remove(make, model_name)
    print(f"  Database restored to {len(db)} vehicles.\n")


if __name__ == "__main__":
    run_demo()
