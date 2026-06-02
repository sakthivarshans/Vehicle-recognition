"""
database/build_db.py
--------------------
Build, enrol, list, or manage the vehicle reference database.

Usage
-----
# Build full database from Stanford Cars training set
python database/build_db.py --mode build

# Enrol a single new vehicle (no retraining!)
python database/build_db.py --mode enrol \
    --make Toyota --model Supra --year 2020 \
    --images ./my_supra_photos/

# List all enrolled vehicles
python database/build_db.py --mode list

# Remove a vehicle
python database/build_db.py --mode remove --make Toyota --model Supra
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

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


# ── Mode: build ───────────────────────────────────────────────────────────────
def build_from_dataset(cfg: dict) -> None:
    """Embed all Stanford Cars training images and populate the database."""
    from tqdm import tqdm

    from database.db_manager import VehicleDatabase
    from train.dataset import StanfordCarsDataset, holdout_split
    from train.model import EmbeddingModel

    dataset_path = cfg["training"]["dataset_path"]
    if not Path(dataset_path).exists():
        logger.error(
            "Dataset not found at '%s'.\n"
            "Download Stanford Cars and place it at data/stanford_cars/",
            dataset_path,
        )
        sys.exit(1)

    device_str = cfg["detector"]["device"]
    device = torch.device("cuda" if device_str == "cuda" and
                          torch.cuda.is_available() else "cpu")

    # Load model
    model = EmbeddingModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    weights_path = Path(cfg["model"]["weights_path"])
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
        logger.info("Loaded trained weights from %s", weights_path)
    else:
        logger.warning(
            "No trained weights at %s — using DINOv2 zero-shot features.",
            weights_path,
        )
    model.eval()

    # Load dataset (training split only — holdouts excluded)
    full_dataset = StanfordCarsDataset(dataset_path, split="train")
    train_subset, _ = holdout_split(
        full_dataset,
        n_holdout_classes=cfg["training"]["holdout_classes"],
    )

    # Group indices by class
    class_to_indices: dict = {}
    for subset_idx in range(len(train_subset)):
        real_idx = train_subset.indices[subset_idx]
        _, label = full_dataset.samples[real_idx]
        class_to_indices.setdefault(label, []).append(real_idx)

    db = VehicleDatabase(
        index_path=cfg["database"]["index_path"],
        metadata_path=cfg["database"]["metadata_path"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )

    BATCH = 32
    total_enrolled = 0

    for class_id, indices in tqdm(class_to_indices.items(), desc="Building DB"):
        class_name = full_dataset.get_class_name(class_id)
        # Parse "2012 Toyota Corolla" → make=Toyota, model=Corolla, year=2012
        parts = class_name.split(" ", 1)
        year  = parts[0] if parts[0].isdigit() else ""
        rest  = parts[1] if len(parts) > 1 else class_name
        name_parts = rest.split(" ", 1)
        make  = name_parts[0]
        model_name = name_parts[1] if len(name_parts) > 1 else rest

        # Embed images in batches
        all_embs = []
        for i in range(0, len(indices), BATCH):
            batch_indices = indices[i:i + BATCH]
            batch_imgs = []
            for idx in batch_indices:
                img_path, _ = full_dataset.samples[idx]
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_imgs.append(img)
                except Exception as e:
                    logger.warning("Could not load %s: %s", img_path, e)

            if not batch_imgs:
                continue

            from train.model import IMAGENET_MEAN, IMAGENET_STD
            from torchvision import transforms
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
            tensors = torch.stack([preprocess(im) for im in batch_imgs]).to(device)
            with torch.no_grad():
                embs = model(tensors).cpu().numpy()
            all_embs.append(embs)

        if all_embs:
            all_embs_np = np.concatenate(all_embs, axis=0)
            db.enrol(make, model_name, year, all_embs_np)
            total_enrolled += 1

    print(
        f"\nDatabase built with {total_enrolled} vehicles, "
        f"{db._index.ntotal} total embeddings."
    )


# ── Mode: enrol ───────────────────────────────────────────────────────────────
def enrol_vehicle(cfg: dict, args: argparse.Namespace) -> None:
    """Enrol a single new vehicle from a folder of images."""
    import torch
    from PIL import Image

    from database.db_manager import VehicleDatabase
    from train.model import EmbeddingModel, IMAGENET_MEAN, IMAGENET_STD
    from torchvision import transforms

    if not all([args.make, args.model, args.year, args.images]):
        logger.error(
            "--mode enrol requires --make, --model, --year, --images"
        )
        sys.exit(1)

    images_path = Path(args.images)
    if not images_path.exists():
        raise FileNotFoundError(f"Images folder not found: {images_path}")

    img_files = list(images_path.glob("*.jpg")) + \
                list(images_path.glob("*.jpeg")) + \
                list(images_path.glob("*.png"))

    if not img_files:
        logger.error("No images found in %s", images_path)
        sys.exit(1)

    device_str = cfg["detector"]["device"]
    device = torch.device("cuda" if device_str == "cuda" and
                          torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
    ).to(device)

    weights_path = Path(cfg["model"]["weights_path"])
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        logger.warning("No trained weights — using zero-shot features.")
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    embs = []
    for img_file in img_files:
        try:
            img = Image.open(img_file).convert("RGB")
            t = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(t).cpu().numpy()
            embs.append(emb)
        except Exception as e:
            logger.warning("Skipping %s: %s", img_file.name, e)

    if not embs:
        logger.error("No valid images could be embedded.")
        sys.exit(1)

    embs_np = np.concatenate(embs, axis=0)

    db = VehicleDatabase(
        index_path=cfg["database"]["index_path"],
        metadata_path=cfg["database"]["metadata_path"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )
    db.enrol(args.make, args.model, args.year, embs_np)
    print(
        f"Enrolled: {args.make} {args.model} ({args.year}) "
        f"with {len(embs)} reference images"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle reference database manager")
    parser.add_argument(
        "--mode", required=True,
        choices=["build", "enrol", "list", "remove"],
        help="Operation mode"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--make",   default=None, help="Vehicle manufacturer")
    parser.add_argument("--model",  default=None, help="Vehicle model name")
    parser.add_argument("--year",   default="",   help="Vehicle year / range")
    parser.add_argument("--images", default=None, help="Path to reference images folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args.config)

    from database.db_manager import VehicleDatabase

    if args.mode == "build":
        build_from_dataset(cfg)

    elif args.mode == "enrol":
        enrol_vehicle(cfg, args)

    elif args.mode == "list":
        db = VehicleDatabase(
            index_path=cfg["database"]["index_path"],
            metadata_path=cfg["database"]["metadata_path"],
            embedding_dim=cfg["model"]["embedding_dim"],
        )
        db.list_vehicles()

    elif args.mode == "remove":
        if not args.make or not args.model:
            logger.error("--mode remove requires --make and --model")
            sys.exit(1)
        db = VehicleDatabase(
            index_path=cfg["database"]["index_path"],
            metadata_path=cfg["database"]["metadata_path"],
            embedding_dim=cfg["model"]["embedding_dim"],
        )
        removed = db.remove(args.make, args.model)
        if removed:
            print(f"Removed {args.make} {args.model} from database.")
        else:
            print(f"{args.make} {args.model} not found in database.")
