"""
train/dataset.py
----------------
StanfordCarsDataset: PyTorch Dataset for the Stanford Cars dataset.

Download the dataset from:
  https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder
or the original:
  https://ai.stanford.edu/~jkrause/cars/car_dataset.html

Expected directory layout
--------------------------
data/stanford_cars/
    cars_train/          ← training images (jpg)
    cars_test/           ← test images (jpg)
    devkit/
        cars_train_annos.mat
        cars_test_annos_withlabels.mat
        cars_meta.mat
"""

import json
import logging
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms

logger = logging.getLogger(__name__)

# ── Transforms ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class StanfordCarsDataset(Dataset):
    """
    Stanford Cars dataset.

    Parameters
    ----------
    root       : path to the dataset root (contains cars_train/, devkit/, …)
    split      : 'train' or 'test'
    transform  : torchvision transform; defaults to train/val augmentation
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        self.root  = Path(root)
        self.split = split

        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got '{split}'")

        self.transform = transform or (
            get_train_transform() if split == "train" else get_val_transform()
        )

        # ── Load class names ─────────────────────────────────────────────
        meta_path = self.root / "devkit" / "cars_meta.mat"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"cars_meta.mat not found at {meta_path}.\n"
                "Make sure the devkit/ folder is inside your dataset root."
            )
        meta = scipy.io.loadmat(str(meta_path))
        self.class_names: List[str] = [
            str(name[0]) for name in meta["class_names"][0]
        ]

        # ── Load annotations ─────────────────────────────────────────────
        if split == "train":
            anno_path  = self.root / "devkit" / "cars_train_annos.mat"
            image_root = self.root / "cars_train"
        else:
            anno_path  = self.root / "devkit" / "cars_test_annos_withlabels.mat"
            image_root = self.root / "cars_test"

        if not anno_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {anno_path}\n"
                "For the test split you need cars_test_annos_withlabels.mat"
            )

        annos = scipy.io.loadmat(str(anno_path))["annotations"][0]

        self.samples: List[Tuple[Path, int]] = []
        for anno in annos:
            fname = str(anno["fname"][0])
            # Labels in .mat are 1-indexed → convert to 0-indexed
            label = int(anno["class"][0][0]) - 1
            img_path = image_root / fname
            if img_path.exists():
                self.samples.append((img_path, label))
            else:
                logger.warning("Image not found, skipping: %s", img_path)

        logger.info(
            "StanfordCarsDataset loaded: split=%s, samples=%d, classes=%d",
            split, len(self.samples), len(self.class_names),
        )

    # ── Dataset protocol ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_name(self, label: int) -> str:
        """Return the human-readable class name for a 0-indexed label."""
        return self.class_names[label]

    def get_labels(self) -> List[int]:
        """Return all labels as a flat list (used for stratified splitting)."""
        return [label for _, label in self.samples]


# ── Holdout split ─────────────────────────────────────────────────────────────
def holdout_split(
    dataset: StanfordCarsDataset,
    n_holdout_classes: int,
    seed: int = 42,
    save_path: Optional[str] = "eval/holdout_classes.json",
) -> Tuple[Subset, Subset]:
    """
    Split the dataset into a training subset and a holdout subset.

    The holdout subset contains ALL images of n_holdout_classes randomly
    selected classes.  The training subset contains NONE of those classes.

    Parameters
    ----------
    dataset           : StanfordCarsDataset instance
    n_holdout_classes : number of classes to withhold
    seed              : random seed for reproducibility
    save_path         : where to save the holdout class names as JSON

    Returns
    -------
    (train_subset, holdout_subset)
    """
    rng = random.Random(seed)
    all_classes = list(range(len(dataset.class_names)))
    holdout_class_ids = set(rng.sample(all_classes, n_holdout_classes))
    train_class_ids   = set(all_classes) - holdout_class_ids

    train_indices:   List[int] = []
    holdout_indices: List[int] = []

    for idx, (_, label) in enumerate(dataset.samples):
        if label in holdout_class_ids:
            holdout_indices.append(idx)
        else:
            train_indices.append(idx)

    logger.info(
        "Holdout split: %d train samples, %d holdout samples (%d classes held out)",
        len(train_indices), len(holdout_indices), n_holdout_classes,
    )

    # ── Persist holdout class names ──────────────────────────────────────
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        holdout_info = {
            str(cid): dataset.get_class_name(cid)
            for cid in sorted(holdout_class_ids)
        }
        with open(save_path, "w") as f:
            json.dump(holdout_info, f, indent=2)
        logger.info("Holdout class names saved to %s", save_path)

    return Subset(dataset, train_indices), Subset(dataset, holdout_indices)


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys

    dataset_path = "data/stanford_cars"
    if not Path(dataset_path).exists():
        print(
            "Stanford Cars dataset not found at data/stanford_cars/\n"
            "Download it and place it there, then re-run.\n"
            "Skipping live dataset test — transform smoke test only."
        )
        # Just verify transforms work
        dummy = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        t = get_train_transform()(dummy)
        print(f"Train transform output shape: {t.shape}")
        t = get_val_transform()(dummy)
        print(f"Val   transform output shape: {t.shape}")
        print("Transform smoke test PASSED")
        sys.exit(0)

    ds = StanfordCarsDataset(dataset_path, split="train")
    print(f"Train samples : {len(ds)}")
    print(f"Classes       : {len(ds.class_names)}")
    img, label = ds[0]
    print(f"Sample shape  : {img.shape}")
    print(f"Label         : {label} → {ds.get_class_name(label)}")

    train_sub, holdout_sub = holdout_split(ds, n_holdout_classes=10)
    print(f"Train subset  : {len(train_sub)}")
    print(f"Holdout subset: {len(holdout_sub)}")
    print("Dataset smoke test PASSED")
