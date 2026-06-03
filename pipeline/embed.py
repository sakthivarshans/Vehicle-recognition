"""
pipeline/embed.py
-----------------
EmbeddingExtractor: wraps EmbeddingModel for batch inference in the pipeline.

Automatically falls back to DINOv2 zero-shot mode when no trained weights
are found, so the pipeline always produces usable output.
"""

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class EmbeddingExtractor:

    def __init__(self, config: dict) -> None:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from train.model import EmbeddingModel

        self.cfg          = config
        self.embedding_dim = config["model"]["embedding_dim"]
        self.device_str   = config["detector"]["device"]
        self.device       = self._resolve_device(self.device_str)

        self.model = EmbeddingModel(
            backbone=config["model"]["backbone"],
            embedding_dim=self.embedding_dim,
            pretrained=config["model"]["pretrained"],
        ).to(self.device)

        weights_path = Path(config["model"]["weights_path"])
        if weights_path.exists():
            self.model.load_state_dict(
                torch.load(weights_path, map_location=self.device)
            )
            logger.info("Loaded trained weights from %s", weights_path)
        else:
            logger.warning(
                "\n"
                "  No trained weights found at %s\n"
                "  Running DINOv2 zero-shot baseline.\n"
                "  For best accuracy, train with:\n"
                "    python train/train_embedding.py\n",
                weights_path,
            )

        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(
            "EmbeddingExtractor ready: backbone=%s, device=%s",
            config["model"]["backbone"], self.device,
        )

    # ── Core extraction ───────────────────────────────────────────────────────
    def extract(self, pil_images: List[Image.Image]) -> np.ndarray:

        if not pil_images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        tensors = torch.stack(
            [self.preprocess(img) for img in pil_images]
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.model(tensors)

        return embeddings.cpu().numpy().astype(np.float32)

    def extract_single(self, pil_image: Image.Image) -> np.ndarray:

        return self.extract([pil_image])

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if device_str == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if device_str not in ("cpu",):
            logger.warning(
                "Requested device '%s' not available — falling back to CPU.",
                device_str,
            )
        return torch.device("cpu")


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path
    import yaml

    logging.basicConfig(level=logging.INFO)
    print("=== EmbeddingExtractor smoke test ===")

    cfg_path = Path("configs/config.yaml")
    if not cfg_path.exists():
        print("Run from the repo root: python pipeline/embed.py")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override to resnet50 so smoke test doesn't need to download DINOv2
    cfg["model"]["backbone"] = "resnet50"
    cfg["model"]["pretrained"] = False

    extractor = EmbeddingExtractor(cfg)

    dummy_imgs = [
        Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        for _ in range(4)
    ]

    import numpy as np
    embs = extractor.extract(dummy_imgs)
    print(f"Batch output shape : {embs.shape}  (expected (4, {cfg['model']['embedding_dim']}))")
    norms = np.linalg.norm(embs, axis=1)
    print(f"L2 norms (all ~1.0): {norms.round(4)}")

    single = extractor.extract_single(dummy_imgs[0])
    print(f"Single output shape: {single.shape}  (expected (1, {cfg['model']['embedding_dim']}))")

    print("EmbeddingExtractor smoke test PASSED")
