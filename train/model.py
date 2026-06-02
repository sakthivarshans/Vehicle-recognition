"""
train/model.py
--------------
EmbeddingModel: produces L2-normalised feature vectors for vehicle crops.
Supports three backbones: DINOv2 (zero-shot), ResNet-50, EfficientNet-B3.

The projection head maps backbone features → fixed-size embedding that is
compared via cosine similarity against the reference database.
"""

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Backbone output dimensions
BACKBONE_DIMS = {
    "dinov2_vitb14": 768,
    "resnet50": 2048,
    "efficientnet_b3": 1536,
}

# ImageNet normalisation (used by all backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class ProjectionHead(nn.Module):
    """Maps backbone features to a compact, L2-normalised embedding vector."""

    def __init__(self, in_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return F.normalize(x, p=2, dim=1)  # L2 normalise


class EmbeddingModel(nn.Module):
    """
    Vehicle embedding model.

    backbone  : one of 'dinov2_vitb14' | 'resnet50' | 'efficientnet_b3'
    embedding_dim : size of the output feature vector (default 256)
    pretrained    : load ImageNet / self-supervised weights for the backbone
    """

    def __init__(
        self,
        backbone: str = "dinov2_vitb14",
        embedding_dim: int = 256,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if backbone not in BACKBONE_DIMS:
            raise ValueError(
                f"Unknown backbone '{backbone}'. "
                f"Choose from {list(BACKBONE_DIMS.keys())}"
            )

        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        in_dim = BACKBONE_DIMS[backbone]

        # ── Build backbone ──────────────────────────────────────────────
        if backbone == "dinov2_vitb14":
            logger.info("Loading DINOv2 ViT-B/14 from torch.hub …")
            self.backbone = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained
            )
            # Freeze DINOv2 weights by default (zero-shot / feature-extractor mode)
            for param in self.backbone.parameters():
                param.requires_grad = False

        elif backbone == "resnet50":
            import torchvision.models as tvm
            logger.info("Loading ResNet-50 …")
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            base = tvm.resnet50(weights=weights)
            # Remove the original FC layer
            self.backbone = nn.Sequential(*list(base.children())[:-1])

        elif backbone == "efficientnet_b3":
            import timm
            logger.info("Loading EfficientNet-B3 via timm …")
            self.backbone = timm.create_model(
                "efficientnet_b3", pretrained=pretrained, num_classes=0
            )

        # ── Projection head ──────────────────────────────────────────────
        self.head = ProjectionHead(in_dim, embedding_dim)

        # ── Preprocessing transform ──────────────────────────────────────
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        logger.info(
            "EmbeddingModel ready: backbone=%s, embedding_dim=%d",
            backbone, embedding_dim,
        )

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, embedding_dim) L2-normalised."""
        if self.backbone_name == "dinov2_vitb14":
            feats = self.backbone(x)                       # (B, 768)
        elif self.backbone_name == "resnet50":
            feats = self.backbone(x).flatten(1)            # (B, 2048)
        else:  # efficientnet_b3
            feats = self.backbone(x)                       # (B, 1536)

        return self.head(feats)                            # (B, emb_dim)

    # ── Convenience helper ───────────────────────────────────────────────
    def get_embedding(
        self,
        pil_image: Image.Image,
        device: Union[str, torch.device] = "cpu",
    ) -> np.ndarray:
        """
        Embed a single PIL Image.

        Returns
        -------
        np.ndarray of shape (1, embedding_dim), float32, L2-normalised.
        """
        self.eval()
        tensor = self.preprocess(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = self.forward(tensor)
        return emb.cpu().numpy().astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"EmbeddingModel(backbone={self.backbone_name}, "
            f"embedding_dim={self.embedding_dim})"
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== EmbeddingModel smoke test ===")

    # Test with a dummy PIL image (no real weights required for shape check)
    dummy_img = Image.fromarray(
        np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    )

    for bb in ["resnet50", "efficientnet_b3"]:
        print(f"\nTesting backbone: {bb}")
        model = EmbeddingModel(backbone=bb, embedding_dim=256, pretrained=False)
        emb = model.get_embedding(dummy_img, device="cpu")
        print(f"  Output shape : {emb.shape}")
        print(f"  L2 norm      : {np.linalg.norm(emb):.4f}  (should be ~1.0)")
        print(f"  {model}")

    print("\nSmoke test PASSED")
