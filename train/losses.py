"""
train/losses.py
---------------
Metric learning loss functions for training the embedding model.

Triplet Loss
    For each anchor image, find a "positive" (same vehicle model) and a
    "negative" (different vehicle model). The loss pushes the anchor closer
    to the positive and farther from the negative by at least `margin`.
    Result: same-model embeddings cluster together; different-model ones spread apart.

ArcFace Loss
    Adds an angular margin between classes in the embedding space, making the
    decision boundary much stricter than a plain softmax classifier.
    Generally gives slightly better accuracy than triplet loss when you have
    a large, well-labelled dataset.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── Triplet Loss ──────────────────────────────────────────────────────────────
class TripletLoss(nn.Module):
    """
    Triplet loss with hard-negative mining via MultiSimilarityMiner.

    Hard mining selects the most informative triplets in each batch —
    the positives that are farthest apart and the negatives that are
    closest together — which makes training much more efficient than
    random triplet sampling.
    """

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        try:
            from pytorch_metric_learning import losses, miners
        except ImportError:
            raise ImportError(
                "pytorch-metric-learning is required.\n"
                "Install it with: pip install pytorch-metric-learning"
            )

        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.miner   = miners.MultiSimilarityMiner()
        logger.info("TripletLoss initialised (margin=%.2f)", margin)

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        hard_pairs = self.miner(embeddings, labels)
        return self.loss_fn(embeddings, labels, hard_pairs)


# ── ArcFace Loss ──────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin) loss.

    Adds a fixed angular margin m to the angle between the embedding and
    the true class centre in the embedding space.  This creates a stricter
    margin than triplet loss and is state-of-the-art for face/vehicle
    recognition tasks.

    Requires num_classes to be known at construction time.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        margin: float = 0.5,
        scale: float = 64.0,
    ) -> None:
        super().__init__()
        try:
            from pytorch_metric_learning import losses
        except ImportError:
            raise ImportError(
                "pytorch-metric-learning is required.\n"
                "Install it with: pip install pytorch-metric-learning"
            )

        self.loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=margin,
            scale=scale,
        )
        logger.info(
            "ArcFaceLoss initialised (classes=%d, margin=%.2f, scale=%.1f)",
            num_classes, margin, scale,
        )

    def forward(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return self.loss_fn(embeddings, labels)

    def parameters(self, recurse: bool = True):
        """ArcFace has learnable class centres — expose them to the optimiser."""
        return self.loss_fn.parameters(recurse=recurse)


# ── Factory ───────────────────────────────────────────────────────────────────
def build_loss(
    loss_type: str,
    config: dict,
    num_classes: int = 196,
) -> Tuple[nn.Module, object]:
    """
    Factory function.

    Parameters
    ----------
    loss_type   : 'triplet' or 'arcface'
    config      : the full config dict (reads training.margin, model.embedding_dim)
    num_classes : number of vehicle classes (needed for ArcFace only)

    Returns
    -------
    (loss_fn, miner)
        miner is None for ArcFace (it does not use explicit mining).
    """
    margin        = config["training"]["margin"]
    embedding_dim = config["model"]["embedding_dim"]

    if loss_type == "triplet":
        loss_fn = TripletLoss(margin=margin)
        # The miner is embedded inside TripletLoss; return None externally
        return loss_fn, None

    elif loss_type == "arcface":
        loss_fn = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            margin=margin,
        )
        return loss_fn, None

    else:
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Choose 'triplet' or 'arcface'."
        )


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import torch

    print("=== losses.py smoke test ===")
    B, D = 16, 256
    embeddings = torch.randn(B, D)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    labels = torch.randint(0, 10, (B,))

    dummy_config = {
        "training": {"margin": 0.3},
        "model":    {"embedding_dim": D},
    }

    # Triplet
    triplet_loss, _ = build_loss("triplet", dummy_config)
    loss_val = triplet_loss(embeddings, labels)
    print(f"Triplet loss value : {loss_val.item():.4f}")

    # ArcFace
    arcface_loss, _ = build_loss("arcface", dummy_config, num_classes=10)
    loss_val = arcface_loss(embeddings, labels)
    print(f"ArcFace loss value : {loss_val.item():.4f}")

    print("Losses smoke test PASSED")
