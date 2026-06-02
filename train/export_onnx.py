"""
train/export_onnx.py
--------------------
Export the trained EmbeddingModel to ONNX format for deployment.

Usage
-----
python train/export_onnx.py
python train/export_onnx.py --config configs/config.yaml --output models/embedding.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train.model import EmbeddingModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def export(args: argparse.Namespace) -> None:
    # ── Load config ───────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    weights_path = Path(cfg["model"]["weights_path"])
    output_path  = Path(args.output or "models/embedding.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    embedding_dim = cfg["model"]["embedding_dim"]

    # ── Load model ────────────────────────────────────────────────────────
    model = EmbeddingModel(
        backbone=cfg["model"]["backbone"],
        embedding_dim=embedding_dim,
        pretrained=False,
    )

    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        logger.info("Loaded weights from %s", weights_path)
    else:
        logger.warning(
            "Weights not found at %s — exporting untrained model (for pipeline testing only).",
            weights_path,
        )

    model.eval()

    # ── Export ────────────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input":     {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
    )
    logger.info("ONNX model exported → %s", output_path)

    # ── Verify with onnxruntime ───────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output_path))
        dummy_np = dummy_input.numpy()
        outputs  = sess.run(None, {"input": dummy_np})
        out_shape = outputs[0].shape
        assert out_shape == (1, embedding_dim), (
            f"Unexpected output shape: {out_shape}"
        )
        print(f"\nONNX export verified. Output shape: {out_shape}")
        print(
            "\nFor TensorRT on Jetson:\n"
            "  trtexec --onnx=models/embedding.onnx "
            "--saveEngine=models/embedding.trt --fp16"
        )
    except ImportError:
        logger.warning(
            "onnxruntime not installed — skipping verification.\n"
            "Install with: pip install onnxruntime"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export embedding model to ONNX")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default=None, help="Output .onnx path")
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
