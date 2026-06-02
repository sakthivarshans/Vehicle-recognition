"""
pipeline/infer.py
-----------------
CLI entry point for vehicle recognition.

Usage
-----
# Recognise vehicles in an image
python pipeline/infer.py --input car.jpg --visualize

# Process a video file
python pipeline/infer.py --input traffic.mp4 --mode video --output result.mp4

# Live webcam
python pipeline/infer.py --mode webcam

# Override thresholds
python pipeline/infer.py --input car.jpg --threshold 0.55 --top-k 3
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,          # keep pipeline internals quiet in CLI
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Pretty table printer ──────────────────────────────────────────────────────
def print_results_table(results: list, elapsed: float) -> None:
    if not results:
        print("\n  No vehicles detected in the image.\n")
        return

    w_make  = max(10, max(len(r["make"])  for r in results) + 2)
    w_model = max(12, max(len(r["model"]) for r in results) + 2)
    w_year  = 8
    w_score = 10

    total = w_make + w_model + w_year + w_score + 18
    line  = "─" * total
    print(f"\n┌{line}┐")
    print(
        f"│  {'#':<4}{'Make':<{w_make}}{'Model':<{w_model}}"
        f"{'Year':<{w_year}}{'Score':<{w_score}}{'Det.Conf':<10}  │"
    )
    print(f"├{line}┤")
    for i, r in enumerate(results, start=1):
        print(
            f"│  {i:<4}"
            f"{r['make']:<{w_make}}"
            f"{r['model']:<{w_model}}"
            f"{r.get('year',''):<{w_year}}"
            f"{r['similarity_score']:<{w_score}.3f}"
            f"{r['confidence']:<10.3f}  │"
        )
    print(f"└{line}┘")
    print(f"  Inference time: {elapsed*1000:.1f} ms\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config not found at {cfg_path}. Run from the repo root.")
        return 1

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.threshold is not None:
        cfg["database"]["similarity_threshold"] = args.threshold
    if args.top_k is not None:
        cfg["database"]["top_k"] = args.top_k

    # Determine mode
    mode = args.mode
    if mode == "auto" and args.input:
        ext = Path(args.input).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            mode = "video"
        elif ext in IMAGE_EXTENSIONS:
            mode = "image"
        else:
            print(f"ERROR: Cannot determine mode from extension '{ext}'. "
                  "Use --mode image|video|webcam.")
            return 1

    # Import recognizer (lazy to keep startup fast if just --help)
    from pipeline.recognize import VehicleRecognizer
    try:
        recognizer = VehicleRecognizer(args.config)
    except Exception as e:
        print(f"ERROR: Failed to initialise pipeline: {e}")
        return 1

    # ── Image mode ────────────────────────────────────────────────────────────
    if mode == "image":
        if not args.input:
            print("ERROR: --input is required for image mode.")
            return 1

        print(f"\nProcessing: {args.input}")
        t0 = time.time()
        try:
            results = recognizer.run_image(args.input)
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            return 1
        elapsed = time.time() - t0

        print_results_table(results, elapsed)

        if args.visualize:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = args.output or f"assets/output_{ts}.jpg"
            saved = recognizer.visualize(args.input, results, out)
            print(f"Annotated image saved → {saved}")

    # ── Video mode ────────────────────────────────────────────────────────────
    elif mode == "video":
        if not args.input:
            print("ERROR: --input is required for video mode.")
            return 1

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = args.output or f"assets/output_{ts}.mp4"
        print(f"\nProcessing video: {args.input}")
        print(f"Output will be saved to: {out}\n")
        try:
            recognizer.run_video(args.input, out)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"ERROR: {e}")
            return 1

    # ── Webcam mode ───────────────────────────────────────────────────────────
    elif mode == "webcam":
        cam_id = int(args.input) if args.input and args.input.isdigit() else 0
        print(f"\nStarting webcam (camera {cam_id}). Press Q to quit.\n")
        try:
            recognizer.run_webcam(cam_id)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            return 1

    else:
        print(f"ERROR: Unknown mode '{mode}'. Choose from image | video | webcam.")
        return 1

    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Make & Model Recognition — inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/infer.py --input car.jpg --visualize
  python pipeline/infer.py --input traffic.mp4 --mode video --output out.mp4
  python pipeline/infer.py --mode webcam
  python pipeline/infer.py --input car.jpg --threshold 0.55 --top-k 3
        """,
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to image or video file (or camera ID for webcam)",
    )
    parser.add_argument(
        "--mode", default="auto",
        choices=["auto", "image", "video", "webcam"],
        help="Processing mode (default: auto-detect from file extension)",
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Config YAML path (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for annotated output file",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Save annotated image to assets/ (image mode only)",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        dest="top_k",
        help="Override top-k from config",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Override similarity threshold from config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main(parse_args()))
