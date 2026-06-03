"""
pipeline/recognize.py
---------------------
VehicleRecognizer: top-level orchestrator that wires together
  VehicleDetector → EmbeddingExtractor → VehicleDatabase

Handles single images, video files, and webcam streams.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import VehicleDatabase
from pipeline.detect import VehicleDetector
from pipeline.embed import EmbeddingExtractor

logger = logging.getLogger(__name__)


class VehicleRecognizer:

    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Config not found at {cfg_path}. Run from the repo root."
            )
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.detector  = VehicleDetector(
            model_path=self.cfg["detector"]["model_path"],
            confidence_threshold=self.cfg["detector"]["confidence_threshold"],
            vehicle_classes=self.cfg["detector"]["vehicle_classes"],
            device=self.cfg["detector"]["device"],
        )
        self.embedder  = EmbeddingExtractor(self.cfg)
        self.database  = VehicleDatabase(
            index_path=self.cfg["database"]["index_path"],
            metadata_path=self.cfg["database"]["metadata_path"],
            embedding_dim=self.cfg["model"]["embedding_dim"],
        )
        logger.info(
            "VehicleRecognizer ready | DB entries: %d", len(self.database)
        )

    # ── Single image ──────────────────────────────────────────────────────────
    def run_image(self, image_path: str) -> List[Dict]:
        """
        Run the full pipeline on a single image file.

        Returns
        -------
        List of result dicts, one per detected vehicle:
        {
            "bbox": [x1, y1, x2, y2],
            "confidence": float,      # detector confidence
            "make": str,
            "model": str,
            "year": str,
            "similarity_score": float,
            "rank": int,
        }
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        return self._process_frame(image)

    # ── Video file ────────────────────────────────────────────────────────────
    def run_video(
        self,
        video_path: str,
        output_path: str,
        show_preview: bool = False,
    ) -> None:
        """
        Process a video file frame by frame and write annotated output.

        Parameters
        ----------
        video_path   : input video path
        output_path  : where to write the annotated MP4
        show_preview : if True, display frames in a window (requires display)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))

        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self._process_frame(frame)
            annotated = self._draw_results(frame, results)
            writer.write(annotated)

            if show_preview:
                cv2.imshow("Vehicle Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - t0
                fps = frame_idx / elapsed
                pct = (frame_idx / total * 100) if total > 0 else 0
                print(f"  Frame {frame_idx}/{total} ({pct:.1f}%) | {fps:.1f} FPS")

        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        elapsed = time.time() - t0
        avg_fps = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\nVideo processing complete: {frame_idx} frames at {avg_fps:.1f} FPS")
        print(f"Output saved to: {output_path}")

    # ── Webcam ────────────────────────────────────────────────────────────────
    def run_webcam(self, camera_id: int = 0) -> None:
        """
        Run real-time recognition from a webcam. Press Q to quit.

        Parameters
        ----------
        camera_id : OpenCV camera index (default 0)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        print("Webcam started. Press Q to quit.")
        frame_count = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame — stopping.")
                break

            results = self._process_frame(frame)
            annotated = self._draw_results(frame, results)

            frame_count += 1
            elapsed = time.time() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            )

            cv2.imshow("Vehicle Recognition (Q to quit)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ── Visualise ─────────────────────────────────────────────────────────────
    def visualize(
        self,
        image_path: str,
        results: List[Dict],
        output_path: Optional[str] = None,
    ) -> str:
        """
        Draw bounding boxes and labels on an image and save it.

        Returns the path to the saved annotated image.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        annotated = self._draw_results(image, results)

        if output_path is None:
            stem = Path(image_path).stem
            output_path = f"assets/{stem}_annotated.jpg"

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, annotated)
        logger.info("Annotated image saved to %s", output_path)
        return output_path

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray) -> List[Dict]:
        """Run detect → embed → query on a single BGR frame."""
        detections = self.detector.detect(frame)
        if not detections:
            return []

        crops = self.detector.crop_vehicles(frame, detections)
        if not crops:
            return []

        embeddings = self.embedder.extract(crops)

        results = []
        for det, emb in zip(detections, embeddings):
            matches = self.database.query(
                emb.reshape(1, -1),
                top_k=self.cfg["database"]["top_k"],
                threshold=self.cfg["database"]["similarity_threshold"],
            )
            best = matches[0]
            results.append({
                "bbox":             det["bbox"],
                "confidence":       det["confidence"],
                "make":             best.get("make", "unknown"),
                "model":            best.get("model", "unknown"),
                "year":             best.get("year", ""),
                "similarity_score": best.get("score", 0.0),
                "rank":             best.get("rank", 1),
            })

        return results

    def _draw_results(
        self, frame: np.ndarray, results: List[Dict]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of the frame."""
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            make  = r["make"]
            model = r["model"]
            score = r["similarity_score"]
            label_top = f"{make} {model}"
            label_bot = f"score: {score:.2f}"

            color = (0, 200, 0) if make != "unknown" else (0, 0, 200)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Background pill for readability
            (tw, th), _ = cv2.getTextSize(
                label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                out,
                (x1, max(0, y1 - th - 24)),
                (x1 + tw + 6, y1),
                color, -1,
            )
            cv2.putText(
                out, label_top,
                (x1 + 3, max(th, y1 - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )
            cv2.putText(
                out, label_bot,
                (x1 + 3, max(th + 14, y1 - 1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
        return out


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== VehicleRecognizer smoke test ===")

    recognizer = VehicleRecognizer("configs/config.yaml")
    print(f"Recognizer initialised | DB size: {len(recognizer.database)}")

    # Create a tiny dummy image and run the pipeline (no detections expected)
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        tmp_path = f.name
    cv2.imwrite(tmp_path, dummy_bgr)

    results = recognizer.run_image(tmp_path)
    os.unlink(tmp_path)
    print(f"Results on blank image: {len(results)}  (expected 0)")
    print("VehicleRecognizer smoke test PASSED")
