"""
pipeline/detect.py
------------------
VehicleDetector: detects and crops vehicles from images using YOLOv8.

COCO vehicle classes used:
  2 = car
  5 = bus
  7 = truck
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Parameters
    ----------
    model_path         : path to YOLO weights file (e.g. 'yolov8n.pt').
                         If the file does not exist, ultralytics will
                         automatically download it on first run.
    confidence_threshold : minimum detection confidence (0–1)
    vehicle_classes    : list of COCO class IDs to keep
    device             : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        vehicle_classes: Optional[List[int]] = None,
        device: str = "cpu",
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required.\n"
                "Install with: pip install ultralytics"
            )

        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = vehicle_classes or [2, 5, 7]
        self.device = device

        # ultralytics auto-downloads weights if path doesn't exist
        self.model = YOLO(model_path)
        logger.info(
            "VehicleDetector loaded: model=%s, device=%s, classes=%s",
            model_path, device, self.vehicle_classes,
        )

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect vehicles in an image.

        Parameters
        ----------
        image : BGR numpy array (OpenCV format), shape (H, W, 3)

        Returns
        -------
        List of dicts: {"bbox": [x1,y1,x2,y2], "confidence": float, "class_id": int}
        """
        h, w = image.shape[:2]
        results = self.model(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Dict] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in self.vehicle_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Clip to image bounds
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": class_id,
                    })

        logger.debug("Detected %d vehicles", len(detections))
        return detections

    def crop_vehicles(
        self,
        image: np.ndarray,
        detections: List[Dict],
        padding: float = 0.10,
    ) -> List[Image.Image]:
        """
        Crop detected vehicle regions from the image.

        Parameters
        ----------
        image      : BGR numpy array
        detections : output of detect()
        padding    : fractional padding added around each bbox (default 10%)

        Returns
        -------
        List of RGB PIL Images, one per detected vehicle.
        """
        h, w = image.shape[:2]
        crops: List[Image.Image] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            pad_x = int(bw * padding)
            pad_y = int(bh * padding)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop_bgr = image[cy1:cy2, cx1:cx2]
            if crop_bgr.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crops.append(Image.fromarray(crop_rgb))

        return crops


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== VehicleDetector smoke test ===")

    # Create a blank dummy image (no actual detections expected)
    dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = VehicleDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.4,
        device="cpu",
    )
    detections = detector.detect(dummy_bgr)
    print(f"Detections on blank image: {len(detections)}  (expected 0)")

    crops = detector.crop_vehicles(dummy_bgr, detections)
    print(f"Crops returned: {len(crops)}")

    print("VehicleDetector smoke test PASSED")
