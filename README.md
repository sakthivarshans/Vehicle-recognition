# Vehicle Make & Model Recognition

A **recognition pipeline** (not classification) that identifies vehicle make and model by comparing embedding vectors against a reference database — so new vehicles can be added without retraining the model.

---

## How It Works

```
Input Image / Video
       ↓
 Vehicle Detection   (YOLOv8 → bounding boxes)
       ↓
 Embedding Model     (DINOv2 / ResNet-50 → 256-d feature vector)
       ↓
 FAISS Database      (cosine similarity search)
       ↓
 Output: Make, Model, Year, Similarity Score
```

**Why recognition, not classification?**
A classifier requires retraining every time a new car model is added.
A recognition model produces an embedding and compares it to a database — add a new vehicle by enrolling reference images only. No retraining ever needed.

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/vehicle-recognition.git
cd vehicle-recognition

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset Download

Download the Stanford Cars dataset and place it at `data/stanford_cars/`:

```
data/stanford_cars/
    cars_train/
    cars_test/
    devkit/
        cars_meta.mat
        cars_train_annos.mat
        cars_test_annos_withlabels.mat
```

**Easiest source:** https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder

---

## Quick Start — Zero-Shot (No Training Required)

You can run the full pipeline immediately using DINOv2 zero-shot features — no GPU or training needed.

**Step 1: Build the reference database**
```bash
python database/build_db.py --mode build
```

**Step 2: Run inference on an image**
```bash
python pipeline/infer.py --input your_car.jpg --visualize
```

That's it. The annotated image will be saved to `assets/`.

---

## Training (For Best Accuracy)

Run on Kaggle (free T4 GPU) or any CUDA machine:

```bash
# Default: DINOv2 backbone + triplet loss
python train/train_embedding.py

# ResNet-50 + ArcFace loss (often higher accuracy)
python train/train_embedding.py --backbone resnet50 --loss arcface

# EfficientNet-B3
python train/train_embedding.py --backbone efficientnet_b3 --loss triplet
```

Training automatically rebuilds the database when it finishes.

After training on Kaggle, download `models/embedding.pth` and place it in the `models/` folder locally.

---

## Running Inference

```bash
# Single image
python pipeline/infer.py --input car.jpg --visualize

# Video file
python pipeline/infer.py --input traffic.mp4 --mode video --output result.mp4

# Live webcam
python pipeline/infer.py --mode webcam

# Override thresholds
python pipeline/infer.py --input car.jpg --threshold 0.55 --top-k 3
```

---

## Adding a New Vehicle (No Retraining!)

```bash
# Enrol from a folder of reference images
python database/build_db.py --mode enrol \
    --make Toyota --model Supra --year 2020 \
    --images ./my_supra_photos/

# List all vehicles in the database
python database/build_db.py --mode list

# Remove a vehicle
python database/build_db.py --mode remove --make Toyota --model Supra
```

---

## Evaluation

```bash
# Top-1 / Top-5 accuracy on test set
python eval/evaluate.py

# Top-1 / Top-5 on holdout split
python eval/evaluate.py --split holdout

# Unseen-vehicle enrolment demo (critical reviewer test)
python eval/unseen_vehicle_demo.py
```

---

## Export to ONNX (for deployment)

```bash
python train/export_onnx.py
```

For TensorRT on Jetson:
```bash
trtexec --onnx=models/embedding.onnx --saveEngine=models/embedding.trt --fp16
```

---

## Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Backbone | DINOv2 ViT-B/14 (zero-shot) or ResNet-50 (fine-tuned) | DINOv2 gives strong features out-of-the-box; ResNet-50 + metric learning gives best accuracy |
| Projection head | FC(→512) → BN → ReLU → FC(→256) → L2-norm | Projects to compact embedding space; L2-norm enables cosine similarity |
| Loss | Triplet loss with hard-negative mining | Pulls same-model embeddings together, pushes different ones apart |
| Database | FAISS IndexFlatIP | Exact cosine similarity search; supports instant enrolment |
| Detector | YOLOv8n | Fast, accurate, pre-trained on COCO vehicles |

---

## Results

| Backbone | Loss | Top-1 Acc | Top-5 Acc |
|----------|------|-----------|-----------|
| DINOv2 (zero-shot) | — | TBD | TBD |
| ResNet-50 | Triplet | TBD | TBD |
| ResNet-50 | ArcFace | TBD | TBD |

*Fill in after running `python eval/evaluate.py`*

---

## Project Structure

```
vehicle-recognition/
├── configs/config.yaml         ← all hyperparameters and paths
├── train/
│   ├── model.py                ← EmbeddingModel (backbone + projection head)
│   ├── dataset.py              ← StanfordCarsDataset + holdout_split
│   ├── losses.py               ← TripletLoss, ArcFaceLoss
│   ├── train_embedding.py      ← main training script
│   └── export_onnx.py          ← ONNX export + verification
├── database/
│   ├── db_manager.py           ← VehicleDatabase (FAISS-backed)
│   └── build_db.py             ← CLI for build / enrol / list / remove
├── pipeline/
│   ├── detect.py               ← VehicleDetector (YOLOv8)
│   ├── embed.py                ← EmbeddingExtractor
│   ├── recognize.py            ← VehicleRecognizer (full pipeline)
│   └── infer.py                ← CLI entry point
├── eval/
│   ├── evaluate.py             ← Top-1 / Top-5 accuracy
│   └── unseen_vehicle_demo.py  ← holdout enrolment demo
└── design_writeup.md           ← architecture decisions
```
