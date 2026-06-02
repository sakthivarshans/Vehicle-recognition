# Design Write-up: Vehicle Make & Model Recognition Pipeline

---

## 1. Why Recognition, Not Classification

### The Open-Set vs Closed-Set Problem

A **classification model** maps inputs to a fixed set of known classes using a softmax layer. This is a *closed-set* system — the model can only predict classes it was trained on. If you want to add a new vehicle make or model, you must:

1. Collect labelled training data for the new class
2. Retrain (or fine-tune) the entire model
3. Redeploy it

This is unacceptable for a production vehicle database that needs to grow continuously.

A **recognition model** is an *open-set* system. It produces an embedding (a fixed-size feature vector) that encodes the visual identity of a vehicle. To identify a query vehicle:

1. Embed the query image → get its feature vector
2. Compare it against a database of known vehicle embeddings
3. Return the closest match above a confidence threshold

To add a new vehicle: embed a few reference images and store them. No retraining. No redeployment. The model stays frozen.

### Why a Softmax Classifier Fails Here

If the Stanford Cars dataset has 196 classes, a softmax classifier has a 196-neuron output layer. Adding a 197th class (say, a 2024 Toyota Crown) requires:
- Adding a new output neuron
- Re-initialising and retraining the output weights
- Risk of catastrophic forgetting of old classes

With an embedding model, the 197th class is handled entirely at the database layer — the neural network is never touched.

---

## 2. Embedding Model Architecture

### Backbone Choice

Three backbones are supported, each targeting a different use case:

**DINOv2 ViT-B/14** (default)
- Self-supervised Vision Transformer trained on 142M images
- Produces semantically rich 768-d CLS token features
- Zero-shot: works out-of-the-box without any vehicle-specific training
- Best choice when GPU access is limited or time is short
- Frozen during training (feature extractor only)

**ResNet-50** (recommended for best accuracy)
- Classic CNN backbone, pre-trained on ImageNet
- 2048-d global average pooled features
- Fine-tuned end-to-end with metric learning loss
- Best accuracy when training data and GPU are available

**EfficientNet-B3**
- More efficient than ResNet-50 with similar or better accuracy
- 1536-d features
- Good choice when deploying on edge hardware (fewer FLOPs)

### Projection Head

All backbones share the same projection head:

```
Linear(backbone_dim → 512)
BatchNorm1d(512)
ReLU
Linear(512 → 256)
L2 normalisation
```

**Why this design:**
- The two-layer MLP compresses high-dimensional backbone features into a compact 256-d space optimised for distance comparisons
- BatchNorm stabilises training and reduces sensitivity to learning rate
- L2 normalisation projects all embeddings onto a unit hypersphere, making cosine similarity equivalent to dot product — enabling fast FAISS inner product search

---

## 3. Metric Learning

### Triplet Loss (Plain English)

For each training step, we sample three images:
- **Anchor**: a car of model X
- **Positive**: another image of the same model X
- **Negative**: an image of a different model Y

The triplet loss says: the anchor should be at least `margin` closer to the positive than to the negative. Mathematically:

```
L = max(0, d(anchor, positive) − d(anchor, negative) + margin)
```

If the anchor is already closer to the positive by more than `margin`, the loss is zero (this triplet is "easy" and contributes nothing to training).

The result: in embedding space, images of the same car model cluster tightly together, while images of different models are pushed apart.

### Hard Negative Mining

Random triplet sampling is wasteful — most randomly selected triplets are already satisfied (easy triplets). Hard negative mining selects the *most informative* triplets per batch:
- **Hard positives**: same-class pairs that are farthest apart (the model is most confused about)
- **Hard negatives**: different-class pairs that are closest together (the model is most likely to confuse)

Training on hard triplets converges faster and produces better embeddings.

We use `pytorch-metric-learning`'s `MultiSimilarityMiner`, which implements an efficient batch-level hard mining strategy.

### ArcFace (Alternative)

ArcFace adds an angular margin directly to the angle between an embedding and its true class centre in the hypersphere. This creates a stricter, more uniform margin than triplet loss and is the state-of-the-art loss for recognition tasks. It generally outperforms triplet loss when training data is large and well-labelled.

---

## 4. Reference Database Design

### Why FAISS IndexFlatIP

**FAISS** (Facebook AI Similarity Search) is a library for efficient nearest-neighbour search over dense vectors.

`IndexFlatIP` performs **exact** inner product search. Since all embeddings are L2-normalised, inner product equals cosine similarity. "Exact" means it checks every entry in the database — for a database of a few thousand vehicles this is fast enough (sub-millisecond) and gives perfect recall.

For very large databases (100k+ entries), `IndexIVFFlat` with approximate search would be more appropriate, but this is overkill for the scale of this task.

### Confidence Threshold

Every query returns a similarity score in [0, 1]. If the best match falls below `similarity_threshold` (default 0.65), the system returns "unknown" rather than a wrong match. This handles:
- Vehicles not in the database
- Very unusual angles or lighting conditions
- Non-vehicle images that were mis-detected

### Mean-Pooling Multiple Reference Images

When enrolling a vehicle, we compute embeddings for each reference image and average them. This produces a single representative embedding that is more stable than any individual image — it captures the average visual appearance of the vehicle across different angles, lighting, and backgrounds.

---

## 5. Unseen Vehicle Handling

The key design requirement: **the model must never be retrained to recognise a new vehicle**.

The enrolment process:
1. Collect 5+ reference images of the new vehicle (different angles preferred)
2. Pass each through the frozen embedding model to get a 256-d vector
3. Average the vectors and L2-normalise → one representative embedding
4. Store (embedding, make, model, year) in the FAISS index

From this point, any query image of that vehicle will produce an embedding close to the enrolled one in the 256-d space, and will be correctly matched via cosine similarity search.

This works because the embedding model has learned a general notion of visual similarity between vehicles — not memorised specific classes. DINOv2 in particular was trained on hundreds of millions of images using self-supervised learning, giving it strong generalisation to unseen vehicle appearances.

---

## 6. Similarity Metric: Cosine vs Euclidean

After L2 normalisation, all embedding vectors lie on a unit hypersphere. In this geometry:

- **Cosine similarity** = dot product of two unit vectors = measures the *angle* between them
- **Euclidean distance** = measures the *chord length* between them

Both are valid on the unit hypersphere and are monotonically related to each other. We use cosine similarity (inner product) because:
1. FAISS `IndexFlatIP` natively computes inner products — no extra computation needed
2. Cosine similarity is bounded in [-1, 1], making thresholds interpretable (0.65 = "65% similar angle")
3. Metric learning losses that use L2-normalised embeddings (triplet, ArcFace) implicitly optimise angular distances

---

*End of design write-up.*
