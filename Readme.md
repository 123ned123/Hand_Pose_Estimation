# Hand Pose Estimation

Estimate the 3D pose of a right hand from a single RGB image or video stream, producing joint coordinates aligned to the **right-hand rule** coordinate system (thumb → X, index → Y, middle → Z).

---

## Project Goals

| Step | Task |
|------|------|
| 1 | Detect and localize the hand in a frame |
| 2 | Segment the hand from the background |
| 3 | Predict 3D coordinates of hand joints |

**Applications:** VR/AR interaction, sign language recognition, human-computer interaction.

---

## Coordinate Convention

The right-hand rule defines the output coordinate frame:

- **X-axis** — Thumb direction  
- **Y-axis** — Index finger direction  
- **Z-axis** — Middle finger direction  

All joint coordinates are predicted relative to this frame.

---

## Project Structure

```
Hand_Pose_Estimation/
├── data/
│   ├── raw/                  # Original images/videos
│   ├── processed/            # Resized, normalized, augmented
│   └── annotations/          # Joint coordinate labels (JSON/CSV)
│
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset class (PyTorch/TF)
│   │   ├── augmentation.py   # Flips, rotations, colour jitter
│   │   └── preprocess.py     # Resize, normalise, keypoint transform
│   │
│   ├── models/
│   │   ├── backbone.py       # Feature extractor (ResNet / EfficientNet)
│   │   ├── detector.py       # Hand detection head (YOLO or SSD)
│   │   ├── segmentor.py      # Hand segmentation (U-Net or SAM)
│   │   └── pose_estimator.py # 3D joint regression head
│   │
│   ├── training/
│   │   ├── train.py          # Training loop
│   │   ├── evaluate.py       # Metrics: PCK, MPJPE
│   │   └── loss.py           # Combined detection + pose loss
│   │
│   ├── inference/
│   │   ├── predict.py        # Single-image inference
│   │   └── video_stream.py   # Real-time webcam pipeline
│   │
│   └── utils/
│       ├── visualize.py      # Draw skeleton overlay on image
│       └── coords.py         # Coordinate frame transforms
│
├── configs/
│   ├── base.yaml             # Shared hyperparameters
│   └── yolo_pose.yaml        # Model-specific overrides
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_inference_demo.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_inference.py
│
├── requirements.txt
└── Readme.md
```

---

## Model Architecture

```
Input Image (RGB)
      │
      ▼
┌─────────────┐
│  Detection  │  YOLO / SSD — bounding box around hand
└──────┬──────┘
       │  cropped ROI
       ▼
┌─────────────┐
│ Segmentation│  U-Net / SAM — binary hand mask
└──────┬──────┘
       │  masked crop
       ▼
┌─────────────┐
│  Backbone   │  ResNet-50 / EfficientNet — feature maps
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Pose Head  │  Regression → 21 joints × (x, y, z)
└─────────────┘
```

---

## Dataset

- Images of hands in various poses and orientations
- Labels: 21 joint 3D coordinates per image
- Split: 80% train / 10% validation / 10% test
- Compatible public datasets: FreiHAND, HO-3D, RHD (Rendered Hand Dataset)

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **MPJPE** | Mean Per Joint Position Error (mm) |
| **PCK** | Percentage of Correct Keypoints within threshold |
| **AUC** | Area under PCK curve |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python src/training/train.py --config configs/base.yaml

# Evaluate
python src/training/evaluate.py --weights checkpoints/best.pt

# Run on webcam
python src/inference/video_stream.py
```

---

## Requirements

```
torch >= 2.0
torchvision
opencv-python
numpy
matplotlib
pyyaml
```
