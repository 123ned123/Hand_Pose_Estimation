"""
YOLOv8 segmentation training for the Polynomial Model finger dataset.

Dataset: data/processed/2026-04-23-07-45_Polynomial_Model/
Format:  YOLO instance segmentation (variable-length polygon per finger)
Classes: right-hand-index (0), right-hand-middle (1), right-hand-thumb (2)

Usage:
    python -m src.training.train_polynomial_model
    python -m src.training.train_polynomial_model --epochs 100 --model yolov8m-seg
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path

from ultralytics import YOLO


# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]          # project root
SRC_DIR = ROOT / "data/processed/2026-04-23-07-45_Polynomial_Model"
SPLIT_DIR = SRC_DIR / "yolo_split"                  # created by prepare_split()
# Generated at runtime with absolute paths so ultralytics resolves it correctly
DATASET_YAML = SPLIT_DIR / "dataset.yaml"
RUNS_DIR = ROOT / "runs/polynomial_model"


# ── Dataset preparation ───────────────────────────────────────────────────────

def _write_dataset_yaml():
    """Write dataset.yaml with absolute path so ultralytics never mis-resolves it."""
    cfg = {
        "path": str(SPLIT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 3,
        "names": {0: "right-hand-index", 1: "right-hand-middle", 2: "right-hand-thumb"},
    }
    DATASET_YAML.write_text(yaml.dump(cfg, sort_keys=False))


def prepare_split(val_ratio: float = 0.2, seed: int = 42, force: bool = False):
    """
    Copy images and labels into yolo_split/{images,labels}/{train,val}/.
    Skips if the split already exists unless force=True.
    Always rewrites dataset.yaml with the current absolute path.
    """
    if SPLIT_DIR.exists() and not force:
        print(f"Split already exists at {SPLIT_DIR}. Use --force to rebuild.")
        _write_dataset_yaml()   # always refresh so absolute path stays correct
        return

    random.seed(seed)

    src_images = sorted((SRC_DIR / "images").glob("*.jpg"))
    if not src_images:
        raise FileNotFoundError(f"No .jpg images found in {SRC_DIR / 'images'}")

    random.shuffle(src_images)
    n_val = max(1, int(len(src_images) * val_ratio))
    val_set = set(p.stem for p in src_images[:n_val])

    for split in ("train", "val"):
        (SPLIT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (SPLIT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train, n_val_actual = 0, 0
    for img_path in src_images:
        label_path = SRC_DIR / "labels" / img_path.with_suffix(".txt").name
        split = "val" if img_path.stem in val_set else "train"

        shutil.copy2(img_path, SPLIT_DIR / "images" / split / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, SPLIT_DIR / "labels" / split / label_path.name)

        if split == "train":
            n_train += 1
        else:
            n_val_actual += 1

    _write_dataset_yaml()
    print(f"Split ready — train: {n_train}, val: {n_val_actual} (total: {len(src_images)})")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model_name: str = "yolov8n-seg",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    lr0: float = 0.01,
    patience: int = 20,
    device: str = "",
    val_ratio: float = 0.2,
    seed: int = 42,
    force_split: bool = False,
):
    prepare_split(val_ratio=val_ratio, seed=seed, force=force_split)

    # Resolve absolute path for the YAML (ultralytics needs it or CWD-relative)
    yaml_path = str(DATASET_YAML.resolve())

    model = YOLO(model_name)  # downloads pretrained weights automatically

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        patience=patience,
        device=device if device else None,
        project=str(RUNS_DIR),
        name="train",
        exist_ok=True,
        seed=seed,
        # augmentation — helpful for small datasets
        flipud=0.0,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        mixup=0.0,
    )

    print("\nTraining complete.")
    print(f"Results saved to: {results.save_dir}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8-seg on Polynomial Model dataset")
    p.add_argument("--model", default="yolov8n-seg",
                   help="YOLO model variant (yolov8n-seg, yolov8s-seg, yolov8m-seg, ...)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs without improvement)")
    p.add_argument("--device", default="",
                   help="cuda device (e.g. 0, 0,1) or 'cpu'. Empty = auto-detect")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-split", action="store_true",
                   help="Rebuild train/val split even if it already exists")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        device=args.device,
        val_ratio=args.val_ratio,
        seed=args.seed,
        force_split=args.force_split,
    )
