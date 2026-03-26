"""
train_currency.py  (updated — handles notes + coins combined)
--------------------------------------------------------------
Run this ONCE to train your combined currency classifier.

Expected folder layout BEFORE running:
    notes_dataset/
        10/        ← ₹10 note images
        20/
        50/
        100/
        200/
        500/
        2000/

    coins_dataset/
        1/         ← ₹1 coin images  (use whatever folder names your dataset has)
        2/
        5/
        10/        ← NOTE: same name "10" as notes — handled automatically below
        ...

This script:
  1. Merges both datasets into one combined/ folder,
     renaming clashing class names (e.g. note_10 vs coin_10)
  2. Auto-splits into train/val (80/20)
  3. Trains YOLOv8n-cls
  4. Saves best model → runs/classify/currency/weights/best.pt

Usage:
    pip install ultralytics
    python train_currency.py
"""

import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_DIR = "coins_dataset"     # your unified folder (denomination/coin subfolders)
MERGED_DIR  = "currency_merged"   # intermediate merged folder (auto-created)
SPLIT_DIR   = "currency_split"    # final train/val split folder (auto-created)

EPOCHS      = 60
IMG_SIZE    = 224
VAL_SPLIT   = 0.2
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def merge_datasets():
    """
    Copy notes and coins into a single merged/ folder from the unified DATASET_DIR.
    Adds prefix 'note_' and 'coin_' to every class folder based on folder name.
    """
    merged = Path(MERGED_DIR)
    dataset_root = Path(DATASET_DIR)

    if merged.exists():
        print(f"✅ Merged folder '{MERGED_DIR}' already exists — skipping merge.")
        return

    print(f"📂 Merging from {DATASET_DIR} ...")

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset folder '{DATASET_DIR}' not found.")

    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue

        # Logic to determine prefix: folders with " Coin" are coins, others are notes
        name = class_dir.name
        if " Coin" in name:
            prefix = "coin"
            # Clean up denomination: "10 Coin" -> "10"
            denom = name.replace(" Coin", "").strip()
        else:
            prefix = "note"
            denom = name.strip()

        class_name = f"{prefix}_{denom}"   # e.g. note_500, coin_10
        out_dir = merged / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        images = (list(class_dir.glob("*.jpg")) +
                  list(class_dir.glob("*.jpeg")) +
                  list(class_dir.glob("*.png")))

        for img in images:
            shutil.copy(img, out_dir / img.name)

        print(f"   {class_name}: {len(images)} images")

    print(f"✅ Merge complete → {MERGED_DIR}/\n")


def split_dataset():
    """Split merged folder into train/val."""
    src  = Path(MERGED_DIR)
    dst  = Path(SPLIT_DIR)

    if dst.exists():
        print(f"✅ Split folder '{SPLIT_DIR}' already exists — skipping split.")
        return

    print(f"✂️  Splitting {MERGED_DIR} → {SPLIT_DIR} (80/20) ...")
    random.seed(RANDOM_SEED)

    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue

        images = (list(class_dir.glob("*.jpg")) +
                  list(class_dir.glob("*.jpeg")) +
                  list(class_dir.glob("*.png")))

        if not images:
            print(f"   ⚠️  No images in '{class_dir.name}', skipping.")
            continue

        random.shuffle(images)
        split_idx  = max(1, int(len(images) * (1 - VAL_SPLIT)))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
            out_dir = dst / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in imgs:
                shutil.copy(img, out_dir / img.name)

        print(f"   {class_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val")

    print(f"✅ Split complete → {SPLIT_DIR}/\n")


def print_class_summary():
    """Print a summary of all classes and image counts."""
    merged = Path(MERGED_DIR)
    if not merged.exists():
        return

    print("\n📊 CLASS SUMMARY")
    print("-" * 40)
    note_classes = sorted([d for d in merged.iterdir() if d.is_dir() and d.name.startswith("note_")])
    coin_classes = sorted([d for d in merged.iterdir() if d.is_dir() and d.name.startswith("coin_")])

    total = 0
    print("  NOTES:")
    for c in note_classes:
        n = len(list(c.glob("*.jpg")) + list(c.glob("*.jpeg")) + list(c.glob("*.png")))
        flag = " ⚠️  (low — consider adding more)" if n < 100 else ""
        print(f"    {c.name:<15} {n} images{flag}")
        total += n

    print("  COINS:")
    for c in coin_classes:
        n = len(list(c.glob("*.jpg")) + list(c.glob("*.jpeg")) + list(c.glob("*.png")))
        flag = " ⚠️  (low — consider adding more)" if n < 100 else ""
        print(f"    {c.name:<15} {n} images{flag}")
        total += n

    print("-" * 40)
    print(f"  Total classes : {len(note_classes) + len(coin_classes)}")
    print(f"  Total images  : {total}")
    print("-" * 40 + "\n")


def train():
    merge_datasets()
    split_dataset()
    print_class_summary()

    model = YOLO("yolov8n-cls.pt")   # downloads ~6MB on first run

    print("🚀 Starting training ...")
    results = model.train(
        data     = SPLIT_DIR,
        epochs   = EPOCHS,
        imgsz    = IMG_SIZE,
        device   = 0,            # ← use GPU
        project  = "runs/classify",
        name     = "currency",
        exist_ok = True,
    )

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print()
    print("   Best model: runs/classify/currency/weights/best.pt")
    print()
    print("   Next steps:")
    print("   1. Copy best.pt into your DivyaDrishti-main/ folder")
    print("   2. Rename it to currency_best.pt")
    print("   3. Run python server.py as normal")
    print("=" * 60)


if __name__ == "__main__":
    train()
