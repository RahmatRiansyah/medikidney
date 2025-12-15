import os
import shutil
import random

# ======================
# KONFIGURASI
# ======================
SOURCE_DIR = "dataset"
CLASSES = ["stone", "normal"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# ======================
# BUAT FOLDER TARGET
# ======================
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(SOURCE_DIR, split, cls), exist_ok=True)

# ======================
# SPLIT DATASET
# ======================
for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(class_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(SOURCE_DIR, "train", cls, img)
        )

    for img in val_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(SOURCE_DIR, "val", cls, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(class_dir, img),
            os.path.join(SOURCE_DIR, "test", cls, img)
        )

    print(f"{cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

print("✅ Dataset split selesai.")
