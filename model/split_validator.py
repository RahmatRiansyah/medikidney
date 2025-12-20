import os
import shutil
import random

# ======================
# KONFIGURASI
# ======================
SOURCE_DIR = "dataset/validator_raw"
TARGET_DIR = "dataset/validator"

CLASSES = ["kidney", "non_kidney"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".PNG")

random.seed(42)

# ======================
# BUAT FOLDER TARGET
# ======================
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(
            os.path.join(TARGET_DIR, split, cls),
            exist_ok=True
        )

# ======================
# AMBIL SEMUA GAMBAR (RECURSIVE)
# ======================
def get_all_images(folder):
    images = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(EXTENSIONS):
                images.append(os.path.join(root, f))
    return images

# ======================
# SPLIT DATASET
# ======================
for cls in CLASSES:
    class_dir = os.path.join(SOURCE_DIR, cls)

    images = get_all_images(class_dir)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img_path in train_imgs:
        shutil.copy(
            img_path,
            os.path.join(TARGET_DIR, "train", cls, os.path.basename(img_path))
        )

    for img_path in val_imgs:
        shutil.copy(
            img_path,
            os.path.join(TARGET_DIR, "val", cls, os.path.basename(img_path))
        )

    for img_path in test_imgs:
        shutil.copy(
            img_path,
            os.path.join(TARGET_DIR, "test", cls, os.path.basename(img_path))
        )

    print(f"{cls}: {len(train_imgs)} train | {len(val_imgs)} val | {len(test_imgs)} test")

print("Split dataset validator selesai.")
