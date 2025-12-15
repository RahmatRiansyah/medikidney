import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

print("Train path:", os.path.abspath("../dataset/train"))
print("Isi train:", os.listdir("../dataset/train"))
# =====================
# KONFIGURASI DASAR
# =====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
DATASET_DIR = "../dataset"

# =====================
# DATA GENERATOR
# =====================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    os.path.join(DATASET_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =====================
# MODEL CNN (TRANSFER LEARNING)
# =====================
base_model = DenseNet121(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # PHASE 1: freeze

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

# =====================
# COMPILE
# =====================
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", keras.metrics.Recall(name="recall")]
)

model.summary()

# =====================
# TRAINING
# =====================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =====================
# SIMPAN MODEL
# =====================
model.save("medikidney_cnn.h5")

print("✅ Training selesai & model disimpan.")
