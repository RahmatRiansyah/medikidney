
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "validator_detector.h5")

IMG_PATH = os.path.join(BASE_DIR, "Normal- (2).jpg")  # ganti nama file

model = load_model(MODEL_PATH)

img = image.load_img(IMG_PATH, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0][0]

print("Raw output model:", pred)
print("Prob kidney:", 1 - pred)
