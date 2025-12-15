from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# =====================
# PATH SETUP (AMAN)
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "..", "model", "medikidney_cnn.h5"
)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

TEMPLATE_FOLDER = os.path.join(
    BASE_DIR, "..", "frontend", "templates"
)

STATIC_FOLDER = os.path.join(
    BASE_DIR, "..", "frontend", "static"
)

# =====================
# FLASK APP
# =====================
app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================
# LOAD MODEL CNN
# =====================
print("Model path:", MODEL_PATH)
print("Model exists:", os.path.exists(MODEL_PATH))

model = load_model(MODEL_PATH)

# =====================
# PREDICTION FUNCTION
# =====================
def predict_ctscan(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Batu Ginjal", float(pred)
    else:
        return "Normal", float(1 - pred)

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"], file.filename
            )
            file.save(save_path)

            result, confidence = predict_ctscan(save_path)

    return render_template(
        "predict.html",
        result=result,
        confidence=confidence
    )

# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    app.run(debug=True)
