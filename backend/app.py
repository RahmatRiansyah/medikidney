from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from flask import Response
from flask import redirect

history = []
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

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/history")
def view_history():
    return render_template("history.html", history=history)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    history.clear()
    return redirect("/history")

@app.route("/export-history/csv")
def export_history_csv():
    def generate():
        yield "No,Nama File,Hasil,Kepercayaan (%),Waktu\n"
        for i, item in enumerate(history, start=1):
            yield f"{i},{item['filename']},{item['result']},{item['confidence']},{item['time']}\n"

    return Response(
        generate(),
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment;filename=riwayat_diagnosis_medikidney.csv"
        }
    )

@app.route("/export-history/pdf")
def export_history_pdf():

    if len(history) == 0:
        return "Tidak ada riwayat untuk diexport", 400

    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], "riwayat_diagnosis_medikidney.pdf")

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Riwayat Diagnosis MediKidney")
    y -= 30

    c.setFont("Helvetica", 10)

    for i, item in enumerate(history, start=1):
        text = f"{i}. {item['filename']} | {item['result']} | {item['confidence']}% | {item['time']}"
        c.drawString(50, y, text)
        y -= 18

        if y < 50:   # pindah halaman jika penuh
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 50

    c.save()

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="riwayat_diagnosis_medikidney.pdf"
    )


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

            history.append({
                "filename": file.filename,
                "result": result,
                "confidence": round(confidence * 100, 2),
                "time": datetime.now().strftime("%d-%m-%Y %H:%M")
            })

    return render_template(
    "predict.html",
    result=result,
    confidence=confidence,
    filename=file.filename if result else None
)


# =====================
# RUN SERVER
# =====================
if __name__ == "__main__":
    app.run(debug=True)
