from flask import (
    Flask, request, render_template,
    redirect, send_from_directory,
    send_file, url_for
)
from flask_login import (
    LoginManager, UserMixin,
    login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import sqlite3
import numpy as np
import os

# =====================
# PATH SETUP
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "medikidney_cnn.h5")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "..", "frontend", "templates")
STATIC_FOLDER = os.path.join(BASE_DIR, "..", "frontend", "static")
DB_PATH = os.path.join(BASE_DIR, "..", "instance", "medikidney.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =====================
# FLASK APP
# =====================
app = Flask(
    __name__,
    template_folder=TEMPLATE_FOLDER,
    static_folder=STATIC_FOLDER
)
app.secret_key = "medikidney-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SESSION_PERMANENT"] = False

# =====================
# LOGIN MANAGER
# =====================
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

# =====================
# USER MODEL
# =====================
class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()

    if row:
        return User(row["id"], row["username"], row["role"])
    return None

# =====================
# LOAD MODEL
# =====================
model = load_model(MODEL_PATH)

# =====================
# PREDICTION FUNCTION
# =====================
def predict_ctscan(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = float(model.predict(img, verbose=0)[0][0])
    confidence = pred if pred > 0.5 else 1 - pred

    if pred > 0.5:
        return "Batu Ginjal", round(confidence * 100, 2)
    else:
        return "Normal", round(confidence * 100, 2)

# =====================
# AUTH ROUTES
# =====================
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            login_user(User(user["id"], user["username"], user["role"]))
            return redirect(url_for("home"))
        else:
            error = "Username atau password salah"

    return render_template("login.html", error=error)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# =====================
# MAIN ROUTES
# =====================
@app.route("/")
@login_required
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = confidence = filename = None

    if request.method == "POST":
        patient_name = request.form.get("patient_name")
        file = request.files.get("file")

        if patient_name and file and file.filename:
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            result, confidence = predict_ctscan(path)

            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO history
                (user_id, patient_name, filename, result, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                current_user.id,
                patient_name,
                filename,
                result,
                float(confidence),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            conn.close()

    return render_template(
        "predict.html",
        result=result,
        confidence=confidence
    )

# =====================
# HISTORY VIEW
# =====================
@app.route("/history")
@login_required
def view_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, patient_name, filename, result, confidence, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,))

    rows = cur.fetchall()
    conn.close()

    history = [{
        "id": r["id"],
        "patient_name": r["patient_name"],
        "filename": r["filename"],
        "result": r["result"],
        "confidence": round(float(r["confidence"]), 2),
        "time": r["created_at"]
    } for r in rows]

    return render_template("history.html", history=history)

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =====================
# EXPORT SINGLE PDF
# =====================
@app.route("/export-history/<int:history_id>/pdf")
@login_required
def export_single_history_pdf(history_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT patient_name, filename, result, confidence, created_at
        FROM history
        WHERE id = ? AND user_id = ?
    """, (history_id, current_user.id))

    data = cur.fetchone()
    conn.close()

    if not data:
        return "Data tidak ditemukan", 404

    pdf_path = os.path.join(
        UPLOAD_FOLDER,
        f"hasil_{data['patient_name'].replace(' ', '_')}.pdf"
    )

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Header/logo
    logo_path = os.path.join(UPLOAD_FOLDER, "header.jpg")
    if os.path.exists(logo_path):
        c.drawImage(logo_path, 50, height - 120, width=60, height=60, mask='auto')
    c.setFont("Helvetica-Bold", 18)
    c.drawString(120, height - 80, "Laporan Hasil Diagnosis CT Scan Ginjal")
    c.setLineWidth(1)
    c.line(50, height - 130, width - 50, height - 130)

    # Tanggal
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 150, f"Tanggal: {data['created_at']}")

    # Data Pasien (tabel rata)
    y = height - 180
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Data Pasien")
    c.setLineWidth(0.5)
    c.line(50, y-5, 200, y-5)
    y -= 20
    c.setFont("Helvetica", 11)
    label_x = 60
    value_x = 160
    row_height = 18
    c.drawString(label_x, y, "Nama Pasien")
    c.drawString(value_x, y, f": {data['patient_name']}")
    y -= row_height
    c.drawString(label_x, y, "Nama File")
    c.drawString(value_x, y, f": {data['filename']}")
    y -= row_height
    c.drawString(label_x, y, "Hasil")
    c.drawString(value_x, y, f": {data['result']}")
    y -= row_height
    c.drawString(label_x, y, "Kepercayaan")
    c.drawString(value_x, y, f": {round(data['confidence'],2)}%")

    # Gambar hasil analisis
    y -= 40
    img_path = os.path.join(UPLOAD_FOLDER, data["filename"])
    if os.path.exists(img_path):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Gambar Analisis:")
        y -= 10
        c.drawImage(img_path, 50, y-220, width=220, height=220, preserveAspectRatio=True, mask='auto')
        y -= 230
    else:
        y -= 10

    # Disclaimer
    c.setFont("Helvetica-Oblique", 10)
    c.setFillColorRGB(0.7, 0.1, 0.1)
    c.drawString(50, y, "⚠️ Hasil ini hanya untuk skrining awal, bukan diagnosis medis.")
    c.setFillColorRGB(0, 0, 0)

    # Footer
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawRightString(width - 50, 40, f"MediKidney | Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
    c.setFillColorRGB(0, 0, 0)

    c.save()
    return send_file(pdf_path, as_attachment=True)

# =====================
# EXPORT ALL HISTORY PDF
# =====================
@app.route("/export-history/pdf")
@login_required
def export_all_history_pdf():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT patient_name, result, confidence, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "Tidak ada data", 400

    pdf_path = os.path.join(UPLOAD_FOLDER, "riwayat_diagnosis.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 50, "Riwayat Diagnosis CT Scan Ginjal")
    c.setLineWidth(1)
    c.line(50, height - 60, width - 50, height - 60)

    # Table header
    y = height - 90
    c.setFont("Helvetica-Bold", 10)
    headers = ["No", "Nama Pasien", "Hasil", "Kepercayaan", "Waktu"]
    col_widths = [30, 120, 80, 80, 120]
    x_positions = [50]
    for w in col_widths[:-1]:
        x_positions.append(x_positions[-1] + w)
    for i, h in enumerate(headers):
        c.drawString(x_positions[i]+2, y, h)
    y -= 15
    c.setLineWidth(0.5)
    c.line(50, y+10, width - 50, y+10)
    c.setFont("Helvetica", 10)

    # Table rows
    for idx, r in enumerate(rows, 1):
        if y < 80:
            c.showPage()
            y = height - 50
        c.drawString(x_positions[0]+2, y, str(idx))
        c.drawString(x_positions[1]+2, y, r['patient_name'])
        c.drawString(x_positions[2]+2, y, r['result'])
        c.drawString(x_positions[3]+2, y, f"{round(r['confidence'],2)}%")
        c.drawString(x_positions[4]+2, y, r['created_at'])
        y -= 15
        c.setLineWidth(0.2)
        c.line(50, y+12, width - 50, y+12)

    c.save()
    return send_file(pdf_path, as_attachment=True)

# =====================
# EXPORT CSV
# =====================
@app.route("/export-history/csv")
@login_required
def export_history_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT patient_name, filename, result, confidence, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,))

    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "Tidak ada data", 400

    csv_path = os.path.join(UPLOAD_FOLDER, "riwayat_diagnosis.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("No,Nama Pasien,File,Hasil,Kepercayaan (%),Waktu\n")
        for idx, r in enumerate(rows, 1):
            waktu = r[4]
            if isinstance(waktu, (int, float)):
                waktu = datetime.fromtimestamp(waktu).strftime("%Y-%m-%d %H:%M:%S")
            # Gunakan tanda kutip untuk setiap kolom agar format tabel tetap rapi jika ada koma di nama/file
            f.write(f'"{idx}","{r[0]}","{r[1]}","{r[2]}","{round(float(r[3]),2)}","{waktu}"\n')

    return send_file(csv_path, as_attachment=True)

# =====================
# CLEAR HISTORY
# =====================
@app.route("/clear-history", methods=["POST"])
@login_required
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM history WHERE user_id = ?", (current_user.id,))
    conn.commit()
    conn.close()
    return redirect(url_for("view_history"))

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run("0.0.0.0", port=8080, debug=True)
