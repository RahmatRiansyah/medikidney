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
import struct
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

    pred = model.predict(img, verbose=0)[0][0]
    pred = float(np.asarray(pred).item())  # ⬅️ FIX PENTING

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
        file = request.files.get("file")
        if file and file.filename:
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            result, confidence = predict_ctscan(path)
            confidence = float(confidence)  # ⬅️ FIX SIMPAN REAL

            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO history (user_id, filename, result, confidence)
                VALUES (?, ?, ?, ?)
            """, (
                current_user.id,
                filename,
                result,
                confidence
            ))
            conn.commit()
            conn.close()

    return render_template(
        "predict.html",
        result=result,
        confidence=confidence,
        filename=filename
    )

@app.route("/history")
@login_required
def view_history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, result, confidence, created_at
        FROM history
        WHERE user_id = ?
        ORDER BY created_at DESC
    """, (current_user.id,))
    rows = cur.fetchall()
    conn.close()

    history = []
    for r in rows:
        conf = r["confidence"]

        if isinstance(conf, (bytes, bytearray)):
            conf = struct.unpack("f", conf)[0]

        history.append({
            "filename": r["filename"],
            "result": r["result"],
            "confidence": round(float(conf), 2),
            "time": r["created_at"]
        })

    return render_template("history.html", history=history)

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =====================
# EXPORT HISTORY PDF
# =====================
@app.route("/export-history/pdf")
@login_required
def export_history_pdf():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, result, confidence, created_at
        FROM history WHERE user_id = ?
    """, (current_user.id,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return "Tidak ada data", 400

    pdf_path = os.path.join(UPLOAD_FOLDER, "riwayat_diagnosis.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    y = 800

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Riwayat Diagnosis MediKidney")
    y -= 30
    c.setFont("Helvetica", 10)

    for i, r in enumerate(rows, 1):
        conf = r[2]
        if isinstance(conf, (bytes, bytearray)):
            conf = struct.unpack("f", conf)[0]

        c.drawString(
            50, y,
            f"{i}. {r[0]} | {r[1]} | {round(float(conf),2)}% | {r[3]}"
        )
        y -= 18
        if y < 50:
            c.showPage()
            y = 800

    c.save()
    return send_file(pdf_path, as_attachment=True)

@app.route("/export-history/csv")
@login_required
def export_history_csv():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, result, confidence, created_at
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
        f.write("Filename,Hasil,Kepercayaan (%),Waktu\n")
        for r in rows:
            conf = r[2]
            if isinstance(conf, (bytes, bytearray)):
                import struct
                conf = struct.unpack("f", conf)[0]

            f.write(f"{r[0]},{r[1]},{round(float(conf),2)},{r[3]}\n")

    return send_file(csv_path, as_attachment=True)

@app.route("/clear-history", methods=["POST"])
@login_required
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM history WHERE user_id = ?",
        (current_user.id,)
    )
    conn.commit()
    conn.close()

    return redirect(url_for("view_history"))

# =====================
# RUN
# =====================
if __name__ == "__main__":
    app.run(debug=True)
