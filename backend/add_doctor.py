import sqlite3
import os
from werkzeug.security import generate_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "..", "instance", "medikidney.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

username = "dokter1"
password = "medikidney123"
full_name = "Dr. Demo Spesialis Urologi"
role = "doctor"

hashed_password = generate_password_hash(password)

cursor.execute("""
INSERT INTO users (username, password, full_name, role)
VALUES (?, ?, ?, ?)
""", (
    username,
    hashed_password,
    full_name,
    role
))

conn.commit()
conn.close()

print("Akun dokter berhasil ditambahkan dengan password terenkripsi!")
