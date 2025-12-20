import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect("backend/database.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS doctors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)
""")

# akun dokter default
c.execute("""
INSERT OR IGNORE INTO doctors (name, username, password)
VALUES (?, ?, ?)
""", (
    "Dr. Admin",
    "dokter",
    generate_password_hash("medikidney123")
))

conn.commit()
conn.close()

print("Database dokter berhasil dibuat")
