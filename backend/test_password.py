import sqlite3
import os
from werkzeug.security import check_password_hash

DB_PATH = os.path.join("instance", "medikidney.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("SELECT password FROM users WHERE username = 'dokter1'")
hashed = cursor.fetchone()[0]

print(check_password_hash(hashed, "medikidney123"))

conn.close()
