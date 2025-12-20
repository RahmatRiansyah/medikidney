import sqlite3
import os

DB_PATH = os.path.join("instance", "medikidney.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("=== USERS ===")
for row in cursor.execute("SELECT id, username, role FROM users"):
    print(row)

conn.close()
