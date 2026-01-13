import sqlite3

conn = sqlite3.connect("instance/medikidney.db")
cur = conn.cursor()

# tambah kolom nama pasien jika belum ada
cur.execute("""
ALTER TABLE history ADD COLUMN patient_name TEXT
""")

conn.commit()
conn.close()

print("Kolom patient_name berhasil ditambahkan")
