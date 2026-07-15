import sqlite3
import os

DB_PATH = 'data/screening_results.db'
if os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT username, email FROM users')
    rows = c.fetchall()
    print("Users found:", rows)
    conn.close()
else:
    print("DB not found")
