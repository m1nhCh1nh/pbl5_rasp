import sqlite3

def init_db():
    conn = sqlite3.connect('waste.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS waste_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            waste_type TEXT,
            class_name TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def insert_log(filename, waste_type, class_name, confidence):
    conn = sqlite3.connect('waste.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO waste_log (filename, waste_type, class_name, confidence)
        VALUES (?, ?, ?, ?)
    ''', (filename, waste_type, class_name, confidence))
    conn.commit()
    conn.close()

def get_logs():
    conn = sqlite3.connect('waste.db')
    c = conn.cursor()
    c.execute('SELECT * FROM waste_log ORDER BY timestamp DESC')
    rows = c.fetchall()
    conn.close()
    return rows 