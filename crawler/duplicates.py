import os
import sqlite3

db_file = os.path.join(os.path.dirname(__file__), "seen.db")

# Create table if it doesn't exist
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS seen_voxes
             (vox_id TEXT PRIMARY KEY)"""
)
conn.commit()
conn.close()


def already_seen(vox_id: str):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    vox_id = vox_id.lower().strip()
    c.execute("SELECT 1 FROM seen_voxes WHERE vox_id = ?", (vox_id,))
    exists = c.fetchone() is not None
    conn.close()
    return exists


def mark_already_seen(vox_id: str):
    vox_id = vox_id.lower().strip()
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO seen_voxes (vox_id) VALUES (?)", (vox_id,))
    conn.commit()
    conn.close()
