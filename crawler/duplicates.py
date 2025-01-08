import os
import sqlite3
import logger
import logging

logger.setup_logging()

db_file = os.path.join(os.path.dirname(__file__), "seen.db")

logging.info(f"DB path: {db_file}")

# Create table if it doesn't exist
conn = sqlite3.connect(db_file)
c = conn.cursor()
c.execute(
    """CREATE TABLE IF NOT EXISTS seen_voxes
             (vox_id TEXT PRIMARY KEY,
              score FLOAT)"""
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


def mark_already_seen(vox_id: str, score: float):
    vox_id = vox_id.lower().strip()
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO seen_voxes (vox_id, score) VALUES (?, ?)",
        (
            vox_id,
            score,
        ),
    )
    conn.commit()
    conn.close()
