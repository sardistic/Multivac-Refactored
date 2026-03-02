import sqlite3
import os

DB_PATH = "/home/coldhunter/Multivac-Modular/conversation_history.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if column exists
        c.execute("PRAGMA table_info(sora_usage)")
        columns = [row[1] for row in c.fetchall()]
        
        if "video_id" not in columns:
            print("Adding video_id column...")
            c.execute("ALTER TABLE sora_usage ADD COLUMN video_id TEXT")
            conn.commit()
            print("Migration successful.")
        else:
            print("Column video_id already exists.")
            
    except Exception as e:
        print(f"Migration failed: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
