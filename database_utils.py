# database_utils.py

import sqlite3
import json
from datetime import datetime

# === Database Connections ===
log_conn = sqlite3.connect('conversation_history.db', check_same_thread=False)
log_cursor = log_conn.cursor()

# === Logs Table ===
def initialize_logs_table():
    log_cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            user_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp TEXT
        )
    ''')
    log_conn.commit()

def log_message(conversation_id, user_id, user_msg, bot_msg):
    timestamp = datetime.utcnow().isoformat()
    log_cursor.execute('''
        INSERT INTO logs (conversation_id, user_id, user_message, bot_response, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (conversation_id, str(user_id), user_msg, bot_msg, timestamp))
    log_conn.commit()

def fetch_conversation(conversation_id):
    log_cursor.execute("SELECT user_message, bot_response FROM logs WHERE conversation_id = ?", (conversation_id,))
    return log_cursor.fetchall()

# === User Locations ===
def create_user_location_table():
    conn = sqlite3.connect('user_locations.db')
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS user_locations (
                user_id INTEGER PRIMARY KEY,
                location TEXT);""")
    conn.commit()

def insert_or_update_user_location(user_id, location):
    conn = sqlite3.connect('user_locations.db')
    c = conn.cursor()
    c.execute("""INSERT OR REPLACE INTO user_locations
                (user_id, location) VALUES (?, ?);""", (user_id, location))
    conn.commit()

def fetch_user_location(user_id):
    conn = sqlite3.connect('user_locations.db')
    c = conn.cursor()
    c.execute("SELECT location FROM user_locations WHERE user_id = ?;", (user_id,))
    result = c.fetchone()
    return result[0] if result else None

# === Memory Consent (if you use it elsewhere) ===
def create_memory_consent_table():
    log_cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_memory_consent (
            user_id TEXT PRIMARY KEY,
            opted_in INTEGER DEFAULT 0
        )
    ''')
    log_conn.commit()

def set_memory_consent(user_id, consent: bool):
    log_cursor.execute('''
        INSERT OR REPLACE INTO user_memory_consent (user_id, opted_in)
        VALUES (?, ?)
    ''', (str(user_id), int(consent)))
    log_conn.commit()

def has_opted_in_memory(user_id):
    log_cursor.execute('SELECT opted_in FROM user_memory_consent WHERE user_id = ?', (str(user_id),))
    row = log_cursor.fetchone()
    return row and row[0] == 1

# === Elasticsearch Index Hook (Optional) ===
try:
    from elasticsearch import Elasticsearch

    es = Elasticsearch(
        "https://localhost:9200",
        basic_auth=("elastic", "ZzxpijeG=eR=eqfe1=Be"),
        verify_certs=False
    )

    def ensure_index():
        try:
            if not es.indices.exists(index="discord_chat_memory"):
                es.indices.create(index="discord_chat_memory", body={
                    "mappings": {
                        "properties": {
                            "user_id": {"type": "keyword"},
                            "channel_id": {"type": "keyword"},
                            "guild_id": {"type": "keyword"},
                            "content": {"type": "text"},
                            "timestamp": {"type": "date"}
                        }
                    }
                })
        except Exception as e:
            print(f"[WARNING] Failed to ensure ES index: {e}")

    def index_user_message(user_id, channel_id, guild_id, content, timestamp):
        try:
            doc = {
                "user_id": str(user_id),
                "channel_id": str(channel_id),
                "guild_id": str(guild_id),
                "content": content,
                "timestamp": timestamp
            }
            es.index(index="discord_chat_memory", document=doc)
        except Exception as e:
            print(f"[ERROR] Failed to index message: {e}")

    ensure_index()

except ImportError:
    def index_user_message(*args, **kwargs):
        pass

# === Message Expansions (truncate/expand store) ===
def init_message_expansions():
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS message_expansions (
                message_id TEXT PRIMARY KEY,
                full_text  TEXT NOT NULL,
                expanded   INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.commit()

def save_message_expansion(message_id: int, full_text: str, expanded: bool = False):
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("""
            INSERT OR REPLACE INTO message_expansions (message_id, full_text, expanded)
            VALUES (?, ?, ?)
        """, (str(message_id), full_text, 1 if expanded else 0))
        conn.commit()

def get_message_expansion(message_id: int):
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT full_text, expanded FROM message_expansions WHERE message_id = ?", (str(message_id),))
        row = c.fetchone()
        return {"full_text": row[0], "expanded": bool(row[1])} if row else None

def set_message_expanded(message_id: int, expanded: bool):
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("UPDATE message_expansions SET expanded=? WHERE message_id=?", (1 if expanded else 0, str(message_id)))
        conn.commit()
# === User/Server Persistent Instructions ===
def init_user_instructions():
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS user_instructions (
                user_id TEXT PRIMARY KEY,
                instruction TEXT,
                updated_at TEXT
            )
        """)
        conn.commit()

def set_user_instruction(user_id: str, instruction: str):
    """Set the system instruction for a specific user."""
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        if not instruction:
            c.execute("DELETE FROM user_instructions WHERE user_id=?", (str(user_id),))
        else:
            c.execute("""
                INSERT OR REPLACE INTO user_instructions (user_id, instruction, updated_at)
                VALUES (?, ?, ?)
            """, (str(user_id), instruction, datetime.utcnow().isoformat()))
        conn.commit()

def get_user_instruction(user_id: str) -> str | None:
    """Get the persistent system instruction for a user."""
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT instruction FROM user_instructions WHERE user_id=?", (str(user_id),))
        row = c.fetchone()
        return row[0] if row else None

# === Initialize Tables ===
initialize_logs_table()
create_user_location_table()
create_memory_consent_table()
init_message_expansions()
init_user_instructions()

# === Sora Usage / Rate Limiting ===
def init_sora_usage():
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS sora_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                video_id TEXT,
                timestamp TEXT
            )
        """)
        conn.commit()

init_sora_usage()

def log_sora_usage(user_id: str, video_id: str = None):
    """Log a successful Sora generation usage."""
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO sora_usage (user_id, video_id, timestamp) VALUES (?, ?, ?)", 
                  (str(user_id), str(video_id) if video_id else None, datetime.utcnow().isoformat()))
        conn.commit()

def get_last_sora_video_id(user_id: str) -> str | None:
    """Get the video_id of the last video generated by this user."""
    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        c.execute("SELECT video_id FROM sora_usage WHERE user_id = ? AND video_id IS NOT NULL ORDER BY id DESC LIMIT 1", (str(user_id),))
        row = c.fetchone()
        return row[0] if row else None

def check_sora_limit(user_id: str, limit: int = 2, window_seconds: int = 3600) -> bool:
    """
    Check if user is within the rate limit.
    Returns True if allowed, False if blocked.
    """
    # Whitelist for administration/testing
    # 54277066459193344 = sardistic (likely)
    WHITELIST = ["54277066459193344", "54280542740287488"]
    if str(user_id) in WHITELIST:
        return True

    with sqlite3.connect('conversation_history.db') as conn:
        c = conn.cursor()
        # Count usages in the last window_seconds
        # We need to do date math in SQLite or Python. 
        # SQLite's datetime functions are a bit tricky with isoformat strings usually works if they are proper.
        # But safest is to fetch recent timestamps and filter in python or use sqlite datetime modifier.
        
        # Let's filter in SQL using datetime modifier
        cutoff_time = datetime.utcnow().timestamp() - window_seconds
        # We stored isoformat. SQLite 'datetime(timestamp)' expects 'YYYY-MM-DD HH:MM:SS'.
        # Actually, let's just use Python for clarity if the volume is low.
        
        c.execute("SELECT timestamp FROM sora_usage WHERE user_id = ?", (str(user_id),))
        rows = c.fetchall()
        
    now = datetime.utcnow()
    count = 0
    for (ts_str,) in rows:
        try:
            ts = datetime.fromisoformat(ts_str)
            age = (now - ts).total_seconds()
            if age < window_seconds:
                count += 1
        except ValueError:
            pass
            
    return count < limit
