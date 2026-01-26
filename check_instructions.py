import sqlite3
import os

db_path = 'conversation_history.db'

if not os.path.exists(db_path):
    print(f"Error: {db_path} not found.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, instruction, updated_at FROM user_instructions")
        rows = cursor.fetchall()
        print("--- User Instructions ---")
        for row in rows:
            print(f"User: {row[0]}")
            print(f"Instruction: {row[1]}")
            print(f"Updated: {row[2]}")
            print("-------------------------")
        conn.close()
    except Exception as e:
        print(f"Database error: {e}")
