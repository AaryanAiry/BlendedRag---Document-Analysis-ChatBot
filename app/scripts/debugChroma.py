# pythonService/scripts/debugChroma.py

import sqlite3
import os
from termcolor import colored

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "app", "data", "chroma", "chroma.sqlite3"
)

def check_db_exists():
    print(colored("Step 1: Checking if chroma.sqlite3 exists...", "cyan"))
    if os.path.exists(DB_PATH):
        print(colored(f"✅ Database file found at: {DB_PATH}", "green"))
        return True
    else:
        print(colored("❌ Database file not found!", "red"))
        return False

def check_tables():
    print(colored("\nStep 2: Checking tables inside DB...", "cyan"))
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in cursor.fetchall()]
        if tables:
            print(colored("✅ Tables found:", "green"), tables)
        else:
            print(colored("❌ No tables found in DB!", "red"))
        conn.close()
        return tables
    except Exception as e:
        print(colored(f"❌ Error reading tables: {e}", "red"))
        return []

def check_embeddings_table():
    print(colored("\nStep 3: Checking embeddings table...", "cyan"))
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Try selecting a few rows
        cursor.execute("SELECT * FROM embeddings LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            print(colored(f"✅ Retrieved {len(rows)} rows from embeddings table.", "green"))
        else:
            print(colored("❌ No rows found in embeddings table.", "red"))
        conn.close()
    except Exception as e:
        print(colored(f"❌ Error accessing embeddings table: {e}", "red"))

if __name__ == "__main__":
    if check_db_exists():
        tables = check_tables()
        if "embeddings" in tables:
            check_embeddings_table()
        else:
            print(colored("❌ 'embeddings' table not found in DB.", "red"))
