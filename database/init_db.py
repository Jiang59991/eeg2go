import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

def initialize_database():
    with open(SCHEMA_PATH, "r") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.executescript(schema_sql)
    conn.commit()
    conn.close()

    print("Database initialized using schema.sql")

if __name__ == "__main__":
    initialize_database()
