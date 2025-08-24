"""Minimal script to check ChromaDB using sqlite3 directly."""

import sqlite3
from pathlib import Path

def check_sqlite_directly():
    db_path = Path("backend/storage/chromadb_storage/chroma.sqlite3")
    
    if not db_path.exists():
        print("ChromaDB SQLite file not found")
        return
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in ChromaDB: {[t[0] for t in tables]}")
        
        # Check collections table
        try:
            cursor.execute("SELECT name FROM collections;")
            collections = cursor.fetchall()
            print(f"Collections: {[c[0] for c in collections]}")
            
            # Check embeddings count
            cursor.execute("SELECT COUNT(*) FROM embeddings;")
            embedding_count = cursor.fetchone()[0]
            print(f"Total embeddings: {embedding_count}")
            
        except sqlite3.Error as e:
            print(f"Error querying collections: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error accessing SQLite database: {e}")

if __name__ == "__main__":
    check_sqlite_directly()