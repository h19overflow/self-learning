"""Quick script to check ChromaDB contents without heavy dependencies."""

import sys
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not available")
    sys.exit(1)

def check_chromadb():
    persist_dir = Path(r"C:\Users\User\Projects\Self_Learning\backend\storage\output_data\chromadb_storage")
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        # List all collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        
        for collection in collections:
            print(f"\nCollection: {collection.name}")
            try:
                count = collection.count()
                print(f"  Documents: {count}")
                
                if count > 0:
                    # Get a few sample documents
                    sample = collection.get(limit=3, include=['documents', 'metadatas'])
                    print(f"  Sample documents: {len(sample.get('documents', []))}")
                    
                    if sample.get('metadatas'):
                        sources = set()
                        for metadata in sample['metadatas']:
                            if metadata and 'source_file' in metadata:
                                sources.add(metadata['source_file'])
                        print(f"  Sample sources: {list(sources)[:3]}")
                else:
                    print("  Collection is empty")
                    
            except Exception as e:
                print(f"  Error accessing collection: {e}")
        
        if not collections:
            print("No collections found in the database")
            
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    check_chromadb()