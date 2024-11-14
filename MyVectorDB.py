import sqlite3
import json
import os
import time
import numpy as np
import faiss  # Import Faiss
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from sklearn.decomposition import PCA

class Text:
    Black = lambda s: "\033[30m%s\033[0m" % s
    Red = lambda s: "\033[31m%s\033[0m" % s
    Green = lambda s: "\033[32m%s\033[0m" % s
    Yellow = lambda s: "\033[33m%s\033[0m" % s
    Blue = lambda s: "\033[34m%s\033[0m" % s
    Purple = lambda s: "\033[35m%s\033[0m" % s
    Cyan = lambda s: "\033[36m%s\033[0m" % s
    Gray = lambda s: "\033[37m%s\033[0m" % s
    DarkGray = lambda s: "\033[90m%s\033[0m" % s
    LightRed = lambda s: "\033[91m%s\033[0m" % s
    LightGreen = lambda s: "\033[92m%s\033[0m" % s
    LightYellow = lambda s: "\033[93m%s\033[0m" % s
    LightBlue = lambda s: "\033[94m%s\033[0m" % s
    LightPurple = lambda s: "\033[95m%s\033[0m" % s
    LightCyan = lambda s: "\033[96m%s\033[0m" % s
    White = lambda s: "\033[97m%s\033[0m" % s

class MyVectorDB:
    def __init__(self, db_path, embedding_func, pca_dim=32, faiss_index_type='IDMap,Flat'):
        self.db_path = db_path
        self.embedding_func = embedding_func
        self.embedding_dim = None
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()
        self._load_embedding_dim()
        
        self.pca_dim = pca_dim
        self.pca = None
        self.index = None
        self.key_to_id = {}  # Mapping from key to Faiss ID
        self.id_to_key = {}  # Mapping from Faiss ID to key
        self.next_id = 0  # Next available Faiss ID
        
        try:
            self._initialize_indexing(faiss_index_type)
        except Exception as e:
            print(f"Warning: Failed to initialize indexing: {e}")

    def _init_db(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                embedding BLOB,
                magnitude REAL
            )
        ''')
        self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_magnitude ON embeddings (magnitude)')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        self.conn.commit()

    def _load_embedding_dim(self):
        dim = self._load_metadata('embedding_dim')
        if dim is not None:
            self.embedding_dim = int(dim)

    def add_item(self, key, item=None):
        if item is None:
            item = key 
        embedding = self.embedding_func(item)
        self._add_embedding(key, embedding)

    def add_items(self, items_dict):
        embeddings = {}
        for i, (k, v) in enumerate(items_dict.items()):
            t0 = time.time()
            embedding = self.embedding_func(v)
            embeddings[k] = embedding
            t = time.time() - t0
            sI = Text.LightYellow(i)
            sV = Text.Cyan(v[:40])
            sS = Text.LightGreen(len(v)/t if t > 0 else 0)
            
            print(f"{sI} {sV} {len(v)} {sS} chars/sec")
            if i % 20000 == 0 and i > 0:
                self._add_embeddings(embeddings)
                embeddings = {}
        if embeddings:
            self._add_embeddings(embeddings)
        
        self._initialize_indexing()

    def _add_embedding(self, key, embedding):
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            self._save_metadata('embedding_dim', str(self.embedding_dim))
        elif len(embedding) != self.embedding_dim:
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                embedding.extend([0] * (self.embedding_dim - len(embedding)))
        
        magnitude = np.linalg.norm(embedding)
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO embeddings (key, embedding, magnitude)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(embedding), magnitude))
        self.conn.commit()

    def _add_embeddings(self, embeddings_dict):
        for key, emb in embeddings_dict.items():
            if self.embedding_dim is None:
                self.embedding_dim = len(emb)
                self._save_metadata('embedding_dim', str(self.embedding_dim))
            elif len(emb) != self.embedding_dim:
                if len(emb) > self.embedding_dim:
                    emb = emb[:self.embedding_dim]
                else:
                    emb.extend([0] * (self.embedding_dim - len(emb)))
            
            magnitude = np.linalg.norm(emb)
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO embeddings (key, embedding, magnitude)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(emb), magnitude))
        self.conn.commit()

    def _initialize_indexing(self, faiss_index_type='IDMap,Flat'):
        self.cursor.execute('SELECT key, embedding FROM embeddings')
        data = self.cursor.fetchall()
        if not data:
            print("No data found in the database. Indexing skipped.")
            return
        keys, embeddings = zip(*[(key, np.array(json.loads(emb_json))) for key, emb_json in data])
        embeddings = np.array(embeddings).astype('float32')

        # Optional: Apply PCA for dimensionality reduction
        if self.pca_dim and self.pca_dim < self.embedding_dim:
            self.pca = PCA(n_components=self.pca_dim)
            embeddings = self.pca.fit_transform(embeddings).astype('float32')
            current_dim = self.pca_dim
        else:
            current_dim = self.embedding_dim

        # Initialize Faiss index
        index_parts = faiss_index_type.split(',')
        if len(index_parts) == 1:
            index = faiss.index_factory(current_dim, index_parts[0], faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.index_factory(current_dim, ','.join(index_parts[1:]), faiss.METRIC_INNER_PRODUCT)
        # Wrap with IDMap to allow custom IDs
        index = faiss.IndexIDMap(index)
        
        # Add embeddings to index with unique integer IDs
        ids = np.arange(self.next_id, self.next_id + len(embeddings))
        self.key_to_id = {key: idx for key, idx in zip(keys, ids)}
        self.id_to_key = {idx: key for key, idx in zip(keys, ids)}
        index.add_with_ids(embeddings, ids)
        self.next_id += len(embeddings)
        self.index = index

    def search(self, query, top_k=10):
        if self.index is None:
            raise ValueError("Indexing has not been initialized. Please add items first.")
        
        query_vector = np.array(self.embedding_func(query)).astype('float32')
        
        if self.pca:
            query_vector = self.pca.transform([query_vector]).astype('float32')[0]
        
        query_vector = np.expand_dims(query_vector, axis=0)
        
        # Perform search
        distances, ids = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue
            key = self.id_to_key.get(idx, None)
            if key:
                # Depending on the index metric, adjust similarity
                # For INNER_PRODUCT, higher is more similar
                # For L2, lower is more similar
                similarity = dist  # Adjust as needed
                results.append((key, similarity))
        return results

    def _save_metadata(self, key, value):
        self.cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        ''', (key, value))
        self.conn.commit()

    def _load_metadata(self, key):
        self.cursor.execute('SELECT value FROM metadata WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def example_embedding_func(text):
    # Replace this with your actual embedding function
    return list(np.random.rand(128)) 

def wikiFast():
    from fileDict3 import FileDict, FileSQL3
    from myEmbedding import Embedding, Text, ConceptNetwork, EmbeddingModel, EmbeddingConcept
    
    db_path = "wikiFast.db"
    wiki = FileDict("wikipedia.sql3")
    with MyVectorDB(db_path, embedding_func=Embedding) as db:
        db.add_items(wiki)

def main():
    from fileDict3 import FileDict, FileSQL3
    from myEmbedding import Embedding, Text, ConceptNetwork, EmbeddingModel, EmbeddingConcept
    GeneratedImagesAll = FileDict("allnsdatasAll.sql", table="generated")
    GeneratedImagesC = FileDict("allnsdatasAll.sql", table="characters")
    Images = FileSQL3("images.sql")
    normals = set(json.loads(GeneratedImagesC["Normal"]))
    tempKeys = set(Images.keys())
    tempDict = {k: v for k, v in GeneratedImagesAll.items()}
    print(len(tempDict.keys()))
    keysAll = set(GeneratedImagesAll)
    embedsDict = {}
    d1 = {}

    for i, (k, v) in enumerate(GeneratedImagesAll.items()):
        if k not in tempKeys:
            continue
        d = json.loads(v)
        prompt = d["prompt"].split("#", 1)[0]
        d1[k] = prompt

    db_path = "myVectorPromptsFast.db"
    
    with MyVectorDB(db_path, embedding_func=Embedding) as db:
        db.add_items(d1)

def test_example():
    from myEmbedding import EmbeddingConcept

    db_path = "wikiFast.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)

    with MyVectorDB(db_path, EmbeddingConcept) as db:
        # db.add_item("item1", "This is the first item")

        items = {
            "item2": "This is the second item",
            "item3": "This is the third item",
            "item4": "This is the fourth item"
        }
        db.add_items(items)

        query = "Search for similar items"
        results = db.search(query, top_k=20)

        print("Search results:")
        for key, distance in results:
            print(f"Key: {key}, Distance: {distance}")

import time
import random
import string

def generate_random_text(length):
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

def test_performance():
    from myEmbedding import Embedding 

    db_path = "wikiFast.db"
    # db_path = "myVectorPrompts.db"
    
    num_items = 100000
    batch_size = 50000
    search_queries = 10

    with MyVectorDB(db_path, Embedding) as db:
        print(f"Adding {num_items} items to the database...")
        start_time = time.time()
        
        for i in range(0, num_items, batch_size):
            # break  # Commented out to allow adding
            items = {f"item{j}": generate_random_text(50) for j in range(i, min(i + batch_size, num_items))}
            db.add_items(items)
            
            if i % 10000 == 0:
                print(f"Added {i} items...")
        
        add_time = time.time() - start_time
        print(f"Time taken to add {num_items} items: {add_time:.2f} seconds")

        print(f"\nPerforming {search_queries} search queries...")
        start_time = time.time()
        
        for i in range(search_queries):
            query = generate_random_text(10)
            results = db.search(query, top_k=100)
            
            if i % 10 == 0:
                print(f"Completed {i} queries...")
        
        search_time = time.time() - start_time
        print(f"Time taken for {search_queries} search queries: {search_time:.2f} seconds")
        print(f"Average time per query: {search_time/search_queries:.4f} seconds")

        print("\nSample search results:")
        query = generate_random_text(10)
        results = db.search(query, top_k=20)
        for key, distance in results:
            print(f"Key: {key}, Distance: {distance}")

if __name__ == "__main__":
    #main()
    # wikiFast()
    test_performance()
