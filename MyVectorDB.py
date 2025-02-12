import sqlite3
import json
import os
import time
import numpy as np
import faiss  # Import Faiss
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

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
    def __init__(self, db_path, embedding_func, faiss_index_type='Flat'):
        """
        初始化 MyVectorDB 類別。

        :param db_path: SQLite 數據庫的路徑。
        :param embedding_func: 用於生成嵌入向量的函數。
        :param faiss_index_type: Faiss 索引的類型（默認為 'Flat'）。
        """
        self.db_path = db_path
        self.embedding_func = embedding_func
        self.embedding_dim = None
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._init_db()
        self._load_embedding_dim()
        
        self.index = None
        self.key_to_id = {}  # 鍵到 Faiss ID 的映射
        self.id_to_key = {}  # Faiss ID 到鍵的映射
        self.next_id = 0  # 下一個可用的 Faiss ID
        
        try:
            self._initialize_indexing(faiss_index_type)
        except Exception as e:
            print(f"警告：初始化索引失敗: {e}")

    def _init_db(self):
        """初始化 SQLite 數據庫表格。"""
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
        """加載嵌入向量的維度。"""
        dim = self._load_metadata('embedding_dim')
        if dim is not None:
            self.embedding_dim = int(dim)

    def add_item(self, key, item=None):
        """
        添加單個項目到數據庫。

        :param key: 項目的鍵。
        :param item: 項目內容（如果為 None，則使用鍵作為內容）。
        """
        if item is None:
            item = key 
        embedding = self.embedding_func(item)
        self._add_embedding(key, embedding)

    def add_items(self, items_dict):
        """
        批量添加項目到數據庫。

        :param items_dict: 包含鍵值對的字典，鍵為項目鍵，值為項目內容。
        """
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
        
        # 重新初始化索引（注意：如果資料量龐大，可考慮只新增部分向量到 Faiss）
        self._initialize_indexing()

    def _add_embedding(self, key, embedding):
        """
        添加單個嵌入向量到數據庫。

        :param key: 項目的鍵。
        :param embedding: 嵌入向量。
        """
        if self.embedding_dim is None:
            self.embedding_dim = len(embedding)
            self._save_metadata('embedding_dim', str(self.embedding_dim))
        elif len(embedding) != self.embedding_dim:
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                embedding = list(embedding) + [0] * (self.embedding_dim - len(embedding))
        
        magnitude = float(np.linalg.norm(embedding))
        
        self.cursor.execute('''
            INSERT OR REPLACE INTO embeddings (key, embedding, magnitude)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(embedding), magnitude))
        self.conn.commit()

    def _add_embeddings(self, embeddings_dict):
        """
        批量添加嵌入向量到數據庫。

        :param embeddings_dict: 包含鍵和嵌入向量的字典。
        """
        for key, emb in embeddings_dict.items():
            if self.embedding_dim is None:
                self.embedding_dim = len(emb)
                self._save_metadata('embedding_dim', str(self.embedding_dim))
            elif len(emb) != self.embedding_dim:
                if len(emb) > self.embedding_dim:
                    emb = emb[:self.embedding_dim]
                else:
                    emb = list(emb) + [0] * (self.embedding_dim - len(emb))
            
            magnitude = float(np.linalg.norm(emb))
            
            self.cursor.execute('''
                INSERT OR REPLACE INTO embeddings (key, embedding, magnitude)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(emb), magnitude))
        self.conn.commit()

    def _iter_embeddings(self):
        """
        使用 fetchone 的方式從資料庫逐筆讀取所有嵌入向量。

        :yield: 每次返回 (key, embedding_json)。
        """
        self.cursor.execute('SELECT key, embedding FROM embeddings')
        while True:
            row = self.cursor.fetchone()
            if row is None:
                break
            yield row

    def _initialize_indexing(self, faiss_index_type='Flat'):
        """
        初始化 Faiss 索引，使用 generator 從資料庫中讀取資料後分批添加到索引中。

        :param faiss_index_type: Faiss 索引的類型（默認為 'Flat'）。
        """
        # 如果資料庫中沒有資料，直接返回。
        self.cursor.execute('SELECT EXISTS(SELECT 1 FROM embeddings)')
        exist = self.cursor.fetchone()
        if not (exist and exist[0]):
            print("數據庫中沒有數據，索引初始化被跳過。")
            return

        #確認 embedding_dim 可用
        if self.embedding_dim is None:
            raise ValueError("未能獲取 embedding_dim，請確定先添加嵌入向量。")

        # 根據指定的索引類型初始化 Faiss 索引物件
        if faiss_index_type.lower() == 'flat':
            base_index = faiss.IndexFlatIP(self.embedding_dim)  # 使用內積相似性
        elif faiss_index_type.lower() == 'flatl2':
            base_index = faiss.IndexFlatL2(self.embedding_dim)  # 使用 L2 距離
        else:
            raise ValueError(f"Unsupported Faiss index type: {faiss_index_type}")
        
        # 包裝為 IDMap 以支持自定義 ID
        index = faiss.IndexIDMap(base_index)
        
        # 分批讀取資料（例如每次 1000 筆，根據實際情況可以調整）
        chunk_size = 1000
        chunk_keys = []
        chunk_embeddings = []

        for key, emb_json in self._iter_embeddings():
            embedding = np.array(json.loads(emb_json)).astype('float32')
            chunk_keys.append(key)
            chunk_embeddings.append(embedding)
            if len(chunk_keys) >= chunk_size:
                chunk_embeddings_np = np.array(chunk_embeddings)
                if isinstance(base_index, faiss.IndexFlatIP):
                    faiss.normalize_L2(chunk_embeddings_np)
                num = len(chunk_keys)
                ids = np.arange(self.next_id, self.next_id + num).astype('int64')
                for k, idx in zip(chunk_keys, ids):
                    self.key_to_id[k] = idx
                    self.id_to_key[idx] = k
                index.add_with_ids(chunk_embeddings_np, ids)
                self.next_id += num
                chunk_keys = []
                chunk_embeddings = []

        # 處理最後不足 chunk_size 的剩餘資料
        if chunk_keys:
            chunk_embeddings_np = np.array(chunk_embeddings)
            if isinstance(base_index, faiss.IndexFlatIP):
                faiss.normalize_L2(chunk_embeddings_np)
            num = len(chunk_keys)
            ids = np.arange(self.next_id, self.next_id + num).astype('int64')
            for k, idx in zip(chunk_keys, ids):
                self.key_to_id[k] = idx
                self.id_to_key[idx] = k
            index.add_with_ids(chunk_embeddings_np, ids)
            self.next_id += num

        self.index = index

    def search(self, query, top_k=10):
        """
        搜索與查詢最相似的項目。

        :param query: 查詢文本。
        :param top_k: 返回最相似的前 k 個結果。
        :return: 包含 (鍵, 相似度) 的列表。
        """
        if self.index is None:
            raise ValueError("索引尚未初始化。請先添加項目。")
        
        query_vector = np.array(self.embedding_func(query)).astype('float32')
        
        # 如果使用內積相似性，則需要對查詢向量進行歸一化
        if isinstance(self.index.index, faiss.IndexFlatIP):
            faiss.normalize_L2(query_vector.reshape(1, -1))
        query_vector = query_vector.reshape(1, -1)
        
        # 執行搜索
        distances, ids = self.index.search(query_vector, top_k)
        
        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue
            key = self.id_to_key.get(idx, None)
            if key:
                # 根據使用的相似性度量調整相似度
                if isinstance(self.index.index, faiss.IndexFlatIP):
                    similarity = float(dist)  # 內積相似性，越高越相似
                else:
                    similarity = float(-dist)  # L2 距離，相似度可定義為其負值
                results.append((key, similarity))
        return results

    def _save_metadata(self, key, value):
        """
        保存元數據到數據庫。

        :param key: 元數據鍵。
        :param value: 元數據值。
        """
        self.cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES (?, ?)
        ''', (key, value))
        self.conn.commit()

    def _load_metadata(self, key):
        """
        從數據庫加載元數據。

        :param key: 元數據鍵。
        :return: 元數據值或 None。
        """
        self.cursor.execute('SELECT value FROM metadata WHERE key = ?', (key,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def close(self):
        """關閉數據庫連接。"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 示例嵌入生成函數（實際應用中請替換為真實函數）
def example_embedding_func(text):
    """
    示例嵌入函數。

    :param text: 輸入文本。
    :return: 嵌入向量（列表）。
    """
    # 這裏使用隨機向量作為示例
    return list(np.random.rand(128))

def wikiFast():
    from fileDict3 import FileDict, FileSQL3
    from myEmbedding import Embedding, Text, ConceptNetwork, EmbeddingModel, EmbeddingConcept
    db_path = "wikiEmbedding.db"
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

import random
import string

def generate_random_text(length):
    """
    生成隨機文本。

    :param length: 文本長度。
    :return: 隨機生成的文本字符串。
    """
    return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))

def test_performance():
    from myEmbedding import Embedding 

    db_path = "wikiFast.db"    
    num_items = 100000
    batch_size = 50000
    search_queries = 10

    with MyVectorDB(db_path, Embedding) as db:
        print(f"正在將 {num_items} 個項目添加到數據庫中...")
        start_time = time.time()
        
        for i in range(0, num_items, batch_size):
            items = {f"item{j}": generate_random_text(50) for j in range(i, min(i + batch_size, num_items))}
            db.add_items(items)
            
            if i % 10000 == 0:
                print(f"已添加 {i} 個項目...")
        
        add_time = time.time() - start_time
        print(f"添加 {num_items} 個項目所花時間: {add_time:.2f} 秒")

        print(f"\n執行 {search_queries} 個搜索查詢...")
        start_time = time.time()
        
        for i in range(search_queries):
            query = generate_random_text(10)
            results = db.search(query, top_k=100)
            
            if i % 10 == 0:
                print(f"已完成 {i} 個查詢...")
        
        search_time = time.time() - start_time
        print(f"{search_queries} 個搜索查詢所花時間: {search_time:.2f} 秒")
        print(f"每個查詢的平均時間: {search_time/search_queries:.4f} 秒")

        print("\n示例搜索結果:")
        query = generate_random_text(10)
        results = db.search(query, top_k=20)
        for key, distance in results:
            print(f"Key: {key}, Distance: {distance}")

if __name__ == "__main__":
    main()
    
