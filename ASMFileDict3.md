# ASMFileDict3.py 完整使用說明

## 目錄

1. [概述](#概述)
2. [安裝與依賴](#安裝與依賴)
3. [FileDict - 鍵值存儲](#filedict---鍵值存儲)
4. [FileSQL3 - 文件塊存儲](#filesql3---文件塊存儲)
5. [FileDataFrame - 大數據處理](#filedataframe---大數據處理)
6. [VectorStore - 向量存儲](#vectorstore---向量存儲)
7. [ANNIndex - 近似最近鄰搜索](#annindex---近似最近鄰搜索)
8. [HammingCalculator - Hamming 距離計算](#hammingcalculator---hamming-距離計算)
9. [HDCVectorOps - 超維計算操作](#hdcvectorops---超維計算操作)
10. [輔助函數](#輔助函數)
11. [資料庫遷移工具](#資料庫遷移工具)
12. [環境變數配置](#環境變數配置)
13. [性能優化建議](#性能優化建議)

---

## 概述

`ASMFileDict3.py` 是一個基於 SQLite 的高性能文件字典系統，整合了：

- **鍵值存儲**：類似 Python dict 的持久化存儲
- **文件塊存儲**：支持大文件分塊存儲和流式讀取
- **大數據處理**：生成器式 DataFrame 操作，內存友好
- **向量存儲**：支持 numpy 數組的二進制序列化
- **近似最近鄰 (ANN)**：基於 LSH 和 Hamming 距離的快速相似度搜索
- **ASM 加速**：可選的 C++ JIT 編譯加速

---

## 安裝與依賴

### 必要依賴

```python
import numpy as np  # 必須安裝
```

### 可選依賴（ASM 加速）

系統需要有 C++ 編譯器（g++、clang++ 或 MSVC）才能啟用 ASM 加速。

### 檢查 ASM 後端狀態

```python
from ASMFileDict3 import asm_available

if asm_available():
    print("ASM 加速已啟用")
else:
    print("使用純 Python 後備實現")
```

---

## FileDict - 鍵值存儲

### 功能說明

`FileDict` 提供類似 Python 字典的介面，但數據持久化存儲在 SQLite 資料庫中。支持緩衝寫入、搜索、批量操作，以及可選的向量化 ANN 搜索。

### 初始化參數

```python
FileDict(
    file_path=":memory:",    # 資料庫路徑，":memory:" 為內存模式
    buffer_size=1000,        # 寫入緩衝區大小
    buffer_idle_time=2,      # 緩衝區空閒時間（秒）後自動提交
    table='filedict',        # 資料表名稱
    ann_config=None          # ANN 配置（可選）
)
```

### 基本操作

```python
from ASMFileDict3 import FileDict

# 創建內存字典
fd = FileDict(":memory:")

# 創建持久化字典
fd = FileDict("my_data.db")

# === 基本讀寫 ===
fd['name'] = 'Alice'
fd['age'] = '25'  # 注意：值必須是字符串

# 讀取
print(fd['name'])           # 輸出: Alice
print(fd.get('name'))       # 輸出: Alice
print(fd.get('unknown'))    # 輸出: None（不存在時返回 None）
print(fd.get('unknown', 'default'))  # 不支持默認值，總是返回 None

# 刪除
del fd['age']

# 長度
print(len(fd))  # 輸出: 1
print(fd.len()) # 輸出: 1

# 關閉（重要！確保數據寫入）
fd.close()
```

### 迭代操作

```python
fd = FileDict(":memory:")

# 批量添加
for i in range(100):
    fd[f'key_{i}'] = f'value_{i}'

# 遍歷所有鍵（生成器，內存友好）
for key in fd.keys():
    print(key)

# 遍歷所有值
for value in fd.values():
    print(value)

# 遍歷所有鍵值對
for key, value in fd.items():
    print(f"{key}: {value}")

# 使用 for 循環直接迭代鍵
for key in fd:
    print(key)

fd.close()
```

### 搜索功能

```python
fd = FileDict(":memory:")

fd['apple'] = 'fruit'
fd['apricot'] = 'fruit'
fd['banana'] = 'fruit'
fd['carrot'] = 'vegetable'
fd['cabbage'] = 'vegetable'

# 按鍵模糊搜索（LIKE 語法，% 為萬用字元）
for key in fd.search_keys('ap%'):  # 以 'ap' 開頭
    print(key)  # apple, apricot

# 按鍵精確搜索
for key in fd.search_keys('apple', like=False):
    print(key)  # apple

# 按值搜索
for key in fd.search_values('fruit'):
    print(key)  # apple, apricot, banana

# 限制結果數量
for key in fd.search_keys('%', limited=2):
    print(key)  # 只返回前 2 個

fd.close()
```

### 批量操作

```python
fd = FileDict(":memory:")

# 從字典導入（會清空現有數據！）
data = {
    'key1': 'value1',
    'key2': 'value2',
    'key3': 'value3'
}
fd.from_dict(data)

# 批量添加（不清空現有數據）
more_data = {
    'key4': 'value4',
    'key5': 'value5'
}
fd.add_items(more_data)

print(len(fd))  # 輸出: 5

fd.close()
```

### 啟用 ANN 向量搜索

```python
from ASMFileDict3 import FileDict, ANNConfig, HDCVectorOps
import numpy as np

fd = FileDict(":memory:")

# 啟用 ANN（指定向量維度）
fd.enable_ann(vector_dim=1024)

# 創建 HDC 向量操作器
hdc = HDCVectorOps(dim=1024)

# 添加數據和對應的向量
words = ['dog', 'cat', 'wolf', 'fish']
for word in words:
    fd[word] = f'{word} is an animal'
    vector = hdc.random_vector(seed=hash(word))
    fd.set_vector(word, vector)

# 通過向量搜索相似的鍵
query_vector = fd.get_vector('dog')
results = fd.search_by_vector(query_vector, k=3)

for key, distance in results:
    print(f"{key}: 距離 {distance}")

# 返回值而不是距離
results = fd.search_by_vector(query_vector, k=3, return_values=True)
for key, value in results:
    print(f"{key}: {value}")

fd.close()
```

### 獲取表列表

```python
fd = FileDict("multi_table.db", table='users')
fd['user1'] = 'Alice'

fd2 = FileDict("multi_table.db", table='products')
fd2['prod1'] = 'iPhone'

# 獲取資料庫中所有表
print(fd.Tables())  # ['users', 'products']

fd.close()
fd2.close()
```

### 緩存統計

```python
fd = FileDict(":memory:")

# ... 進行一些操作 ...

stats = fd.get_cache_stats()
print(f"緩存命中: {stats['hits']}")
print(f"緩存未命中: {stats['misses']}")
print(f"命中率: {stats['hit_rate']:.2%}")

fd.close()
```

---

## FileSQL3 - 文件塊存儲

### 功能說明

`FileSQL3` 用於存儲大文件，自動將文件分塊存儲在 SQLite 中，支持流式讀取和隨機訪問。

### 初始化

```python
from ASMFileDict3 import FileSQL3

# 內存模式
fs = FileSQL3(":memory:")

# 持久化模式
fs = FileSQL3("files.db")
```

### 存儲文件

```python
fs = FileSQL3("files.db")

# 從本地文件存儲
fs.put("local_file.mp4", p_path="videos/my_video.mp4")

# 指定描述
fs.put("document.pdf", p_path="docs/readme.pdf", description="項目說明文檔")

# 指定塊大小（默認 8MB）
fs.put("large_file.bin", block_size=1024*1024*16)  # 16MB 塊

fs.close()
```

### 從字節數據存儲

```python
fs = FileSQL3(":memory:")

# 直接從 bytes 存儲
data = b"Hello, World!" * 1000
fs.putBytes(data, "hello.txt")

# 從字節流存儲（適合大數據）
def data_generator():
    for i in range(100):
        yield f"Line {i}\n".encode()

fs.putBytesStream(data_generator(), "lines.txt")

fs.close()
```

### 讀取文件

```python
fs = FileSQL3("files.db")

# 獲取文件對象
f = fs.get("videos/my_video.mp4")

if f:
    # 文件屬性
    print(f"大小: {f.size} bytes")
    print(f"創建時間: {f.create}")
    print(f"修改時間: {f.modified}")
    print(f"MIME 類型: {f.mimetype}")
    print(f"編碼: {f.encoding}")
    
    # 讀取全部內容
    content = f.read()
    
    # 讀取指定大小
    f.seek(0)
    chunk = f.read(1024)  # 讀取 1KB
    
    # 隨機訪問
    f.seek(1000)
    print(f"當前位置: {f.tell()}")
    
    # 分塊讀取
    f.seek(0)
    while True:
        chunk = f.read(1024 * 1024)  # 每次讀 1MB
        if not chunk:
            break
        # 處理 chunk...

fs.close()
```

### 使用 with 語句

```python
fs = FileSQL3("files.db")

f = fs.get("data.bin")
if f:
    with f:  # 支持上下文管理器
        content = f.read()
        # 處理 content...

fs.close()
```

### 搜索和遍歷

```python
fs = FileSQL3("files.db")

# 搜索文件路徑（LIKE 語法）
for path in fs.search("images/%"):  # 所有 images/ 下的文件
    print(path)

for path in fs.search("%.jpg"):  # 所有 .jpg 文件
    print(path)

# 遍歷所有文件
for path in fs.keys():
    print(path)

# 或使用 for 循環
for path in fs:
    print(path)

# 檢查文件是否存在
if "videos/my_video.mp4" in fs:
    print("文件存在")

# 獲取文件數量
print(f"共 {len(fs)} 個文件")

fs.close()
```

### 獲取文件信息

```python
fs = FileSQL3("files.db")

info = fs.get_file_info("docs/readme.pdf")
if info:
    print(f"路徑: {info['path']}")
    print(f"大小: {info['length']}")
    print(f"創建時間: {info['created']}")
    print(f"修改時間: {info['modified']}")
    print(f"MIME 類型: {info['mimetype']}")
    print(f"編碼: {info['encoding']}")
    print(f"描述: {info['description']}")

# 獲取所有文件總大小
total_size = fs.get_total_size()
print(f"總大小: {total_size / 1024 / 1024:.2f} MB")

fs.close()
```

### 更新文件元數據

```python
fs = FileSQL3("files.db")

# 更新描述
fs.update_files_table("docs/readme.pdf", description="更新後的描述")

# 更新多個欄位
fs.update_files_table(
    "docs/readme.pdf",
    description="新描述",
    mimetype="application/pdf"
)

fs.close()
```

### 刪除文件

```python
fs = FileSQL3("files.db")

# 刪除單個文件
success = fs.delete("videos/old_video.mp4")
print(f"刪除{'成功' if success else '失敗'}")

# 批量刪除
paths = ["temp/file1.txt", "temp/file2.txt", "temp/file3.txt"]
deleted_count = fs.delete_batch(paths)
print(f"成功刪除 {deleted_count} 個文件")

fs.close()
```

### 導出文件

```python
fs = FileSQL3("files.db")

# 導出到本地
success = fs.export_file("videos/my_video.mp4", "/path/to/output.mp4")
if success:
    print("導出成功")

fs.close()
```

### 資料庫優化

```python
fs = FileSQL3("files.db")

# 刪除大量文件後，執行 VACUUM 優化
fs.vacuum()

fs.close()
```

---

## FileDataFrame - 大數據處理

### 功能說明

`FileDataFrame` 提供類似 pandas DataFrame 的介面，但基於 SQLite 存儲，使用生成器式處理，內存友好，適合處理大規模數據。

### 初始化和創建表

```python
from ASMFileDict3 import FileDataFrame

# 內存模式
df = FileDataFrame(":memory:")

# 持久化模式
df = FileDataFrame("data.db", table='my_table')

# 創建表結構
df.create_table([
    ('id', 'INTEGER'),
    ('name', 'TEXT'),
    ('age', 'INTEGER'),
    ('salary', 'REAL'),
    ('active', 'INTEGER')  # SQLite 沒有布爾類型
])

df.close()
```

### 從字典創建

```python
df = FileDataFrame(":memory:")

# 自動推斷類型
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85.5, 90.0, 78.5]
}
df.create_from_dict(data)

# 查看結構
print(df.columns())  # ['name', 'age', 'score']
print(df.dtypes())   # {'name': 'TEXT', 'age': 'INTEGER', 'score': 'REAL'}

df.close()
```

### 插入數據

```python
df = FileDataFrame(":memory:")
df.create_table([('id', 'INTEGER'), ('value', 'TEXT')])

# 插入單行
df.insert((1, 'first'))
df.insert((2, 'second'))

# 批量插入
rows = [(i, f'value_{i}') for i in range(3, 103)]
df.insert_many(rows)

# 從生成器插入（內存高效）
def row_generator():
    for i in range(103, 10003):
        yield (i, f'generated_{i}')

df.insert_from_generator(row_generator(), batch_size=1000)

print(len(df))  # 10002

df.close()
```

### 查詢數據

```python
df = FileDataFrame(":memory:")
df.create_table([('id', 'INTEGER'), ('name', 'TEXT'), ('score', 'REAL')])
df.insert_many([
    (1, 'Alice', 85.5),
    (2, 'Bob', 90.0),
    (3, 'Charlie', 78.5),
    (4, 'Diana', 92.0),
    (5, 'Eve', 88.0)
])

# 查詢所有（返回生成器）
for row in df.select():
    print(row)  # {'id': 1, 'name': 'Alice', 'score': 85.5}

# 指定列
for row in df.select(columns='name, score'):
    print(row)  # {'name': 'Alice', 'score': 85.5}

# 條件查詢
for row in df.select(where='score > 85'):
    print(row)

# 排序
for row in df.select(order_by='score DESC'):
    print(row)

# 限制和偏移
for row in df.select(limit=2, offset=1):
    print(row)

# 組合查詢
for row in df.select(
    columns='name, score',
    where='score > 80',
    order_by='score DESC',
    limit=3
):
    print(row)

# 返回元組而非字典
for row in df.select_values():
    print(row)  # (1, 'Alice', 85.5)

# 轉換為列表（注意內存使用）
all_rows = df.to_list()
limited_rows = df.to_list(limit=100)

df.close()
```

### 聚合函數

```python
df = FileDataFrame(":memory:")
df.create_table([('category', 'TEXT'), ('amount', 'REAL')])
df.insert_many([
    ('A', 100), ('A', 150), ('A', 200),
    ('B', 50), ('B', 75),
    ('C', 300)
])

# 計數
total = df.count()
print(f"總行數: {total}")  # 6

count_a = df.count(where="category = 'A'")
print(f"A 類數量: {count_a}")  # 3

# 求和
total_amount = df.sum('amount')
print(f"總金額: {total_amount}")  # 875

# 平均值
avg_amount = df.avg('amount')
print(f"平均金額: {avg_amount:.2f}")  # 145.83

# 最小/最大值
min_amount = df.min('amount')
max_amount = df.max('amount')
print(f"最小: {min_amount}, 最大: {max_amount}")  # 50, 300

# 分組聚合
for group in df.group_by('category', 'SUM(amount) as total, COUNT(*) as count'):
    print(group)
    # {'category': 'A', 'total': 450, 'count': 3}
    # {'category': 'B', 'total': 125, 'count': 2}
    # {'category': 'C', 'total': 300, 'count': 1}

df.close()
```

### 去重查詢

```python
df = FileDataFrame(":memory:")
df.create_table([('category', 'TEXT')])
df.insert_many([
    ('A',), ('B',), ('A',), ('C',), ('B',), ('A',)
])

# 獲取唯一值
for value in df.distinct('category'):
    print(value)  # ('A',), ('B',), ('C',)

df.close()
```

### 更新和刪除

```python
df = FileDataFrame(":memory:")
df.create_table([('id', 'INTEGER'), ('status', 'TEXT')])
df.insert_many([(i, 'pending') for i in range(100)])

# 更新
updated = df.update("status = 'done'", where='id < 50')
print(f"更新了 {updated} 行")

# 刪除
deleted = df.delete(where='id >= 80')
print(f"刪除了 {deleted} 行")

# 刪除所有（不帶 where）
# deleted = df.delete()

df.close()
```

### 函數式操作

```python
df = FileDataFrame(":memory:")
df.create_table([('id', 'INTEGER'), ('value', 'REAL')])
df.insert_many([(i, i * 1.5) for i in range(100)])

# map：對每行應用函數
def double_value(row):
    return row['value'] * 2

for result in df.map(double_value):
    print(result)

# filter：過濾行
def is_large(row):
    return row['value'] > 100

for row in df.filter(is_large):
    print(row)

# reduce：歸約
def sum_values(acc, row):
    return acc + row['value']

total = df.reduce(sum_values, 0)
print(f"總和: {total}")

# batch_process：批量處理
def process(row):
    return f"ID: {row['id']}, Value: {row['value']}"

for result in df.batch_process(process):
    print(result)

df.close()
```

### 創建索引

```python
df = FileDataFrame("data.db")
df.create_table([('id', 'INTEGER'), ('category', 'TEXT')])

# 創建普通索引
df.create_index('category')

# 創建唯一索引
df.create_index('id', unique=True)

# 優化資料庫
df.vacuum()

df.close()
```

---

## VectorStore - 向量存儲

### 功能說明

`VectorStore` 專門用於存儲和檢索 numpy 數組（向量），支持元數據管理、批量操作和可選的 ANN 索引。

### 初始化

```python
from ASMFileDict3 import VectorStore, ANNConfig
import numpy as np

# 基本初始化（ANN 默認關閉）
vs = VectorStore(":memory:")

# 指定表名
vs = VectorStore("vectors.db", table='embeddings')

# 啟用 ANN
ann_config = ANNConfig(enabled=True, dim=1024)
vs = VectorStore("vectors.db", ann_config=ann_config)

vs.close()
```

### 存儲向量

```python
vs = VectorStore(":memory:")

# 存儲浮點向量
v1 = np.random.rand(128).astype(np.float32)
vs.put('embedding_1', v1)

# 存儲整數向量
v2 = np.random.randint(0, 256, size=64, dtype=np.uint8)
vs.put('binary_1', v2)

# 存儲帶元數據的向量
v3 = np.random.rand(128).astype(np.float32)
vs.put('embedding_2', v3, extra={
    'source': 'document.txt',
    'model': 'bert-base',
    'version': 1
})

# 批量存儲
items = [
    ('vec_0', np.random.rand(128).astype(np.float32), {'idx': 0}),
    ('vec_1', np.random.rand(128).astype(np.float32), {'idx': 1}),
    ('vec_2', np.random.rand(128).astype(np.float32), {'idx': 2}),
]
vs.put_batch(items)

vs.close()
```

### 獲取向量

```python
vs = VectorStore(":memory:")

v = np.random.rand(128).astype(np.float32)
vs.put('my_vector', v, extra={'label': 'test'})

# 獲取向量
retrieved = vs.get('my_vector')
if retrieved is not None:
    print(f"形狀: {retrieved.shape}")
    print(f"類型: {retrieved.dtype}")

# 獲取元數據
meta = vs.get_metadata('my_vector')
if meta:
    print(f"鍵: {meta.key}")
    print(f"類型: {meta.dtype}")
    print(f"形狀: {meta.shape}")
    print(f"創建時間: {meta.created}")
    print(f"修改時間: {meta.modified}")
    print(f"額外數據: {meta.extra}")  # {'label': 'test'}

vs.close()
```

### 刪除和檢查

```python
vs = VectorStore(":memory:")

vs.put('vec1', np.random.rand(10).astype(np.float32))
vs.put('vec2', np.random.rand(10).astype(np.float32))

# 檢查存在
print('vec1' in vs)  # True
print('vec3' in vs)  # False

# 獲取數量
print(len(vs))  # 2

# 刪除
success = vs.delete('vec1')
print(success)  # True

vs.close()
```

### 遍歷

```python
vs = VectorStore(":memory:")

for i in range(10):
    vs.put(f'vec_{i}', np.random.rand(10).astype(np.float32))

# 遍歷所有鍵
for key in vs.keys():
    print(key)

# 遍歷所有鍵值對
for key, vector in vs.items():
    print(f"{key}: shape={vector.shape}")

vs.close()
```

### ANN 搜索（Hamming 距離）

```python
from ASMFileDict3 import VectorStore, ANNConfig
import numpy as np

# 創建啟用 ANN 的存儲
config = ANNConfig(enabled=True, dim=1024)
vs = VectorStore(":memory:", ann_config=config)

# 存儲二進制向量（uint64 類型）
for i in range(100):
    v = np.random.randint(0, 2**64, size=16, dtype=np.uint64)  # 16 * 64 = 1024 bits
    vs.put(f'vec_{i}', v)

# 搜索最近鄰
query = np.random.randint(0, 2**64, size=16, dtype=np.uint64)
results = vs.search_nearest(query, k=5)

for key, distance in results:
    print(f"{key}: Hamming 距離 = {distance}")

# 搜索相似向量（按相似度閾值）
similar = vs.search_similar(query, threshold=0.9, max_results=10)
for key, similarity in similar:
    print(f"{key}: 相似度 = {similarity:.2%}")

# 手動啟用/停用 ANN
vs.enable_ann()  # 啟用
vs.disable_ann()  # 停用

print(vs.ann_enabled)  # 檢查狀態

vs.close()
```

---

## ANNIndex - 近似最近鄰搜索

### 功能說明

`ANNIndex` 實現了基於 LSH（Locality-Sensitive Hashing）的近似最近鄰搜索，使用 POPCNT 加速的 Hamming 距離計算。**默認關閉，需顯式啟用**。

### ANNConfig 配置

```python
from ASMFileDict3 import ANNConfig

config = ANNConfig(
    enabled=False,        # 是否啟用（默認關閉）
    dim=10000,            # 向量維度（位數）
    num_tables=10,        # LSH 表數量（更多 = 更準確但更慢）
    hash_bits=16,         # 每個 hash 的位數
    max_candidates=100    # 最大候選數量
)
```

### 基本使用

```python
from ASMFileDict3 import ANNIndex, ANNConfig, HDCVectorOps
import numpy as np

# 創建啟用的 ANN 索引
config = ANNConfig(enabled=True, dim=1024)
ann = ANNIndex(config)

# 創建 HDC 向量操作器
hdc = HDCVectorOps(dim=1024)

# 添加向量
for i in range(1000):
    v = hdc.random_vector(seed=i)
    ann.add(f'vec_{i}', v)

# 獲取向量
v = ann.get('vec_42')

# 檢查和統計
print(len(ann))          # 1000
print('vec_42' in ann)   # True

# 搜索（使用 ANN 加速）
query = hdc.random_vector(seed=42)
results = ann.search(query, k=10, use_ann=True)

for key, distance in results:
    print(f"{key}: 距離 = {distance}")

# 強制使用精確搜索
exact_results = ann.search(query, k=10, use_ann=False)

# 相似度搜索
similar = ann.search_similar(query, threshold=0.95, max_results=50)
for key, similarity in similar:
    print(f"{key}: {similarity:.2%}")

# 刪除
ann.remove('vec_42')

# 啟用/停用
ann.disable()
ann.enable()

ann.close() if hasattr(ann, 'close') else None
```

### 默認關閉行為

```python
from ASMFileDict3 import ANNIndex, ANNConfig

# 默認配置是關閉的
default_config = ANNConfig()
print(default_config.enabled)  # False

# 默認索引是關閉的
ann = ANNIndex()
print(ann.enabled)  # False

# 需要顯式啟用
ann.enable()
print(ann.enabled)  # True
```

---

## HammingCalculator - Hamming 距離計算

### 功能說明

`HammingCalculator` 提供高效的 Hamming 距離計算，使用 POPCNT 指令（如果 ASM 後端可用）。

### 基本使用

```python
from ASMFileDict3 import HammingCalculator
import numpy as np

calc = HammingCalculator()

# 計算字節序列的 Hamming 距離
a = bytes([0b11110000, 0b10101010])
b = bytes([0b11110000, 0b01010101])
dist = calc.distance_bytes(a, b)
print(f"字節 Hamming 距離: {dist}")  # 8（第二個字節完全不同）

# 計算 uint64 數組的 Hamming 距離
a = np.array([0xFF00FF00FF00FF00], dtype=np.uint64)
b = np.array([0x00FF00FF00FF00FF], dtype=np.uint64)
dist = calc.distance_u64(a, b)
print(f"uint64 Hamming 距離: {dist}")  # 64
```

### 批量計算

```python
from ASMFileDict3 import HammingCalculator
import numpy as np

calc = HammingCalculator()

# 一個查詢向量 vs 多個候選向量
query = np.random.randint(0, 2**64, size=16, dtype=np.uint64)
candidates = np.random.randint(0, 2**64, size=(1000, 16), dtype=np.uint64)

# 批量計算（比逐個計算快很多）
distances = calc.distance_batch(query, candidates)
print(f"距離數組形狀: {distances.shape}")  # (1000,)
```

### Top-K 搜索

```python
from ASMFileDict3 import HammingCalculator
import numpy as np

calc = HammingCalculator()

query = np.random.randint(0, 2**64, size=16, dtype=np.uint64)
candidates = np.random.randint(0, 2**64, size=(1000, 16), dtype=np.uint64)

# 找出 top-10 最近鄰
indices, distances = calc.topk(query, candidates, k=10)

for i, (idx, dist) in enumerate(zip(indices, distances)):
    print(f"第 {i+1} 近鄰: 索引={idx}, 距離={dist}")
```

---

## HDCVectorOps - 超維計算操作

### 功能說明

`HDCVectorOps` 實現了超維計算（Hyperdimensional Computing）的核心操作，用於構建語義記憶、關聯學習等應用。

### 初始化

```python
from ASMFileDict3 import HDCVectorOps

# 默認維度（10000 位）
hdc = HDCVectorOps()

# 自定義維度（必須是 64 的倍數）
hdc = HDCVectorOps(dim=4096)

print(hdc.dim)       # 向量維度（位數）
print(hdc.vec_len)   # uint64 數量
```

### 生成隨機向量

```python
from ASMFileDict3 import HDCVectorOps
import numpy as np

hdc = HDCVectorOps(dim=1024)

# 隨機向量
v1 = hdc.random_vector()

# 可重現的隨機向量
v2 = hdc.random_vector(seed=42)
v3 = hdc.random_vector(seed=42)

assert np.array_equal(v2, v3)  # 相同種子 = 相同向量
```

### 綁定操作（XOR）

```python
from ASMFileDict3 import HDCVectorOps
import numpy as np

hdc = HDCVectorOps(dim=1024)

# 創建基礎向量
apple = hdc.random_vector(seed=hash('apple'))
red = hdc.random_vector(seed=hash('red'))
sweet = hdc.random_vector(seed=hash('sweet'))

# 綁定：將概念關聯
red_apple = hdc.bind(apple, red)

# XOR 的可逆性：從綁定中提取
extracted = hdc.bind(red_apple, red)
assert np.array_equal(extracted, apple)  # 還原原始向量
```

### 捆綁操作（Majority）

```python
from ASMFileDict3 import HDCVectorOps

hdc = HDCVectorOps(dim=1024)

# 創建多個向量
vectors = [
    hdc.random_vector(seed=hash('dog')),
    hdc.random_vector(seed=hash('cat')),
    hdc.random_vector(seed=hash('wolf')),
]

# 捆綁：創建「哺乳動物」的原型向量
mammal_prototype = hdc.bundle(vectors)

# 原型與各成員相似
for name, vec in zip(['dog', 'cat', 'wolf'], vectors):
    sim = hdc.similarity(mammal_prototype, vec)
    print(f"{name} 與原型相似度: {sim:.2%}")
```

### 循環移位（Permutation）

```python
from ASMFileDict3 import HDCVectorOps

hdc = HDCVectorOps(dim=1024)

v = hdc.random_vector(seed=42)

# 左移 1 位
shifted = hdc.permute(v, shift=1)

# 右移（負值）
right_shifted = hdc.permute(v, shift=-1)

# 用於序列編碼
words = ['the', 'quick', 'brown', 'fox']
word_vecs = [hdc.random_vector(seed=hash(w)) for w in words]

# 編碼序列（考慮位置）
sequence_vec = hdc.bundle([
    hdc.permute(word_vecs[0], shift=0),
    hdc.permute(word_vecs[1], shift=1),
    hdc.permute(word_vecs[2], shift=2),
    hdc.permute(word_vecs[3], shift=3),
])
```

### Popcount 和相似度

```python
from ASMFileDict3 import HDCVectorOps

hdc = HDCVectorOps(dim=1024)

v = hdc.random_vector(seed=42)

# 計算向量中 1 的個數
ones = hdc.popcount(v)
print(f"1 的個數: {ones}")  # 約為 512（隨機向量約一半是 1）

# 計算相似度（基於 Hamming 距離）
v1 = hdc.random_vector(seed=1)
v2 = hdc.random_vector(seed=2)
v1_copy = hdc.random_vector(seed=1)

sim_same = hdc.similarity(v1, v1_copy)
sim_diff = hdc.similarity(v1, v2)

print(f"相同向量相似度: {sim_same:.2%}")  # 100%
print(f"不同向量相似度: {sim_diff:.2%}")  # 約 50%
```

### 完整語義記憶示例

```python
from ASMFileDict3 import HDCVectorOps, FileDict

def build_semantic_memory():
    """構建語義記憶系統"""
    hdc = HDCVectorOps(dim=10000)
    fd = FileDict(":memory:")
    fd.enable_ann(vector_dim=10000)
    
    # 定義概念屬性
    concepts = {
        'dog': ['animal', 'mammal', 'pet', 'loyal', 'bark'],
        'cat': ['animal', 'mammal', 'pet', 'independent', 'meow'],
        'wolf': ['animal', 'mammal', 'wild', 'pack', 'howl'],
        'goldfish': ['animal', 'fish', 'pet', 'aquatic', 'silent'],
    }
    
    # 為每個屬性創建基礎向量
    attr_vecs = {}
    for attrs in concepts.values():
        for attr in attrs:
            if attr not in attr_vecs:
                attr_vecs[attr] = hdc.random_vector(seed=hash(attr))
    
    # 為每個概念創建組合向量
    for concept, attrs in concepts.items():
        vecs = [attr_vecs[a] for a in attrs]
        concept_vec = hdc.bundle(vecs)
        
        fd[concept] = f"{concept}: {', '.join(attrs)}"
        fd.set_vector(concept, concept_vec)
    
    # 測試聯想
    print("查詢 'wolf' 的相似概念：")
    query_vec = fd.get_vector('wolf')
    for key, dist in fd.search_by_vector(query_vec, k=4):
        sim = 1.0 - (dist / 10000)
        print(f"  {key}: {sim:.1%}")
    
    # 屬性查詢：pet + mammal
    print("\n查詢 'pet + mammal' 的概念：")
    query_vec = hdc.bundle([attr_vecs['pet'], attr_vecs['mammal']])
    fd.set_vector('_query', query_vec)
    
    for key, dist in fd.search_by_vector(query_vec, k=4):
        if not key.startswith('_'):
            sim = 1.0 - (dist / 10000)
            print(f"  {key}: {sim:.1%}")
    
    fd.close()

build_semantic_memory()
```

---

## 輔助函數

### MIME 類型處理

```python
from ASMFileDict3 import getType

# 獲取 MIME 類型
print(getType('photo.jpg'))    # image/jpeg
print(getType('video.mp4'))    # video/mp4
print(getType('unknown.xyz'))  # application/octet-stream
```

### Range 頭解析

```python
from ASMFileDict3 import parse_range_header

# HTTP Range 頭解析
total_length = 10000

start, end = parse_range_header('bytes=0-999', total_length)
print(f"開始: {start}, 結束: {end}")  # 0, 999

start, end = parse_range_header('bytes=500-', total_length)
print(f"開始: {start}, 結束: {end}")  # 500, 9999

start, end = parse_range_header(None, total_length)
print(f"開始: {start}, 結束: {end}")  # 0, 9999
```

### 播放列表生成

```python
from ASMFileDict3 import list_to_m3u, list_to_m3u8

files = ['music/song1.mp3', 'music/song2.mp3', 'music/song3.mp3']

# 生成 M3U 播放列表
m3u = list_to_m3u(files, px='http://server/')
print(m3u)

# 生成 M3U8 播放列表
m3u8 = list_to_m3u8(files, px='http://server/')
print(m3u8)
```

### HTML 包裝函數

```python
from ASMFileDict3 import wrapA, wrapM, wrapV, wrapAu, wrapAuto

# 生成超鏈接
html = wrapA('documents/file.pdf')
print(html)  # <a href='get?path=documents/file.pdf'>documents/file.pdf</a>

# 生成圖片預覽
html = wrapM('images/photo.jpg', height=200)
print(html)  # 帶圖片的鏈接

# 生成視頻播放器
html = wrapV('videos/movie.mp4')
print(html)  # 帶視頻控件

# 生成音頻播放器
html = wrapAu('music/song.mp3')
print(html)  # 帶音頻控件

# 自動選擇（根據 MIME 類型）
html = wrapAuto('photo.jpg')     # 圖片預覽
html = wrapAuto('video.mp4')     # 視頻播放器
html = wrapAuto('song.mp3')      # 音頻播放器
html = wrapAuto('document.pdf')  # 普通鏈接
```

### FileSQL3 列表生成

```python
from ASMFileDict3 import FileSQL3, listFileSQL3

fs = FileSQL3("files.db")

# 生成 HTML 文件列表（自動根據類型選擇顯示方式）
html = listFileSQL3(fs, q='images/%')
print(html)

fs.close()
```

### 分塊讀取

```python
from ASMFileDict3 import yield_file_chunks

with open('large_file.bin', 'rb') as f:
    for chunk in yield_file_chunks(f, chunk_size=1024*1024):
        # 處理每個 1MB 塊
        process(chunk)
```

---

## 資料庫遷移工具

### 遷移 FileSQL3 資料庫

```python
from ASMFileDict3 import newFileSQL3FromOld

# 將舊資料庫遷移到新資料庫
newFileSQL3FromOld(
    oldPath="old_files.db",
    newPath="new_files.db"
)

# 遷移後可以安全使用新資料庫
fs = FileSQL3("new_files.db")
```

### 遷移 FileDict 資料庫

```python
from ASMFileDict3 import newFileDictFromOld

# 自動掃描並遷移所有 key-value 表
newFileDictFromOld(
    oldPath="old_data.db",
    newPath="new_data.db"
)

# 遷移後使用
fd = FileDict("new_data.db", table='my_table')
```

---

## 環境變數配置

| 環境變數 | 說明 | 默認值 |
|---------|------|--------|
| `ASMFILEDICT_VERBOSE` | 啟用詳細日誌 | `"0"` |
| `ASMFILEDICT_DISABLE_ASM` | 禁用 ASM 後端 | `"0"` |
| `ASMFILEDICT_OPT` | C++ 優化級別 | `"3"` |

```bash
# 啟用詳細日誌
export ASMFILEDICT_VERBOSE=1

# 禁用 ASM 加速
export ASMFILEDICT_DISABLE_ASM=1

# 調整優化級別
export ASMFILEDICT_OPT=2
```

---

## 性能優化建議

### FileDict

1. **使用適當的緩衝區大小**：大量寫入時增加 `buffer_size`
2. **批量操作**：使用 `add_items()` 而非單個 `__setitem__`
3. **使用生成器**：遍歷時使用 `keys()`、`values()`、`items()` 而非轉列表

```python
# 好
fd = FileDict(":memory:", buffer_size=5000)
fd.add_items({f'k{i}': f'v{i}' for i in range(10000)})

# 不好
fd = FileDict(":memory:", buffer_size=100)
for i in range(10000):
    fd[f'k{i}'] = f'v{i}'
```

### FileSQL3

1. **選擇合適的塊大小**：大文件使用更大的塊（如 16MB）
2. **使用流式寫入**：大數據使用 `putBytesStream()`
3. **定期執行 VACUUM**：刪除大量文件後

### FileDataFrame

1. **使用生成器插入**：`insert_from_generator()` 比 `insert_many()` 更省內存
2. **避免 `to_list()`**：大數據集應使用生成器遍歷
3. **創建索引**：頻繁查詢的列應創建索引

### VectorStore 和 ANNIndex

1. **ANN 默認關閉**：只在需要時啟用
2. **調整 LSH 參數**：`num_tables` 越多越準確但越慢
3. **限制候選數量**：`max_candidates` 控制精度/速度平衡

---

## 完整示例：文檔相似度搜索系統

```python
from ASMFileDict3 import (
    FileDict, FileSQL3, VectorStore, HDCVectorOps, ANNConfig
)
import numpy as np
import hashlib

class DocumentSearchSystem:
    """文檔相似度搜索系統"""
    
    def __init__(self, db_path: str):
        # 文檔內容存儲
        self.files = FileSQL3(f"{db_path}_files.db")
        
        # 文檔元數據存儲
        self.metadata = FileDict(f"{db_path}_meta.db")
        
        # 向量存儲（啟用 ANN）
        ann_config = ANNConfig(enabled=True, dim=4096)
        self.vectors = VectorStore(f"{db_path}_vectors.db", ann_config=ann_config)
        
        # HDC 向量操作
        self.hdc = HDCVectorOps(dim=4096)
        
        # 詞彙向量緩存
        self._word_vectors = {}
    
    def _get_word_vector(self, word: str) -> np.ndarray:
        """獲取或創建詞彙向量"""
        if word not in self._word_vectors:
            seed = int(hashlib.md5(word.encode()).hexdigest()[:16], 16)
            self._word_vectors[word] = self.hdc.random_vector(seed=seed)
        return self._word_vectors[word]
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """將文本轉換為向量"""
        words = text.lower().split()
        if not words:
            return self.hdc.random_vector()
        
        word_vecs = [self._get_word_vector(w) for w in words]
        return self.hdc.bundle(word_vecs)
    
    def add_document(self, doc_id: str, content: bytes, title: str = ""):
        """添加文檔"""
        # 存儲內容
        self.files.putBytes(content, doc_id)
        
        # 存儲元數據
        self.metadata[doc_id] = title or doc_id
        
        # 創建並存儲向量
        text = content.decode('utf-8', errors='ignore')
        vector = self._text_to_vector(text)
        self.vectors.put(doc_id, vector)
    
    def search(self, query: str, k: int = 5):
        """搜索相似文檔"""
        query_vector = self._text_to_vector(query)
        results = self.vectors.search_nearest(query_vector, k=k)
        
        output = []
        for doc_id, distance in results:
            title = self.metadata.get(doc_id, doc_id)
            similarity = 1.0 - (distance / self.hdc.dim)
            output.append({
                'id': doc_id,
                'title': title,
                'similarity': similarity
            })
        
        return output
    
    def get_document(self, doc_id: str) -> bytes:
        """獲取文檔內容"""
        f = self.files.get(doc_id)
        if f:
            return f.read()
        return None
    
    def close(self):
        """關閉所有連接"""
        self.files.close()
        self.metadata.close()
        self.vectors.close()


# 使用示例
def demo_search_system():
    system = DocumentSearchSystem("my_docs")
    
    # 添加文檔
    docs = [
        ("doc1", b"Python is a programming language for data science", "Python intro"),
        ("doc2", b"Machine learning uses algorithms to learn from data", "ML basics"),
        ("doc3", b"Deep learning is a subset of machine learning", "Deep learning"),
        ("doc4", b"Python is popular for machine learning applications", "Python ML"),
        ("doc5", b"Cooking recipes for healthy meals", "Cooking guide"),
    ]
    
    for doc_id, content, title in docs:
        system.add_document(doc_id, content, title)
    
    # 搜索
    query = "machine learning python"
    results = system.search(query, k=3)
    
    print(f"搜索: '{query}'")
    print("-" * 40)
    for r in results:
        print(f"{r['title']}: {r['similarity']:.1%}")
    
    system.close()

demo_search_system()
```

---

## 運行測試

```python
from ASMFileDict3 import testAll, PerformanceBenchmark

# 運行所有單元測試
success = testAll()

# 運行性能基準測試
if success:
    PerformanceBenchmark.run_all()
```