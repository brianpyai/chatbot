# ASMOriginAI.py 完整使用說明

## 目錄

1. [系統概述](#1-系統概述)
2. [環境配置與依賴](#2-環境配置與依賴)
3. [核心位運算模組 (BitOps)](#3-核心位運算模組-bitops)
4. [超維計算向量操作 (HDCVectorOps)](#4-超維計算向量操作-hdcvectorops)
5. [語義編碼器 (BitSemanticEncoder)](#5-語義編碼器-bitsemanticencoder)
6. [顏色映射器 (ColorMapper)](#6-顏色映射器-colormapper)
7. [演化引擎 (EvolutionEngine)](#7-演化引擎-evolutionengine)
8. [鍵值存儲 (KVStore)](#8-鍵值存儲-kvstore)
9. [學習數據庫 (LearnDatabase)](#9-學習數據庫-learndatabase)
10. [滾動數據集管理器 (RollingDatasetManager)](#10-滾動數據集管理器-rollingdatasetmanager)
11. [自動訓練器 (AutoTrainer)](#11-自動訓練器-autotrainer)
12. [夢境模組 (EnhancedDreamModule)](#12-夢境模組-enhanceddreammodule)
13. [多模態渲染器 (MultiModalRenderer)](#13-多模態渲染器-multimodalrenderer)
14. [4D 世界模組](#14-4d-世界模組)
15. [主系統類 (ASMsuperAIsystem)](#15-主系統類-asmsuperaisystem)
16. [數據結構](#16-數據結構)
17. [工具函數](#17-工具函數)
18. [測試與性能基準](#18-測試與性能基準)
19. [完整使用範例](#19-完整使用範例)

---

## 1. 系統概述

`ASMOriginAI.py` 是一個純位元計算的多模態 AI 專屬系統，具備以下核心能力：

| 功能領域 | 說明 |
|---------|------|
| **純位元計算** | 使用二進制超維向量進行所有語義運算 |
| **多模態處理** | 支援文本、圖像、JSON、二進制等多種數據類型 |
| **驚奇度學習** | 基於新穎性自動決定學習策略 |
| **夢境重組** | 類人腦的記憶鞏固與知識重組機制 |
| **4D 世界建模** | 真實 3D/4D 空間的圖文對齊與渲染 |
| **想像回灌訓練** | 生成虛擬場景並自我訓練 |

### 執行方式

```bash
# 直接執行（自動運行測試 + 性能測試 + 演示生成）
python ASMOriginAI.py

# 環境變數配置
SUPERAI_VERBOSE=1              # 啟用詳細日誌
SUPERAI_KEEP_TEST_OUTPUTS=1    # 保留測試輸出圖片（預設開啟）
SUPERAI_DISABLE_4D=1           # 停用 4D 世界模組
```

---

## 2. 環境配置與依賴

### 2.1 必需依賴

```python
import numpy as np  # 核心數值計算
```

### 2.2 可選依賴

```python
# 圖像處理
from PIL import Image, ImageDraw, ImageFont, ImageFilter  # HAS_PIL

# 視頻處理
import cv2  # HAS_CV2

# 加速位運算
import ASMOriginBitComputing  # HAS_ORIGINBIT

# 4D 世界引擎
from ASMOrigin4DEngine import Scene4DEngine, ...  # HAS_4D_ENGINE
import ASM4Dobjects  # HAS_4D_OBJECTS

# 高品質文字渲染
from ASMoneMatrixBitData import ASMoneMatrixBitData  # HAS_ONEMATRIX

# 位世界 AI 引擎
from ASMbitWorldAIEngine import BitOps, HDCVectorOps, ...  # HAS_BITWORLD

# 文件字典存儲
from ASMFileDict3 import FileDict, FileSQL3  # HAS_FILEDICT

# 數據集管理
from ASMmyDatasets import ASMDatasetConfig, ...  # HAS_DATASETS
```

### 2.3 全局配置常量

```python
# 存儲目錄
SUPER_AI_DIR = "./superAI"
TEST_OUTPUT_DIR = "./superAI/test_outputs"

# 記憶體預算（位元）
CORE_MEMORY_BITS = 8 * 1024 * 1024           # 1MB
CACHE_MEMORY_BITS = 8 * 8 * 1024 * 1024      # 8MB
EXTENDED_MEMORY_BITS = 8 * 256 * 1024 * 1024 # 256MB

# 默認參數
DEFAULT_VECTOR_DIM = 1764    # 42 × 42
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_SAMPLES = 100000
DEFAULT_STORAGE_MB = 1000

# 驚奇度閾值
SURPRISE_THRESHOLD_LOW = 0.37
SURPRISE_THRESHOLD_HIGH = 0.63
```

---

## 3. 核心位運算模組 (BitOps)

`BitOps` 類提供基礎的位元級運算，是整個系統的計算基礎。

### 3.1 XOR 運算

```python
@staticmethod
def xor(a: np.ndarray, b: np.ndarray) -> np.ndarray
```

**功能**：對兩個數組進行逐位異或運算

**使用示例**：
```python
import numpy as np

a = np.array([0xFF00FF00], dtype=np.uint64)
b = np.array([0x00FF00FF], dtype=np.uint64)

result = BitOps.xor(a, b)
print(f"結果: {hex(result[0])}")  # 0xffffffff
```

### 3.2 AND 運算

```python
@staticmethod
def and_op(a: np.ndarray, b: np.ndarray) -> np.ndarray
```

**功能**：對兩個數組進行逐位與運算

**使用示例**：
```python
a = np.array([0xFF00FF00], dtype=np.uint64)
b = np.array([0x0F0F0F0F], dtype=np.uint64)

result = BitOps.and_op(a, b)
print(f"結果: {hex(result[0])}")  # 0x0f000f00
```

### 3.3 OR 運算

```python
@staticmethod
def or_op(a: np.ndarray, b: np.ndarray) -> np.ndarray
```

**功能**：對兩個數組進行逐位或運算

**使用示例**：
```python
a = np.array([0xFF000000], dtype=np.uint64)
b = np.array([0x00FF0000], dtype=np.uint64)

result = BitOps.or_op(a, b)
print(f"結果: {hex(result[0])}")  # 0xffff0000
```

### 3.4 Popcount（位計數）

```python
@staticmethod
def popcount(a: np.ndarray) -> int
```

**功能**：計算數組中所有「1」位的總數

**使用示例**：
```python
a = np.array([0xFF], dtype=np.uint8)
count = BitOps.popcount(a)
print(f"1 的數量: {count}")  # 8
```

### 3.5 漢明相似度

```python
@staticmethod
def hamming_similarity(a: np.ndarray, b: np.ndarray) -> float
```

**功能**：計算兩個向量的漢明相似度（0~1）

**使用示例**：
```python
a = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
b = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)

similarity = BitOps.hamming_similarity(a, b)
print(f"相似度: {similarity}")  # 1.0

# 不同向量
c = np.array([0x0000000000000000], dtype=np.uint64)
similarity = BitOps.hamming_similarity(a, c)
print(f"相似度: {similarity}")  # 0.0
```

### 3.6 多數投票

```python
@staticmethod
def majority_vote(vectors: List[np.ndarray]) -> np.ndarray
```

**功能**：對多個向量進行位級多數投票，生成共識向量

**使用示例**：
```python
vectors = [
    np.array([0b11110000], dtype=np.uint8),
    np.array([0b11100000], dtype=np.uint8),
    np.array([0b11000000], dtype=np.uint8),
]

result = BitOps.majority_vote(vectors)
# 每個位取多數：位7,6,5 都是 1（3票 > 1.5），其他位各有不同
```

---

## 4. 超維計算向量操作 (HDCVectorOps)

`HDCVectorOps` 類實現超維計算（Hyperdimensional Computing）的核心向量操作。

### 4.1 初始化

```python
def __init__(self, dim: int = DEFAULT_VECTOR_DIM)
```

**參數**：
- `dim`: 向量維度（自動對齊到 64 的倍數）

**使用示例**：
```python
# 創建 1024 維的向量操作器
ops = HDCVectorOps(dim=1024)
print(f"實際維度: {ops.dim}")        # 1024
print(f"向量長度: {ops.vec_len}")    # 16 (1024/64)
```

### 4.2 隨機向量生成

```python
def random_vector(self, seed: Optional[int] = None) -> np.ndarray
```

**功能**：生成隨機超維向量（uint64 packed）

**使用示例**：
```python
ops = HDCVectorOps(dim=1024)

# 使用固定種子生成（可重現）
v1 = ops.random_vector(seed=42)
v2 = ops.random_vector(seed=42)
assert np.array_equal(v1, v2)  # 相同種子產生相同向量

# 不同種子產生不同向量
v3 = ops.random_vector(seed=43)
assert not np.array_equal(v1, v3)
```

### 4.3 綁定操作 (Bind)

```python
def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray
```

**功能**：將兩個向量綁定（XOR），用於創建關聯

**使用示例**：
```python
ops = HDCVectorOps(dim=1024)

key_vec = ops.random_vector(seed=1)
value_vec = ops.random_vector(seed=2)

# 綁定 key 和 value
bound = ops.bind(key_vec, value_vec)

# XOR 是自逆的，可以還原
unbound = ops.bind(bound, value_vec)
assert np.array_equal(unbound, key_vec)
```

### 4.4 捆束操作 (Bundle)

```python
def bundle(self, vectors: List[np.ndarray]) -> np.ndarray
```

**功能**：將多個向量捆束成一個（多數投票），用於創建集合

**使用示例**：
```python
ops = HDCVectorOps(dim=1024)

# 創建多個概念向量
dog = ops.random_vector(seed=1)
cat = ops.random_vector(seed=2)
bird = ops.random_vector(seed=3)

# 捆束成「動物」概念
animals = ops.bundle([dog, cat, bird])

# animals 與每個成員都有一定相似度
print(f"與 dog 相似度: {ops.similarity(animals, dog):.3f}")
print(f"與 cat 相似度: {ops.similarity(animals, cat):.3f}")
```

### 4.5 相似度計算

```python
def similarity(self, a: np.ndarray, b: np.ndarray) -> float
```

**功能**：計算兩個向量的漢明相似度

**使用示例**：
```python
ops = HDCVectorOps(dim=1024)

v1 = ops.random_vector(seed=1)
v2 = ops.random_vector(seed=1)
v3 = ops.random_vector(seed=2)

# 相同向量
print(f"相同: {ops.similarity(v1, v2)}")  # 1.0

# 隨機向量（期望約 0.5）
print(f"隨機: {ops.similarity(v1, v3):.3f}")  # ~0.5
```

---

## 5. 語義編碼器 (BitSemanticEncoder)

`BitSemanticEncoder` 類將各種數據類型編碼為二進制超維向量。

### 5.1 初始化

```python
def __init__(
    self,
    hdc_ops: HDCVectorOps,
    dim_bits: Optional[int] = None,
    seed: int = 0,
    text_ngram_k: int = 2,
    text_use_positions: bool = True,
    text_hash_fanout: int = 4,
)
```

**參數**：
| 參數 | 說明 |
|------|------|
| `hdc_ops` | HDCVectorOps 實例 |
| `dim_bits` | 向量維度（位元數） |
| `seed` | 編碼種子 |
| `text_ngram_k` | N-gram 最大長度 |
| `text_use_positions` | 是否使用位置信息 |
| `text_hash_fanout` | 特徵哈希扇出係數 |

**使用示例**：
```python
ops = HDCVectorOps(dim=1764)
encoder = BitSemanticEncoder(
    hdc_ops=ops,
    dim_bits=1764,
    seed=123,
    text_ngram_k=2,
    text_use_positions=True
)
```

### 5.2 文本編碼

```python
def encode_text(self, text: str, salt: str = "TXT") -> np.ndarray
```

**功能**：將文本編碼為超維向量，使用 token + n-gram + 位置信息

**使用示例**：
```python
ops = HDCVectorOps(dim=1764)
encoder = BitSemanticEncoder(ops, dim_bits=1764)

# 編碼文本
v1 = encoder.encode_text("Hello World")
v2 = encoder.encode_text("Hello World")
v3 = encoder.encode_text("Goodbye World")

# 相同文本產生相同向量
assert np.array_equal(v1, v2)

# 相似文本有較高相似度
similarity = encoder.hdc.similarity(v1, v3)
print(f"相似度: {similarity:.3f}")  # > 0.5（共享 "World"）
```

### 5.3 JSON 編碼

```python
def encode_json(self, obj: Any, salt: str = "JSON") -> np.ndarray
```

**功能**：將 JSON 對象序列化後編碼為向量

**使用示例**：
```python
encoder = BitSemanticEncoder(HDCVectorOps(dim=1764), dim_bits=1764)

data = {
    "name": "Alice",
    "age": 30,
    "skills": ["Python", "ML"]
}

vector = encoder.encode_json(data)
print(f"向量形狀: {vector.shape}")  # (28,) for 1764 bits
```

### 5.4 圖像編碼

```python
def encode_image(self, image: np.ndarray, salt: str = "IMG") -> np.ndarray
```

**功能**：將圖像降採樣並二值化為超維向量

**使用示例**：
```python
import numpy as np

encoder = BitSemanticEncoder(HDCVectorOps(dim=1764), dim_bits=1764)

# 創建測試圖像
image = np.zeros((128, 128, 3), dtype=np.uint8)
image[32:96, 32:96] = (255, 255, 255)  # 白色方塊

vector = encoder.encode_image(image)
print(f"圖像向量: {vector.dtype}, 長度={vector.size}")
```

### 5.5 多模態編碼

```python
def encode_multimodal(self, image: np.ndarray, text: str, salt: str = "MM") -> np.ndarray
```

**功能**：融合圖像和文本為統一向量，使結果與兩者都保持約 0.75 相似度

**使用示例**：
```python
encoder = BitSemanticEncoder(HDCVectorOps(dim=1764), dim_bits=1764)

# 圖像
image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

# 文本
text = "A beautiful sunset over the ocean"

# 多模態編碼
mm_vector = encoder.encode_multimodal(image, text)

# 驗證與原始模態的相似度
img_vector = encoder.encode_image(image)
txt_vector = encoder.encode_text(text)

print(f"與圖像相似度: {encoder.hdc.similarity(mm_vector, img_vector):.3f}")
print(f"與文本相似度: {encoder.hdc.similarity(mm_vector, txt_vector):.3f}")
```

### 5.6 通用編碼

```python
def encode_any(self, x: Any, modality: str = "auto") -> np.ndarray
```

**功能**：自動檢測數據類型並編碼

**使用示例**：
```python
encoder = BitSemanticEncoder(HDCVectorOps(dim=1764), dim_bits=1764)

# 自動檢測類型
v1 = encoder.encode_any("Hello")                    # 文本
v2 = encoder.encode_any({"key": "value"})           # JSON
v3 = encoder.encode_any(np.zeros((64, 64, 3)))      # 圖像

# 指定類型
v4 = encoder.encode_any("123", modality="text")
```

---

## 6. 顏色映射器 (ColorMapper)

`ColorMapper` 類將數值數組映射為 RGB 顏色。

### 6.1 初始化

```python
def __init__(self, palette: str = "viridis")
```

**可用調色板**：
- `viridis`: 藍-綠-黃漸變（默認）
- `plasma`: 紫-紅-黃漸變
- `fire`: 黑-紅-黃-白漸變

**使用示例**：
```python
mapper = ColorMapper(palette="viridis")
```

### 6.2 數組映射

```python
def map_array(self, arr: np.ndarray, vmin: Optional[float] = None,
              vmax: Optional[float] = None) -> np.ndarray
```

**功能**：將 2D 數組映射為 RGB 圖像

**使用示例**：
```python
import numpy as np

mapper = ColorMapper(palette="plasma")

# 創建測試數據
data = np.random.rand(100, 100)

# 映射為 RGB
rgb = mapper.map_array(data)
print(f"輸出形狀: {rgb.shape}")  # (100, 100, 3)
print(f"數據類型: {rgb.dtype}")  # uint8

# 自定義範圍
rgb_custom = mapper.map_array(data, vmin=0.2, vmax=0.8)
```

---

## 7. 演化引擎 (EvolutionEngine)

`EvolutionEngine` 類實現簡化的演化動力學。

### 7.1 初始化

```python
def __init__(self, width=256, height=256)
```

**使用示例**：
```python
engine = EvolutionEngine(width=64, height=64)
```

### 7.2 噪聲初始化

```python
def noise(self, level: float, seed: Optional[int] = None)
```

**功能**：用隨機噪聲初始化狀態

**使用示例**：
```python
engine = EvolutionEngine(width=64, height=64)

# 設置噪聲
engine.matrix.noise(level=0.5, seed=42)
```

### 7.3 演化步進

```python
def step(self)
```

**功能**：執行一步演化（平滑 + 微噪聲）

**使用示例**：
```python
engine = EvolutionEngine(width=64, height=64)
engine.matrix.noise(0.5, seed=42)

# 執行 100 步演化
for _ in range(100):
    engine.step()

# 獲取當前狀態
state = engine.get_state()
print(f"狀態形狀: {state.shape}")  # (64, 64)
```

---

## 8. 鍵值存儲 (KVStore)

`KVStore` 類提供帶語義查詢能力的鍵值存儲。

### 8.1 初始化

```python
def __init__(self, encoder: HDCVectorOps, capacity: int = 1000000,
             db_path: Optional[str] = None)
```

**參數**：
| 參數 | 說明 |
|------|------|
| `encoder` | HDCVectorOps 實例 |
| `capacity` | 最大容量 |
| `db_path` | 持久化路徑（可選） |

**使用示例**：
```python
encoder = HDCVectorOps(dim=1024)

# 記憶體存儲
kv = KVStore(encoder, capacity=10000)

# 持久化存儲
kv_persistent = KVStore(encoder, capacity=10000, db_path="./data/kv.db")
```

### 8.2 存儲數據

```python
def put(self, key: str, value: Any,
        value_type: Optional[str] = None,
        metadata: Optional[Dict] = None) -> KVRecord
```

**功能**：存儲鍵值對，自動檢測類型並生成向量

**使用示例**：
```python
encoder = HDCVectorOps(dim=1024)
kv = KVStore(encoder, capacity=10000)

# 存儲各種類型
kv.put("user:name", "Alice")
kv.put("user:age", 30)
kv.put("user:profile", {"role": "admin", "active": True})
kv.put("user:avatar", np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

# 帶元數據
record = kv.put("doc:123", "Important document",
                metadata={"author": "Bob", "date": "2025-01-01"})
print(f"記錄創建時間: {record.created}")
```

### 8.3 獲取數據

```python
def get(self, key: str) -> Optional[Any]
def get_record(self, key: str) -> Optional[KVRecord]
```

**使用示例**：
```python
# 獲取值
name = kv.get("user:name")
print(f"姓名: {name}")  # Alice

# 獲取不存在的鍵
value = kv.get("nonexistent")
print(f"不存在: {value}")  # None

# 獲取完整記錄
record = kv.get_record("user:name")
print(f"類型: {record.value_type}")
print(f"訪問次數: {record.accessed}")
```

### 8.4 語義查詢

```python
def query_similar(self, query: Any, k: int = 10) -> List[Tuple[KVRecord, float]]
```

**功能**：查找與查詢最相似的 k 個記錄

**使用示例**：
```python
encoder = HDCVectorOps(dim=1024)
kv = KVStore(encoder, capacity=10000)

# 存儲一些數據
kv.put("animal:dog", "A furry pet that barks")
kv.put("animal:cat", "A furry pet that meows")
kv.put("animal:fish", "Lives in water")
kv.put("fruit:apple", "A red fruit")

# 語義查詢
results = kv.query_similar("furry animal", k=3)
for record, similarity in results:
    print(f"{record.key}: {similarity:.3f}")
```

### 8.5 夢境模式

```python
def set_dream_mode(self, in_dream: bool)
```

**功能**：設置夢境模式（只讀）

**使用示例**：
```python
kv.set_dream_mode(True)

# 可以讀取
value = kv.get("user:name")

# 不能寫入
try:
    kv.put("new_key", "new_value")
except PermissionError:
    print("夢境模式下不能寫入")

kv.set_dream_mode(False)
# 現在可以寫入了
kv.put("new_key", "new_value")
```

### 8.6 刪除與清理

```python
def delete(self, key: str) -> bool
def clear(self)
def close(self)
```

**使用示例**：
```python
# 刪除單個
deleted = kv.delete("user:name")
print(f"刪除成功: {deleted}")

# 清空所有
kv.clear()

# 關閉（釋放資源）
kv.close()
```

---

## 9. 學習數據庫 (LearnDatabase)

`LearnDatabase` 類實現帶演化和夢境重組能力的學習存儲。

### 9.1 初始化

```python
def __init__(self, encoder: HDCVectorOps, capacity: int = 100000,
             db_path: Optional[str] = None,
             semantic_encoder: Optional[BitSemanticEncoder] = None)
```

**使用示例**：
```python
encoder = HDCVectorOps(dim=1764)
semantic = BitSemanticEncoder(encoder, dim_bits=1764)

db = LearnDatabase(
    encoder=encoder,
    capacity=100000,
    db_path="./data/learn.db",
    semantic_encoder=semantic
)
```

### 9.2 學習數據

```python
def learn(
    self,
    input_data: Any,
    label: Optional[str] = None,
    modality: str = "auto",
    surprise_score: float = 0.5,
    initial_energy: float = 100.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> LearnRecord
```

**使用示例**：
```python
encoder = HDCVectorOps(dim=1764)
db = LearnDatabase(encoder, capacity=10000)

# 學習文本
record1 = db.learn(
    input_data="Python is a great programming language",
    label="programming",
    modality="text",
    surprise_score=0.8
)

# 學習圖像
image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
record2 = db.learn(
    input_data=image,
    label="random_image",
    modality="image"
)

# 學習結構化數據
data = {"type": "event", "action": "click", "target": "button"}
record3 = db.learn(
    input_data=data,
    label="user_interaction",
    modality="structured",
    metadata={"source": "web", "timestamp": "2025-01-01T12:00:00"}
)

print(f"記錄 ID: {record1.record_id}")
print(f"能量: {record1.energy}")
print(f"模態: {record1.modality}")
```

### 9.3 直接向量學習

```python
def learn_vector(
    self,
    vector: np.ndarray,
    input_data: Any = None,
    label: Optional[str] = None,
    modality: str = "vector",
    surprise_score: float = 0.5,
    initial_energy: float = 100.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> LearnRecord
```

**功能**：直接使用外部提供的超維向量學習

**使用示例**：
```python
# 假設有外部生成的向量
external_vector = encoder.random_vector(seed=12345)

record = db.learn_vector(
    vector=external_vector,
    input_data={"source": "4D_world"},
    label="scene_001",
    modality="world",
    metadata={"scene_id": 1}
)
```

### 9.4 演化反饋

```python
def evolve(self, record_id: str, feedback: float) -> bool
```

**功能**：根據反饋調整記錄的能量和信任度

**使用示例**：
```python
record = db.learn("Test concept", label="test", initial_energy=50.0)

# 正向反饋（增加能量）
db.evolve(record.record_id, feedback=0.8)
print(f"能量增加到: {record.energy}")

# 負向反饋（減少能量）
db.evolve(record.record_id, feedback=-0.5)
print(f"能量減少到: {record.energy}")

print(f"訓練迭代次數: {record.training_iterations}")
print(f"演化代數: {record.evolution_generation}")
```

### 9.5 語義查詢

```python
def query_similar(self, query: Any, k: int = 10) -> List[Tuple[LearnRecord, float]]
```

**使用示例**：
```python
# 存入一些數據
for i in range(100):
    db.learn(f"Concept about topic {i % 10}", label=f"topic_{i % 10}")

# 查詢相似記錄
results = db.query_similar("topic 5", k=5)
for record, similarity in results:
    print(f"[{record.label}] 相似度: {similarity:.3f}")
```

### 9.6 標籤查詢

```python
def query_by_label(self, label: str) -> List[LearnRecord]
```

**使用示例**：
```python
# 查詢特定標籤的所有記錄
programming_records = db.query_by_label("programming")
print(f"找到 {len(programming_records)} 條記錄")
```

### 9.7 夢境重組

```python
def dream_reorganize(self, concepts: List[LearnRecord]) -> List[LearnRecord]
```

**功能**：在夢境模式下重組概念，創建新的知識連接

**使用示例**：
```python
# 必須在夢境模式下
db.set_dream_mode(True)

# 獲取一些概念
concepts = list(db.concepts())[:10]

# 重組
new_concepts = db.dream_reorganize(concepts)
print(f"創建了 {len(new_concepts)} 個新概念")

for nc in new_concepts:
    print(f"  夢境概念: {nc.label}, 能量={nc.energy}")

db.set_dream_mode(False)
```

### 9.8 夢境清理

```python
def dream_cleanup(self, energy_threshold: float = 10.0) -> int
```

**功能**：清理低能量記錄

**使用示例**：
```python
db.set_dream_mode(True)

# 清理能量低於 10 的記錄
removed = db.dream_cleanup(energy_threshold=10.0)
print(f"移除了 {removed} 條低能量記錄")

db.set_dream_mode(False)
```

### 9.9 能量衰減

```python
def decay_energy(self, rate: float = 0.99)
```

**功能**：對所有記錄應用能量衰減

**使用示例**：
```python
# 每次調用使能量乘以衰減率
db.decay_energy(rate=0.99)

# 模擬時間流逝
for _ in range(100):
    db.decay_energy(rate=0.99)
```

### 9.10 獲取高能量記錄

```python
def get_top_energy(self, n: int) -> List[LearnRecord]
```

**使用示例**：
```python
top_records = db.get_top_energy(10)
for r in top_records:
    print(f"{r.label}: 能量={r.energy:.1f}, 信任={r.trust_score:.2f}")
```

---

## 10. 滾動數據集管理器 (RollingDatasetManager)

`RollingDatasetManager` 類管理具有自動淘汰機制的數據集。

### 10.1 初始化

```python
def __init__(self,
             learn_db: LearnDatabase,
             storage_dir: str,
             config: Optional[ASMDatasetConfig] = None)
```

**使用示例**：
```python
from ASMmyDatasets import ASMDatasetConfig

config = ASMDatasetConfig(
    storage_budget_mb=1000,
    max_samples=100000,
    batch_size=100
)

manager = RollingDatasetManager(
    learn_db=learn_db,
    storage_dir="./data/rolling",
    config=config
)
```

### 10.2 添加樣本

```python
def add_sample(self, data: Dict[str, Any],
               surprise_score: float,
               source_dataset: str = "unknown") -> RollingDatasetSample
```

**使用示例**：
```python
sample = manager.add_sample(
    data={"text": "New learning sample", "features": [1, 2, 3]},
    surprise_score=0.75,
    source_dataset="web_crawl"
)

print(f"樣本 ID: {sample.sample_id}")
print(f"驚奇度: {sample.surprise_score}")
```

### 10.3 獲取訓練批次

```python
def get_training_batch(self, batch_size: int = 32,
                       priority: str = "surprise") -> List[RollingDatasetSample]
```

**參數**：
- `priority`: 
  - `"surprise"`: 優先高驚奇度
  - `"random"`: 隨機
  - `"recent"`: 優先最新

**使用示例**：
```python
# 獲取高驚奇度優先的批次
batch = manager.get_training_batch(batch_size=32, priority="surprise")

for sample in batch:
    print(f"處理: {sample.sample_id}, 驚奇度={sample.surprise_score:.2f}")
```

### 10.4 獲取統計

```python
def get_stats(self) -> Dict[str, Any]
```

**使用示例**：
```python
stats = manager.get_stats()
print(f"總添加: {stats['total_added']}")
print(f"總淘汰: {stats['total_evicted']}")
print(f"高驚奇度數: {stats['high_surprise_count']}")
print(f"當前大小: {stats['current_size']}")
print(f"緩存命中率: {stats['cache']['hit_rate']:.2%}")
```

---

## 11. 自動訓練器 (AutoTrainer)

`AutoTrainer` 類實現基於驚奇度的自動學習決策。

### 11.1 初始化

```python
def __init__(self,
             learn_db: LearnDatabase,
             rolling_dataset: RollingDatasetManager,
             encoder: HDCVectorOps)
```

### 11.2 計算驚奇度

```python
def compute_surprise(self, data: Dict[str, Any]) -> float
```

**功能**：計算輸入數據的驚奇度（0~1），基於與現有知識的相似度

**使用示例**：
```python
trainer = AutoTrainer(learn_db, rolling_dataset, encoder)

# 計算驚奇度
surprise = trainer.compute_surprise({"text": "New concept"})
print(f"驚奇度: {surprise:.3f}")

# 驚奇度解釋
if surprise < 0.37:
    print("低驚奇度：強化現有知識")
elif surprise > 0.63:
    print("高驚奇度：學習新知識")
else:
    print("中等驚奇度：忽略")
```

### 11.3 處理樣本

```python
def process_sample(self, data: Dict[str, Any],
                   source: str = "unknown",
                   label: Optional[str] = None) -> TrainingEvent
```

**功能**：根據驚奇度自動決定處理策略

**使用示例**：
```python
event = trainer.process_sample(
    data={"text": "Important new information"},
    source="user_input",
    label="knowledge"
)

print(f"事件 ID: {event.event_id}")
print(f"驚奇度: {event.surprise_score:.3f}")
print(f"動作: {event.action}")  # "learn", "strengthen", 或 "ignore"
```

### 11.4 批量訓練

```python
def train_batch(self, samples: List[Dict[str, Any]],
                source: str = "batch") -> List[TrainingEvent]
```

**使用示例**：
```python
samples = [
    {"text": f"Sample {i}", "value": i}
    for i in range(100)
]

events = trainer.train_batch(samples, source="batch_import")
print(f"處理了 {len(events)} 個樣本")

# 統計動作分布
actions = [e.action for e in events]
print(f"學習: {actions.count('learn')}")
print(f"強化: {actions.count('strengthen')}")
print(f"忽略: {actions.count('ignore')}")
```

### 11.5 後台訓練

```python
def start_background_training(self, interval: float = 1.0)
def stop_background_training(self)
```

**使用示例**：
```python
# 啟動後台訓練
trainer.start_background_training(interval=0.5)

# 添加一些數據供訓練
for i in range(100):
    rolling_dataset.add_sample(
        data={"text": f"Training sample {i}"},
        surprise_score=0.7
    )

time.sleep(5)  # 讓後台處理

# 停止後台訓練
trainer.stop_background_training()

# 查看統計
stats = trainer.get_stats()
print(f"已處理: {stats['total_events']}")
```

---

## 12. 夢境模組 (EnhancedDreamModule)

`EnhancedDreamModule` 類實現類人腦的記憶鞏固機制。

### 12.1 初始化

```python
def __init__(self,
             kv_store: KVStore,
             learn_db: LearnDatabase,
             encoder: HDCVectorOps)
```

### 12.2 夢境循環

```python
def dream_cycle(self, full_cycle: bool = True) -> Dict[str, Any]
```

**功能**：執行完整或部分夢境循環

**夢境階段**：
1. **REM（快速眼動）**：
   - 隨機訪問 KV Store
   - 重組 Learn 概念，創建新連接
   
2. **NREM（非快速眼動）**：
   - 鞏固高能量概念
   - 清理低能量記錄

**使用示例**：
```python
dream_module = EnhancedDreamModule(kv_store, learn_db, encoder)

# 執行完整夢境循環
result = dream_module.dream_cycle(full_cycle=True)

print(f"新概念創建: {result['new_concepts_created']}")
print(f"概念強化: {result['concepts_strengthened']}")
print(f"概念移除: {result['concepts_removed']}")
print(f"KV 查詢: {result['kv_queries']}")
print(f"耗時: {result['duration_ms']:.2f} ms")

# 只執行 REM 階段
partial_result = dream_module.dream_cycle(full_cycle=False)
```

### 12.3 獲取統計

```python
def get_stats(self) -> Dict[str, Any]
```

**使用示例**：
```python
stats = dream_module.get_stats()
print(f"夢境循環次數: {stats['dream_cycles']}")
print(f"總 KV 查詢: {stats['kv_queries']}")
print(f"總重組: {stats['learn_reorganized']}")
print(f"總清理: {stats['learn_cleaned']}")
print(f"是否正在做夢: {stats['is_dreaming']}")
```

---

## 13. 多模態渲染器 (MultiModalRenderer)

`MultiModalRenderer` 類將各種數據類型渲染為圖像/視頻。

### 13.1 初始化

```python
def __init__(self,
             output_dir: str,
             width: int = 256,
             height: int = 256,
             fps: int = 30)
```

**使用示例**：
```python
renderer = MultiModalRenderer(
    output_dir="./outputs",
    width=512,
    height=512,
    fps=30
)
```

### 13.2 設置調色板

```python
def set_palette(self, palette: str)
```

**使用示例**：
```python
renderer.set_palette("plasma")  # 或 "viridis", "fire"
```

### 13.3 數據轉圖像

```python
def data_to_image(self, data: Any, effect: str = "none") -> np.ndarray
```

**支持的數據類型**：
- NumPy 數組（1D, 2D, 3D）
- 字典
- 字符串
- bytes

**效果類型**：
- `"none"`: 無效果
- `"glow"`: 發光效果
- `"wave"`: 波浪效果
- `"fire"`: 火焰效果

**使用示例**：
```python
renderer = MultiModalRenderer("./outputs", width=256, height=256)

# 2D 數組
data_2d = np.random.rand(64, 64)
image1 = renderer.data_to_image(data_2d, effect="none")

# 帶效果
image2 = renderer.data_to_image(data_2d, effect="glow")
image3 = renderer.data_to_image(data_2d, effect="wave")
image4 = renderer.data_to_image(data_2d, effect="fire")

# 字典
data_dict = {"key": "value", "numbers": [1, 2, 3]}
image5 = renderer.data_to_image(data_dict)

# RGB 圖像直接傳遞
rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
image6 = renderer.data_to_image(rgb)
```

### 13.4 幀管理

```python
def add_frame(self, data: Any, effect: str = "none")
def add_rgb_frame(self, frame_rgb_u8: np.ndarray)
def clear_frames(self)
def get_frame_count(self) -> int
```

**使用示例**：
```python
renderer.clear_frames()

# 添加動畫幀
for i in range(60):
    data = np.sin(np.linspace(0, 4*np.pi, 64*64) + i/10).reshape(64, 64)
    renderer.add_frame(data, effect="wave")

print(f"總幀數: {renderer.get_frame_count()}")

# 直接添加 RGB 幀
rgb_frame = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
renderer.add_rgb_frame(rgb_frame)
```

### 13.5 保存輸出

```python
def save_json(self, data: Any, filename: str) -> str
def save_png(self, data: Any = None, filename: str = "output.png",
             effect: str = "none") -> str
def save_gif(self, filename: str = "output.gif",
             duration: Optional[int] = None,
             loop: int = 0) -> str
def save_mp4(self, filename: str = "output.mp4",
             codec: str = "mp4v") -> str
def save_all(self, data: Any = None, prefix: str = "output",
             effect: str = "none") -> Dict[str, str]
```

**使用示例**：
```python
renderer = MultiModalRenderer("./outputs", width=256, height=256)

# 保存 JSON
data = {"result": [1, 2, 3], "status": "success"}
json_path = renderer.save_json(data, "result.json")

# 保存 PNG
array_data = np.random.rand(64, 64)
png_path = renderer.save_png(array_data, "visualization.png", effect="glow")

# 創建動畫幀
renderer.clear_frames()
for i in range(30):
    frame_data = np.random.rand(64, 64) * (i / 30)
    renderer.add_frame(frame_data)

# 保存 GIF
gif_path = renderer.save_gif("animation.gif", duration=50)

# 保存 MP4
mp4_path = renderer.save_mp4("animation.mp4")

# 一次保存所有格式
files = renderer.save_all(data, prefix="complete_output", effect="wave")
print(f"保存的文件: {files}")
```

---

## 14. 4D 世界模組

4D 世界模組提供真實 3D/4D 空間的渲染和圖文對齊能力。

### 14.1 環境規格 (WorldEnvSpec)

```python
@dataclass
class WorldEnvSpec:
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ambient_color: Tuple[float, float, float] = (0.2, 0.2, 0.25)
    light_direction: Tuple[float, float, float] = (1.0, -1.0, -0.5)
    light_color: Tuple[float, float, float] = (1.0, 0.98, 0.95)
    background_color: Tuple[float, float, float] = (0.05, 0.05, 0.08)
```

### 14.2 相機規格 (WorldCameraSpec)

```python
@dataclass
class WorldCameraSpec:
    position: Tuple[float, float, float] = (6.0, 5.0, 6.0)
    target: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    width: float = 10.0
    height: float = 7.5
    image_width: int = 800
    image_height: int = 600
    time: float = 0.0
```

### 14.3 物件規格 (WorldObjectSpec)

```python
@dataclass
class WorldObjectSpec:
    name: str
    kind: str = "box"  # box/plane/sphere/asm4dobject
    params: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    texture_rgb: Optional[np.ndarray] = None
    diffuse: float = 0.9
    ambient: float = 0.1
    affected_by_gravity: bool = False
```

**使用示例**：
```python
# 方塊
box = WorldObjectSpec(
    name="cube_1",
    kind="box",
    params={"width": 1.0, "height": 1.5, "depth": 1.0},
    position=(0.0, 0.75, 0.0),
    rotation=(0.0, 0.5, 0.0),
    base_color=(0.8, 0.2, 0.2)
)

# 球體
sphere = WorldObjectSpec(
    name="ball_1",
    kind="sphere",
    params={"radius": 0.5},
    position=(2.0, 0.5, 0.0),
    base_color=(0.2, 0.8, 0.2)
)

# 帶貼圖的平面
texture = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
plane = WorldObjectSpec(
    name="textured_plane",
    kind="plane",
    params={"width": 4.0, "height": 3.0},
    position=(0.0, 2.0, -2.0),
    base_color=(1.0, 1.0, 1.0),
    texture_rgb=texture
)
```

### 14.4 文字規格 (WorldTextSpec)

```python
@dataclass
class WorldTextSpec:
    text: str
    position: Tuple[float, float, float] = (0.0, 1.5, 0.0)
    size: Tuple[float, float] = (2.5, 0.8)  # (width, height)
    font_size: int = 40
    fg: Tuple[int, int, int] = (10, 10, 10)
    bg: Tuple[int, int, int] = (245, 245, 245)
    rotation: Optional[Tuple[float, float, float]] = None
    billboard: bool = True  # 自動面向相機
```

**使用示例**：
```python
# 廣告牌文字（自動面向相機）
text_billboard = WorldTextSpec(
    text="Hello 4D World!",
    position=(0.0, 3.0, 0.0),
    size=(5.0, 1.5),
    font_size=48,
    fg=(0, 0, 0),
    bg=(255, 255, 255),
    billboard=True
)

# 固定方向文字
text_fixed = WorldTextSpec(
    text="Fixed orientation",
    position=(2.0, 1.0, 2.0),
    size=(3.0, 0.8),
    rotation=(0.0, 0.785, 0.0),  # 45度
    billboard=False
)
```

### 14.5 場景規格 (WorldSceneSpec)

```python
@dataclass
class WorldSceneSpec:
    env: WorldEnvSpec = field(default_factory=WorldEnvSpec)
    camera: WorldCameraSpec = field(default_factory=WorldCameraSpec)
    objects: List[WorldObjectSpec] = field(default_factory=list)
    texts: List[WorldTextSpec] = field(default_factory=list)
    seed: Optional[int] = None
    description: str = ""
```

**完整場景示例**：
```python
scene = WorldSceneSpec(
    env=WorldEnvSpec(
        ambient_color=(0.2, 0.2, 0.25),
        light_direction=(1.0, -1.0, -0.5),
        background_color=(0.05, 0.05, 0.08)
    ),
    camera=WorldCameraSpec(
        position=(6.0, 5.0, 6.0),
        target=(0.0, 1.0, 0.0),
        image_width=800,
        image_height=600
    ),
    objects=[
        WorldObjectSpec(
            name="floor",
            kind="plane",
            params={"width": 10.0, "height": 10.0},
            position=(0.0, 0.0, 0.0),
            base_color=(0.5, 0.5, 0.5)
        ),
        WorldObjectSpec(
            name="cube",
            kind="box",
            params={"width": 1.0, "height": 1.0, "depth": 1.0},
            position=(0.0, 0.5, 0.0),
            base_color=(0.8, 0.2, 0.2)
        )
    ],
    texts=[
        WorldTextSpec(
            text="3D Scene Demo",
            position=(0.0, 2.5, 0.0),
            size=(4.0, 1.0),
            billboard=True
        )
    ],
    description="A simple demonstration scene"
)
```

### 14.6 場景生成器 (WorldSceneGenerator)

```python
class WorldSceneGenerator:
    @staticmethod
    def random_scene(seed: Optional[int] = None,
                     image_width: int = 800,
                     image_height: int = 600) -> WorldSceneSpec
    
    @staticmethod
    def showroom_scene_with_texture(seed: Optional[int] = None,
                                    image_width: int = 800,
                                    image_height: int = 600) -> WorldSceneSpec
```

**使用示例**：
```python
# 生成隨機場景
random_scene = WorldSceneGenerator.random_scene(seed=42)
print(f"場景描述: {random_scene.description}")
print(f"物件數量: {len(random_scene.objects)}")

# 生成帶貼圖的展廳場景
showroom = WorldSceneGenerator.showroom_scene_with_texture(seed=123)
```

### 14.7 世界渲染器 (World4DRenderer)

```python
class World4DRenderer:
    def __init__(self, output_dir: str)
    
    def render(self, scene: WorldSceneSpec, 
               save_path: Optional[str] = None) -> SceneObservation
    
    def save_rgb(self, rgb: np.ndarray, path: str) -> str
```

**使用示例**：
```python
# 需要 ASMOrigin4DEngine
if HAS_4D_ENGINE:
    renderer = World4DRenderer(output_dir="./outputs")
    
    # 創建場景
    scene = WorldSceneGenerator.random_scene(seed=42)
    
    # 渲染
    observation = renderer.render(scene, save_path="./outputs/scene.png")
    
    # 訪問結果
    rgb_image = observation.rgb
    metadata = observation.meta
    
    print(f"圖像尺寸: {rgb_image.shape}")
    print(f"物件數量: {len(metadata['objects'])}")
    print(f"文字數量: {len(metadata['texts'])}")
```

---

## 15. 主系統類 (ASMsuperAIsystem)

`ASMsuperAIsystem` 是整合所有功能的主系統類。

### 15.1 初始化

```python
def __init__(self,
             dim: int = DEFAULT_VECTOR_DIM,
             storage_dir: str = SUPER_AI_DIR,
             auto_load: bool = True)
```

**使用示例**：
```python
# 創建系統
system = ASMsuperAIsystem(
    dim=1764,
    storage_dir="./my_ai_data",
    auto_load=True  # 自動加載持久化數據
)

# 查看組件
print(f"向量維度: {system.dim}")
print(f"有 4D 世界: {system.world4d is not None}")
```

### 15.2 K,V 輸入

```python
def input_kv(self, key: str, value: Any,
             value_type: Optional[str] = None,
             metadata: Optional[Dict] = None) -> KVRecord
```

**使用示例**：
```python
system = ASMsuperAIsystem()

# 存儲各種數據
system.input_kv("user:name", "Alice")
system.input_kv("user:age", 30)
system.input_kv("user:profile", {"role": "admin"})

# 帶元數據
record = system.input_kv(
    "document:123",
    "Important content",
    metadata={"author": "Bob", "created": "2025-01-01"}
)
```

### 15.3 學習輸入

```python
def input_learn(self, input_data: Any,
                label: Optional[str] = None,
                modality: str = "auto",
                metadata: Optional[Dict[str, Any]] = None) -> LearnRecord
```

**使用示例**：
```python
# 學習文本
record1 = system.input_learn(
    "Machine learning is a subset of AI",
    label="ml_definition"
)

# 學習圖像
image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
record2 = system.input_learn(
    image,
    label="sample_image",
    modality="image"
)

# 學習結構化數據
data = {"event": "click", "target": "button", "timestamp": 12345}
record3 = system.input_learn(
    data,
    label="user_event",
    modality="structured"
)
```

### 15.4 混合輸入

```python
def input_mix(self, input_data: Any, mode_config: Dict[str, Any]) -> Dict[str, Any]
```

**使用示例**：
```python
config = {
    "kv_operations": [
        {"action": "put", "key": "setting:theme", "value": "dark"},
        {"action": "get", "key": "user:name"}
    ],
    "learn_operations": [
        {"action": "learn", "label": "new_knowledge"},
        {"action": "query", "k": 5}
    ]
}

result = system.input_mix(
    input_data={"text": "Mixed mode input"},
    mode_config=config
)

print(f"KV 結果: {result['kv_results']}")
print(f"Learn 結果: {result['learn_results']}")
```

### 15.5 4D 世界渲染

```python
def render_world_scene(self, scene: WorldSceneSpec,
                       save_prefix: Optional[str] = None) -> SceneObservation
```

**使用示例**：
```python
if system.world4d is not None:
    scene = WorldSceneGenerator.random_scene(seed=42)
    
    observation = system.render_world_scene(
        scene,
        save_prefix="my_scene"
    )
    
    print(f"渲染完成: {observation.rgb.shape}")
```

### 15.6 4D 世界訓練

```python
def train_world_scene(self, scene: WorldSceneSpec,
                      label: Optional[str] = "world_scene",
                      save_prefix: Optional[str] = None,
                      source: str = "world") -> Dict[str, Any]
```

**功能**：渲染 4D 場景並作為訓練樣本存入系統

**使用示例**：
```python
if system.world4d is not None:
    scene = WorldSceneGenerator.showroom_scene_with_texture(seed=123)
    
    result = system.train_world_scene(
        scene,
        label="showroom",
        save_prefix="trained_scene",
        source="world_training"
    )
    
    print(f"記錄 ID: {result['record_id']}")
    print(f"驚奇度: {result['surprise']:.3f}")
    print(f"圖像路徑: {result['image_path']}")
```

### 15.7 想像回灌訓練

```python
def imagination_self_train(self,
                           cycles: int = 1,
                           scenes_per_cycle: int = 5,
                           save_dir: Optional[str] = None) -> Dict[str, Any]
```

**功能**：自動生成虛擬場景並用於自我訓練

**使用示例**：
```python
if system.world4d is not None:
    result = system.imagination_self_train(
        cycles=3,
        scenes_per_cycle=10,
        save_dir="./imagination_outputs"
    )
    
    print(f"訓練循環: {result['cycles']}")
    print(f"每循環場景: {result['scenes_per_cycle']}")
    print(f"總訓練數: {result['trained']}")
    print(f"耗時: {result['elapsed_s']:.2f} 秒")
```

### 15.8 查詢輸出

```python
def output_query(self, query: Any,
                 query_type: str = "auto",
                 k: int = 10) -> Dict[str, Any]
```

**使用示例**：
```python
# 存入一些數據
system.input_kv("doc:1", "Python programming guide")
system.input_learn("Neural networks are powerful", label="nn")

# 查詢
result = system.output_query("programming", k=5)

print(f"查詢: {result['query']}")
print(f"延遲: {result['latency_ms']:.2f} ms")
for match in result['matches']:
    print(f"  {match['source']}: 相似度={match['similarity']:.3f}")
```

### 15.9 RGA 輸出（檢索增強生成）

```python
def output_rga(self, input_data: Any,
               context_k: int = 5,
               generation_steps: int = 50) -> Dict[str, Any]
```

**使用示例**：
```python
result = system.output_rga(
    input_data="Generate something about AI",
    context_k=5,
    generation_steps=100
)

print(f"檢索到的上下文: {len(result['retrieved_context'])}")
print(f"生成結果形狀: {np.array(result['generated']).shape}")
print(f"延遲: {result['latency_ms']:.2f} ms")
```

### 15.10 生成輸出

```python
def output_generate(self,
                    context: Optional[np.ndarray] = None,
                    mode: str = "evolution",
                    steps: int = 100,
                    effect: str = "none") -> Dict[str, Any]
```

**模式**：
- `"evolution"`: 演化引擎生成
- `"world"` / `"4d"` / `"scene"`: 4D 世界生成

**使用示例**：
```python
# 演化生成
result1 = system.output_generate(
    mode="evolution",
    steps=100,
    effect="wave"
)
print(f"生成幀數: {result1['frame_count']}")

# 4D 世界生成
if system.world4d is not None:
    result2 = system.output_generate(
        mode="world",
        steps=1
    )
    print(f"世界元數據: {result2.get('world_meta', {}).get('description')}")
```

### 15.11 訓練

```python
def train(self, data: Dict[str, Any],
          source: str = "manual",
          label: Optional[str] = None) -> TrainingEvent

def train_batch(self, samples: List[Dict[str, Any]],
                source: str = "batch") -> List[TrainingEvent]
```

**使用示例**：
```python
# 單個訓練
event = system.train(
    data={"text": "New knowledge", "value": 42},
    source="user_input",
    label="knowledge"
)
print(f"動作: {event.action}, 驚奇度: {event.surprise_score:.3f}")

# 批量訓練
samples = [{"text": f"Sample {i}"} for i in range(100)]
events = system.train_batch(samples, source="batch_import")
```

### 15.12 自動訓練

```python
def start_auto_training(self, interval: float = 1.0)
def stop_auto_training(self)
```

**使用示例**：
```python
# 啟動
system.start_auto_training(interval=0.5)

# ... 添加數據 ...

# 停止
system.stop_auto_training()
```

### 15.13 夢境

```python
def dream(self, full_cycle: bool = True) -> Dict[str, Any]
```

**使用示例**：
```python
# 執行夢境循環
result = system.dream(full_cycle=True)

print(f"新概念: {result['new_concepts_created']}")
print(f"強化: {result['concepts_strengthened']}")
print(f"清理: {result['concepts_removed']}")
```

### 15.14 渲染與保存

```python
def render_to_image(self, data: Any,
                    effect: str = "none",
                    save: bool = False,
                    filename: str = "output.png") -> np.ndarray

def save_output(self, data: Any,
                prefix: str = "output",
                formats: List[str] = None,
                effect: str = "none") -> Dict[str, str]
```

**使用示例**：
```python
# 渲染為圖像
data = np.random.rand(64, 64)
image = system.render_to_image(data, effect="glow", save=True, filename="test.png")

# 保存多格式
files = system.save_output(
    data={"result": [1, 2, 3]},
    prefix="my_output",
    formats=["json", "png", "gif"],
    effect="wave"
)
print(f"保存的文件: {files}")
```

### 15.15 系統統計

```python
def get_stats(self) -> SystemStats
```

**使用示例**：
```python
stats = system.get_stats()

print(f"KV 存儲大小: {stats.kv_store_size}")
print(f"學習數據庫大小: {stats.learn_db_size}")
print(f"滾動數據集大小: {stats.rolling_dataset_size}")
print(f"總處理樣本: {stats.total_samples_processed}")
print(f"總訓練事件: {stats.total_training_events}")
print(f"總夢境循環: {stats.total_dream_cycles}")
print(f"存儲使用: {stats.storage_usage_mb:.2f} MB")
```

### 15.16 關閉系統

```python
def shutdown(self)
```

**使用示例**：
```python
# 正確關閉系統
system.shutdown()
```

---

## 16. 數據結構

### 16.1 KVRecord

```python
@dataclass
class KVRecord:
    key: str              # 鍵名
    value: Any            # 值
    value_type: str       # 類型 ("text", "number", "dict", ...)
    vector: np.ndarray    # 超維向量
    created: str          # 創建時間 (ISO 格式)
    accessed: int = 0     # 訪問次數
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 16.2 LearnRecord

```python
@dataclass
class LearnRecord:
    record_id: str                        # 唯一 ID
    input_data: Any                       # 原始輸入
    label: Optional[str]                  # 標籤
    vector: np.ndarray                    # 超維向量
    kernel_config: Optional[np.ndarray]   # 核配置
    energy: float = 100.0                 # 能量 (0~200)
    trust_score: float = 1.0              # 信任度 (0~1)
    created: str                          # 創建時間
    modified: str                         # 修改時間
    training_iterations: int = 0          # 訓練迭代次數
    evolution_generation: int = 0         # 演化代數
    modality: str = "unknown"             # 模態
    can_dream_reorganize: bool = True     # 可否夢境重組
    surprise_score: float = 0.5           # 驚奇度
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 16.3 TrainingEvent

```python
@dataclass
class TrainingEvent:
    event_id: str                         # 事件 ID
    sample_id: str                        # 樣本 ID
    surprise_score: float                 # 驚奇度
    action: str                           # 動作 ("learn", "strengthen", "ignore")
    timestamp: str                        # 時間戳
    details: Dict[str, Any] = field(default_factory=dict)
```

### 16.4 SceneObservation

```python
@dataclass
class SceneObservation:
    rgb: np.ndarray          # RGB 圖像 (H, W, 3), uint8
    meta: Dict[str, Any]     # 場景元數據
```

### 16.5 SystemStats

```python
@dataclass
class SystemStats:
    kv_store_size: int = 0
    learn_db_size: int = 0
    rolling_dataset_size: int = 0
    total_samples_processed: int = 0
    total_training_events: int = 0
    total_dream_cycles: int = 0
    storage_usage_mb: float = 0.0
    avg_surprise_score: float = 0.5
```

---

## 17. 工具函數

### 17.1 穩定哈希

```python
def stable_hash64(data: Any, salt: str = "") -> int
def stable_hash32(data: Any, salt: str = "") -> int
```

**功能**：生成跨進程穩定的哈希值（解決 Python hash 隨機化問題）

**使用示例**：
```python
h1 = stable_hash64("hello", salt="my_salt")
h2 = stable_hash64("hello", salt="my_salt")
assert h1 == h2  # 總是相同

h3 = stable_hash32("hello")
print(f"32位哈希: {h3}")
```

### 17.2 RGB 格式確保

```python
def ensure_rgb_u8(img: np.ndarray) -> np.ndarray
```

**功能**：將各種格式的圖像轉換為 RGB uint8

**使用示例**：
```python
# 灰度圖
gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
rgb = ensure_rgb_u8(gray)
print(f"形狀: {rgb.shape}")  # (64, 64, 3)

# RGBA
rgba = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
rgb = ensure_rgb_u8(rgba)

# Float 圖像
float_img = np.random.rand(64, 64, 3)
rgb = ensure_rgb_u8(float_img)
print(f"類型: {rgb.dtype}")  # uint8
```

### 17.3 位打包

```python
def pack_bits_to_u64(bits: np.ndarray, out_words: int) -> np.ndarray
```

**功能**：將 0/1 位數組打包為 uint64

**使用示例**：
```python
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 8, dtype=np.uint8)
packed = pack_bits_to_u64(bits, out_words=1)
print(f"打包結果: {packed}")
```

---

## 18. 測試與性能基準

### 18.1 運行所有測試

```python
def testAll() -> bool
```

**使用示例**：
```python
# 運行完整測試套件
all_passed = testAll()

if all_passed:
    print("所有測試通過！")
else:
    print("有測試失敗")
```

### 18.2 性能基準

```python
class PerformanceBenchmark:
    @staticmethod
    def benchmark_kv_operations()
    
    @staticmethod
    def benchmark_learn_operations()
    
    @staticmethod
    def benchmark_surprise_calculation()
    
    @staticmethod
    def benchmark_rendering()
    
    @staticmethod
    def benchmark_4d_world_rendering()
    
    @staticmethod
    def run_all()
```

**使用示例**：
```python
# 運行所有性能測試
PerformanceBenchmark.run_all()

# 或單獨運行
PerformanceBenchmark.benchmark_kv_operations()
```

### 18.3 演示生成

```python
def generate_demo_outputs()
```

**使用示例**：
```python
# 生成完整演示輸出
generate_demo_outputs()
# 輸出到 ./superAI/demos/
```

---

## 19. 完整使用範例

### 19.1 基礎工作流

```python
from ASMOriginAI import (
    ASMsuperAIsystem, WorldSceneGenerator,
    HAS_4D_ENGINE, HAS_PIL
)
import numpy as np

# 1. 創建系統
system = ASMsuperAIsystem(
    dim=1764,
    storage_dir="./my_ai_project",
    auto_load=True
)

# 2. K,V 存儲
system.input_kv("config:version", "1.0.0")
system.input_kv("user:preferences", {"theme": "dark", "language": "zh"})

# 3. 學習數據
system.input_learn(
    "深度學習是機器學習的一個子領域",
    label="dl_definition"
)

# 4. 圖像學習
if HAS_PIL:
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[32:96, 32:96] = (255, 0, 0)  # 紅色方塊
    system.input_learn(image, label="red_square", modality="image")

# 5. 訓練
events = system.train_batch([
    {"text": f"Sample data {i}", "value": i}
    for i in range(50)
])
print(f"訓練了 {len(events)} 個樣本")

# 6. 夢境整理
dream_result = system.dream(full_cycle=True)
print(f"夢境創建了 {dream_result['new_concepts_created']} 個新概念")

# 7. 查詢
query_result = system.output_query("機器學習", k=5)
for match in query_result['matches']:
    print(f"  {match['source']}: {match.get('label', match.get('key', 'N/A'))}")

# 8. 生成輸出
gen_result = system.output_generate(mode="evolution", steps=50)
print(f"生成了 {gen_result['frame_count']} 幀")

# 9. 保存
if HAS_PIL:
    files = system.save_output(
        data={"query": query_result},
        prefix="demo_output",
        formats=["json", "png", "gif"]
    )
    print(f"保存了: {files}")

# 10. 統計
stats = system.get_stats()
print(f"系統統計: KV={stats.kv_store_size}, Learn={stats.learn_db_size}")

# 11. 關閉
system.shutdown()
```

### 19.2 4D 世界工作流

```python
from ASMOriginAI import (
    ASMsuperAIsystem, WorldSceneSpec, WorldObjectSpec, 
    WorldTextSpec, WorldEnvSpec, WorldCameraSpec,
    WorldSceneGenerator, HAS_4D_ENGINE
)
import numpy as np
import math

if not HAS_4D_ENGINE:
    print("4D 引擎不可用")
else:
    system = ASMsuperAIsystem(dim=1764, storage_dir="./4d_project")
    
    if system.world4d is not None:
        # 方法 1: 使用生成器創建隨機場景
        random_scene = WorldSceneGenerator.random_scene(seed=42)
        obs1 = system.render_world_scene(random_scene, save_prefix="random_scene")
        print(f"隨機場景渲染完成: {obs1.rgb.shape}")
        
        # 方法 2: 手動構建場景
        custom_scene = WorldSceneSpec(
            env=WorldEnvSpec(
                ambient_color=(0.15, 0.15, 0.2),
                background_color=(0.02, 0.02, 0.05)
            ),
            camera=WorldCameraSpec(
                position=(8.0, 6.0, 8.0),
                target=(0.0, 1.5, 0.0),
                image_width=1024,
                image_height=768
            ),
            objects=[
                WorldObjectSpec(
                    name="ground",
                    kind="plane",
                    params={"width": 12.0, "height": 12.0},
                    position=(0.0, 0.0, 0.0),
                    base_color=(0.4, 0.5, 0.4)
                ),
                WorldObjectSpec(
                    name="tower",
                    kind="box",
                    params={"width": 1.5, "height": 4.0, "depth": 1.5},
                    position=(0.0, 2.0, 0.0),
                    base_color=(0.7, 0.7, 0.8),
                    rotation=(0.0, math.pi/6, 0.0)
                ),
                WorldObjectSpec(
                    name="sphere",
                    kind="sphere",
                    params={"radius": 0.8},
                    position=(3.0, 0.8, 2.0),
                    base_color=(0.9, 0.3, 0.3)
                )
            ],
            texts=[
                WorldTextSpec(
                    text="自定義 4D 場景",
                    position=(0.0, 4.5, 0.0),
                    size=(6.0, 1.5),
                    font_size=48,
                    billboard=True
                )
            ],
            description="Custom 4D scene with tower and sphere"
        )
        
        obs2 = system.render_world_scene(custom_scene, save_prefix="custom_scene")
        print(f"自定義場景渲染完成: {obs2.rgb.shape}")
        
        # 方法 3: 帶貼圖的場景
        texture = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(256):
            for x in range(256):
                texture[y, x] = (
                    int(127 + 127 * math.sin(x / 20)),
                    int(127 + 127 * math.cos(y / 20)),
                    int(127 + 127 * math.sin((x + y) / 30))
                )
        
        textured_scene = WorldSceneGenerator.showroom_scene_with_texture(seed=123)
        # 替換貼圖
        for obj in textured_scene.objects:
            if obj.name == "image_plane":
                obj.texture_rgb = texture
        
        obs3 = system.render_world_scene(textured_scene, save_prefix="textured_scene")
        
        # 方法 4: 想像回灌訓練
        training_result = system.imagination_self_train(
            cycles=2,
            scenes_per_cycle=5,
            save_dir="./imagination"
        )
        print(f"想像訓練完成: 訓練了 {training_result['trained']} 個場景")
        
        # 方法 5: 訓練單個場景
        train_result = system.train_world_scene(
            custom_scene,
            label="my_custom_scene",
            save_prefix="trained_custom",
            source="manual"
        )
        print(f"場景訓練完成: 驚奇度={train_result['surprise']:.3f}")
    
    system.shutdown()
```

### 19.3 後台訓練與夢境循環

```python
from ASMOriginAI import ASMsuperAIsystem
import time
import random

system = ASMsuperAIsystem(dim=1764, storage_dir="./background_training")

# 啟動後台訓練
system.start_auto_training(interval=0.5)

# 模擬持續輸入
for i in range(100):
    # 隨機生成樣本
    sample = {
        "text": f"Continuous learning sample {i}",
        "value": random.random(),
        "category": random.choice(["A", "B", "C"])
    }
    
    # 添加到滾動數據集
    system.rolling_dataset.add_sample(
        data=sample,
        surprise_score=random.random(),
        source_dataset="stream"
    )
    
    # 每 20 個樣本執行一次夢境
    if i > 0 and i % 20 == 0:
        dream_result = system.dream(full_cycle=True)
        print(f"夢境 {i//20}: 創建={dream_result['new_concepts_created']}, "
              f"強化={dream_result['concepts_strengthened']}")
    
    time.sleep(0.1)

# 停止後台訓練
system.stop_auto_training()

# 最終統計
stats = system.get_stats()
trainer_stats = system.trainer.get_stats()
dream_stats = system.dream_module.get_stats()

print("\n=== 最終統計 ===")
print(f"學習數據庫大小: {stats.learn_db_size}")
print(f"總訓練事件: {trainer_stats['total_events']}")
print(f"  - 學習: {trainer_stats['learned']}")
print(f"  - 強化: {trainer_stats['strengthened']}")
print(f"  - 忽略: {trainer_stats['ignored']}")
print(f"夢境循環次數: {dream_stats['dream_cycles']}")

system.shutdown()
```

### 19.4 語義搜索應用

```python
from ASMOriginAI import ASMsuperAIsystem
import json

system = ASMsuperAIsystem(dim=1764, storage_dir="./semantic_search")

# 構建知識庫
documents = [
    {"id": 1, "title": "Python 入門", "content": "Python 是一種高級編程語言"},
    {"id": 2, "title": "機器學習基礎", "content": "機器學習讓計算機從數據中學習"},
    {"id": 3, "title": "深度學習", "content": "深度學習使用多層神經網絡"},
    {"id": 4, "title": "自然語言處理", "content": "NLP 讓計算機理解人類語言"},
    {"id": 5, "title": "計算機視覺", "content": "CV 讓計算機理解圖像和視頻"},
]

# 索引文檔
for doc in documents:
    system.input_kv(
        f"doc:{doc['id']}",
        doc,
        metadata={"indexed": True}
    )
    system.input_learn(
        f"{doc['title']} {doc['content']}",
        label=f"doc_{doc['id']}"
    )

# 語義搜索函數
def semantic_search(query: str, top_k: int = 3):
    result = system.output_query(query, k=top_k)
    
    print(f"\n查詢: '{query}'")
    print("-" * 50)
    
    for i, match in enumerate(result['matches'], 1):
        if match['source'] == 'kv_semantic':
            value = system.kv_store.get(match['key'])
            if isinstance(value, dict):
                print(f"{i}. [{match['similarity']:.3f}] {value.get('title', 'N/A')}")
                print(f"   {value.get('content', '')[:50]}...")
        elif match['source'] == 'learn':
            print(f"{i}. [{match['similarity']:.3f}] Label: {match['label']}")
    
    return result

# 測試搜索
semantic_search("如何學習編程")
semantic_search("神經網絡和 AI")
semantic_search("處理圖片")

system.shutdown()
```

---

## 附錄：環境變數

| 變數名 | 默認值 | 說明 |
|--------|--------|------|
| `SUPERAI_VERBOSE` | `"0"` | 設為 `"1"` 啟用詳細日誌 |
| `SUPERAI_KEEP_TEST_OUTPUTS` | `"1"` | 設為 `"0"` 不保留測試輸出 |
| `SUPERAI_DISABLE_4D` | `"0"` | 設為 `"1"` 強制停用 4D 模組 |

## 附錄：依賴安裝

```bash
# 必需
pip install numpy

# 推薦
pip install pillow     # 圖像處理
pip install opencv-python  # 視頻處理

# 可選（自定義模組）
# ASMOriginBitComputing - 加速位運算
# ASMOrigin4DEngine - 4D 渲染
# ASM4Dobjects - 4D 物件庫
# ASMoneMatrixBitData - 高品質文字渲染
# ASMbitWorldAIEngine - 位世界 AI 引擎
# ASMFileDict3 - 高效文件字典
# ASMmyDatasets - 數據集管理
```