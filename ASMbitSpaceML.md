
# ASMbitSpaceML.py 完整使用說明

## 目錄

1. [概述](#1-概述)
2. [環境變數配置](#2-環境變數配置)
3. [核心工具函數](#3-核心工具函數)
4. [統一媒體編解碼器](#4-統一媒體編解碼器-unifiedmediacodec)
5. [位元打包工具](#5-位元打包工具-packedbits)
6. [後端管理](#6-後端管理-bitmlbackend)
7. [神經網路層](#7-神經網路層)
8. [分類模型](#8-分類模型)
9. [HNN-Transformer 語言模型](#9-hnn-transformer-語言模型)
10. [HNN-GAN 圖像生成](#10-hnn-gan-圖像生成)
11. [訓練優化器](#11-訓練優化器)
12. [視覺化工具](#12-視覺化工具)
13. [MNIST 整合](#13-mnist-整合)
14. [相機處理](#14-相機處理)
15. [音樂生成](#15-音樂生成)
16. [應用層整合](#16-應用層整合-bitspaceapplication)
17. [測試與基準測試](#17-測試與基準測試)

---

## 1. 概述

`ASMbitSpaceML.py` 是一個純位元運算的機器學習框架，實現了「超空間神經網路」(Hyperspace Neural Networks, HNN)。主要特點：

- **純位元運算**：使用 XNOR + Popcount 進行高效的相似度計算
- **JIT 加速**：優先使用 `ASMbitJIT.py` 提供的極限內核
- **無需 matplotlib**：內建純 Python PNG 編碼器
- **統一編碼**：支援文字、圖像、音訊的可逆位元空間編碼
- **完整 ML 管線**：分類器、Transformer、GAN、視覺化工具

### 依賴項

```python
# 必要
import numpy as np

# 可選
import cv2                    # 僅相機 demo 使用
from ASMbitJIT import _BACKENDS  # JIT 加速
```

---

## 2. 環境變數配置

```python
import os

# ============= 基礎配置 =============
os.environ["BITSPACE_ML_OUTPUT_DIR"] = "./bitSpace"      # 輸出目錄
os.environ["BITSPACE_ML_DATA_DIR"] = "./bitSpaceData"    # 資料目錄
os.environ["BITSPACE_ML_ENABLE_JIT"] = "1"               # 啟用 JIT (0=強制 NumPy fallback)
os.environ["BITSPACE_ML_VERBOSE"] = "1"                  # 詳細日誌
os.environ["BITSPACE_ML_HEAVY_BENCH"] = "0"              # 重度基準測試

# ============= MNIST 配置 =============
os.environ["BITSPACE_ML_SKIP_MNIST"] = "0"               # 跳過 MNIST 測試
os.environ["BITSPACE_ML_MNIST_DOWNLOAD"] = "1"           # 自動下載 MNIST
os.environ["BITSPACE_ML_MNIST_DIR"] = "./bitSpaceData/mnist"
os.environ["BITSPACE_ML_MNIST_TRAIN_N"] = "10000"        # 訓練樣本數
os.environ["BITSPACE_ML_MNIST_TEST_N"] = "2000"          # 測試樣本數
os.environ["BITSPACE_ML_MNIST_REFINE_STEPS"] = "0"       # SBGD 精煉步數

# ============= 相機配置 =============
os.environ["BITSPACE_ML_RUN_CAMERA_DEMO"] = "0"          # 執行相機 demo
os.environ["BITSPACE_ML_CAMERA_FRAMES"] = "30"           # 擷取幀數
os.environ["BITSPACE_ML_CAMERA_SIZE"] = "128"            # 輸出尺寸
```

### 使用配置類別

```python
from ASMbitSpaceML import BitSpaceMLConfig

# 讀取配置
print(f"輸出目錄: {BitSpaceMLConfig.OUTPUT_DIR}")
print(f"JIT 啟用: {BitSpaceMLConfig.ENABLE_JIT}")
print(f"MNIST 目錄: {BitSpaceMLConfig.MNIST_DIR}")
```

---

## 3. 核心工具函數

### 3.1 位元操作

```python
from ASMbitSpaceML import (
    as_bits, bit_density, binary_entropy, 
    bit_boundary_length_2d, u8_entropy
)
import numpy as np

# ============= as_bits: 轉換為 {0,1} 位元陣列 =============
data = np.array([0, 5, -1, 0.5, 0])
bits = as_bits(data)
print(bits)  # [0 1 1 1 0] - 非零值變為 1

# ============= bit_density: 計算位元密度 (1的比例) =============
bits = np.array([1, 0, 1, 1, 0, 1])
density = bit_density(bits)
print(f"密度: {density:.3f}")  # 0.667

# ============= binary_entropy: 二元熵 =============
p = 0.5  # 50% 是 1
entropy = binary_entropy(p)
print(f"熵: {entropy:.3f}")  # 1.0 (最大熵)

p = 0.1
entropy = binary_entropy(p)
print(f"熵: {entropy:.3f}")  # 0.469

# ============= bit_boundary_length_2d: 2D邊界長度 =============
img = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 1]
], dtype=np.uint8)
boundary = bit_boundary_length_2d(img)
print(f"邊界長度: {boundary}")  # 計算相鄰不同值的數量

# ============= u8_entropy: uint8 陣列的香農熵 =============
data = np.random.randint(0, 256, 1000, dtype=np.uint8)
entropy = u8_entropy(data)
print(f"u8 熵: {entropy:.3f} bits")
```

### 3.2 位元打包/解包

```python
from ASMbitSpaceML import (
    packbits_rowwise, unpackbits_rowwise,
    packbits_1d, unpackbits_1d
)
import numpy as np

# ============= 行方向打包 (2D) =============
bits_2d = np.array([
    [1, 0, 1, 1, 0, 0, 1, 0, 1, 1],  # 10 bits
    [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
], dtype=np.uint8)

packed, row_bytes = packbits_rowwise(bits_2d, bitorder="little")
print(f"原始形狀: {bits_2d.shape}")       # (2, 10)
print(f"打包後形狀: {packed.shape}")      # (2, 2) - 每行 10 bits -> 2 bytes
print(f"每行位元組數: {row_bytes}")       # 2

# 解包還原
unpacked = unpackbits_rowwise(packed, n_bits=10, bitorder="little")
print(f"還原形狀: {unpacked.shape}")      # (2, 10)
assert np.array_equal(unpacked, bits_2d)

# ============= 1D 打包 =============
bits_1d = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)  # 9 bits
packed_1d = packbits_1d(bits_1d, bitorder="little")
print(f"1D 打包: {bits_1d.shape} -> {packed_1d.shape}")  # (9,) -> (2,)

unpacked_1d = unpackbits_1d(packed_1d, n_bits=9, bitorder="little")
assert np.array_equal(unpacked_1d, bits_1d)
```

### 3.3 XNOR-Popcount 相似度

```python
from ASMbitSpaceML import (
    xnor_popcount_bits_packed_numpy,
    hamming_distance_bits_packed_numpy
)
import numpy as np

# 建立兩個打包的位元向量
n_bits = 64
rng = np.random.RandomState(42)
a_bits = rng.randint(0, 2, n_bits, dtype=np.uint8)
b_bits = rng.randint(0, 2, n_bits, dtype=np.uint8)

# 打包
a_packed = np.packbits(a_bits, bitorder="little")
b_packed = np.packbits(b_bits, bitorder="little")

# XNOR-Popcount: 計算相同位元的數量
matches = xnor_popcount_bits_packed_numpy(a_packed, b_packed, n_bits, bitorder="little")
print(f"相同位元數: {matches} / {n_bits}")

# Hamming 距離: 不同位元的數量
hamming = hamming_distance_bits_packed_numpy(a_packed, b_packed, n_bits, bitorder="little")
print(f"Hamming 距離: {hamming}")

# 驗證: matches + hamming = n_bits
assert matches + hamming == n_bits
```

### 3.4 迭代器工具

```python
from ASMbitSpaceML import iter_minibatches
import numpy as np

n_samples = 100
batch_size = 32

# 產生隨機打亂的 mini-batch 索引
rng = np.random.RandomState(123)
for batch_idx in iter_minibatches(n_samples, batch_size, rng=rng, shuffle=True):
    print(f"Batch 大小: {len(batch_idx)}, 索引範例: {batch_idx[:3]}")
```

---

## 4. 統一媒體編解碼器 (UnifiedMediaCodec)

### 4.1 MediaType 枚舉

```python
from ASMbitSpaceML import MediaType

print(MediaType.TEXT_UTF8.value)      # "text_utf8"
print(MediaType.IMAGE_U8_GRAY.value)  # "image_u8_gray"
print(MediaType.IMAGE_U8_RGB.value)   # "image_u8_rgb"
print(MediaType.AUDIO_PCM16.value)    # "audio_pcm16"
print(MediaType.BYTES.value)          # "bytes"
```

### 4.2 UnifiedPacket 資料結構

```python
from ASMbitSpaceML import UnifiedPacket
import numpy as np

# 建立一個封包
payload = np.array([65, 66, 67], dtype=np.uint8)  # "ABC"
pkt = UnifiedPacket(
    media_type="text_utf8",
    payload_u8=payload,
    n_bits=payload.size * 8,
    bitorder="little",
    meta={"encoding": "utf-8"}
)

# 屬性存取
print(f"類型: {pkt.media_type}")
print(f"位元數: {pkt.n_bits}")
print(f"SHA256: {pkt.sha256()[:16]}...")

# 轉換為位元陣列
bits = pkt.to_bits()
print(f"位元形狀: {bits.shape}")

# 轉換為 bytes
raw = pkt.to_bytes()
print(f"原始資料: {raw}")

# 儲存/載入
pkt.save_npz("my_packet.npz")
loaded = UnifiedPacket.load_npz("my_packet.npz")
assert loaded.sha256() == pkt.sha256()
```

### 4.3 文字編解碼

```python
from ASMbitSpaceML import UnifiedMediaCodec

codec = UnifiedMediaCodec()

# 編碼文字
text = "你好, World! 🌍"
pkt = codec.encode_text(text, encoding="utf-8")
print(f"封包類型: {pkt.media_type}")
print(f"位元組數: {pkt.payload_u8.size}")
print(f"元資料: {pkt.meta}")

# 解碼文字
decoded = codec.decode_text(pkt)
print(f"解碼結果: {decoded}")
assert decoded == text
```

### 4.4 圖像編解碼

```python
from ASMbitSpaceML import UnifiedMediaCodec
import numpy as np

codec = UnifiedMediaCodec()

# ============= 灰階圖像 =============
gray_img = np.random.randint(0, 256, (64, 48), dtype=np.uint8)
pkt_gray = codec.encode_image_u8(gray_img)
print(f"灰階圖像類型: {pkt_gray.media_type}")  # image_u8_gray
print(f"形狀元資料: {pkt_gray.meta['shape']}")  # [64, 48]

decoded_gray = codec.decode_image_u8(pkt_gray)
assert np.array_equal(decoded_gray, gray_img)

# ============= RGB 圖像 =============
rgb_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
pkt_rgb = codec.encode_image_u8(rgb_img)
print(f"RGB 圖像類型: {pkt_rgb.media_type}")  # image_u8_rgb

decoded_rgb = codec.decode_image_u8(pkt_rgb)
assert np.array_equal(decoded_rgb, rgb_img)
```

### 4.5 音訊編解碼

```python
from ASMbitSpaceML import UnifiedMediaCodec
import numpy as np

codec = UnifiedMediaCodec()

# 產生 PCM16 音訊 (1秒, 16kHz)
sample_rate = 16000
duration = 1.0
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
frequency = 440  # A4 音符
pcm = (np.sin(2 * np.pi * frequency * t) * 16000).astype(np.int16)

# 編碼
pkt = codec.encode_audio_pcm16(pcm, sample_rate=sample_rate)
print(f"音訊類型: {pkt.media_type}")
print(f"元資料: {pkt.meta}")  # shape, dtype, sample_rate

# 解碼
decoded_pcm, sr = codec.decode_audio_pcm16(pkt)
print(f"採樣率: {sr}")
assert np.array_equal(decoded_pcm, pcm)
```

### 4.6 任意位元組編解碼

```python
from ASMbitSpaceML import UnifiedMediaCodec

codec = UnifiedMediaCodec()

# 編碼任意 bytes
data = b"\x00\x01\x02\xff\xfe"
pkt = codec.encode_bytes(data)

# 解碼
decoded = codec.decode_bytes(pkt)
assert decoded == data
```

---

## 5. 位元打包工具 (PackedBits)

```python
from ASMbitSpaceML import PackedBits
import numpy as np

# ============= 從位元陣列建立 =============
bits = np.array([
    [1, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1, 1, 0, 1]
], dtype=np.uint8)

packed = PackedBits.from_bits(bits)
print(f"位元數: {packed.n_bits}")          # 8
print(f"每行位元組: {packed.row_bytes}")   # 1
print(f"打包資料形狀: {packed.data.shape}")

# ============= 轉換回位元 =============
unpacked = packed.to_bits()
assert np.array_equal(unpacked, bits)

# ============= 2D 存取 =============
data_2d = packed.as_2d()
print(f"2D 形狀: {data_2d.shape}")

# ============= 直接建構 =============
raw_packed = np.array([[0b01001101], [0b10110010]], dtype=np.uint8)
pb = PackedBits(data=raw_packed, n_bits=8, bitorder="little")
```

---

## 6. 後端管理 (BitMLBackend)

```python
from ASMbitSpaceML import BitMLBackend
import numpy as np

# ============= 建立後端 =============
backend = BitMLBackend(enable_jit=True)  # 優先使用 ASMbitJIT

# 檢視後端資訊
info = backend.info()
print(f"後端: {info['backend']}")
print(f"ASMbitJIT 可用: {info['asmjit_available']}")
print(f"JIT 啟用: {info['enable_jit']}")

# ============= XNOR-Popcount =============
n_bits = 128
a = np.random.randint(0, 256, (n_bits + 7) // 8, dtype=np.uint8)
b = np.random.randint(0, 256, (n_bits + 7) // 8, dtype=np.uint8)

matches = backend.xnor_popcount_bits(a, b, n_bits)
print(f"匹配位元數: {matches}")

# ============= Hamming 距離 =============
hamming = backend.hamming_distance_bits(a, b, n_bits)
print(f"Hamming 距離: {hamming}")

# ============= 矩陣乘法 (雙極分數) =============
# A: (M, row_bytes), B: (N, row_bytes)
M, N = 16, 8
row_bytes = (n_bits + 7) // 8
A = np.random.randint(0, 256, (M, row_bytes), dtype=np.uint8)
B = np.random.randint(0, 256, (N, row_bytes), dtype=np.uint8)

# 輸出: (M, N) int32, 值 = 2*matches - n_bits
scores = backend.xnor_popcount_matmul_bipolar_i32(A, B, n_bits)
print(f"分數矩陣形狀: {scores.shape}")  # (16, 8)
print(f"分數範圍: [{scores.min()}, {scores.max()}]")  # [-n_bits, +n_bits]

# ============= 二值線性前向傳播 =============
batch_size = 32
in_features = 64
out_features = 10
in_bytes = (in_features + 7) // 8

X_packed = np.random.randint(0, 256, (batch_size, in_bytes), dtype=np.uint8)
W_packed = np.random.randint(0, 256, (out_features, in_bytes), dtype=np.uint8)
bias = np.zeros(out_features, dtype=np.int32)

output = backend.binary_linear_forward(
    X_packed=X_packed,
    W_packed=W_packed,
    batch_size=batch_size,
    out_features=out_features,
    in_features=in_features,
    bias=bias
)
print(f"輸出形狀: {output.shape}")  # (32, 10)
```

---

## 7. 神經網路層

### 7.1 BitDensePackedLayer (全連接層)

```python
from ASMbitSpaceML import BitDensePackedLayer, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立層 =============
layer = BitDensePackedLayer(
    in_features=784,
    out_features=128,
    bias=None,             # 可選偏置
    weights_bits=None,     # None = 隨機初始化
    seed=42,
    name="fc1",
    backend=backend
)

print(f"輸入特徵: {layer.in_features}")
print(f"輸出特徵: {layer.out_features}")
print(f"權重密度: {layer.weights_density():.3f}")

# ============= 前向傳播 (位元輸入) =============
batch_size = 64
X_bits = np.random.randint(0, 2, (batch_size, 784), dtype=np.uint8)

scores = layer.forward_bits(X_bits)
print(f"輸出形狀: {scores.shape}")  # (64, 128)
print(f"分數範圍: [{scores.min()}, {scores.max()}]")

# ============= 前向傳播 (打包輸入) =============
X_packed = np.packbits(X_bits, axis=1, bitorder="little")
scores_packed = layer.forward_packed(X_packed)
assert np.array_equal(scores, scores_packed)

# ============= 轉換為位元輸出 =============
output_bits = layer.forward_to_bits(X_bits, threshold=0)
print(f"輸出位元形狀: {output_bits.shape}")  # (64, 128)
print(f"輸出位元範圍: {output_bits.min()}, {output_bits.max()}")  # 0, 1

# ============= 權重存取 =============
W = layer.get_weights_bits()
print(f"權重形狀: {W.shape}")  # (128, 784)

# 修改權重
new_W = np.random.randint(0, 2, (128, 784), dtype=np.uint8)
layer.set_weights_bits(new_W)

# 單一位元存取
bit_val = layer.weight_bit_get(out_idx=0, bit_idx=100)
print(f"權重[0,100] = {bit_val}")

# 單一位元翻轉
layer.weight_bit_flip(out_idx=0, bit_idx=100)
assert layer.weight_bit_get(0, 100) != bit_val

# ============= 偏置操作 =============
layer.ensure_bias()  # 確保偏置存在 (初始化為 0)
print(f"偏置形狀: {layer.bias.shape}")

# 偏置位元翻轉 (用於 SBGD 優化)
layer.bias_bit_flip(out_idx=0, bit_pos=3)  # 翻轉第 0 個輸出的偏置第 3 位
```

### 7.2 BitConv2DPackedLayer (卷積層)

```python
from ASMbitSpaceML import BitConv2DPackedLayer, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立卷積層 =============
conv = BitConv2DPackedLayer(
    in_channels=1,
    out_channels=16,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=None,
    weights_bits=None,  # 隨機初始化
    seed=42,
    name="conv1",
    backend=backend
)

print(f"輸入通道: {conv.in_channels}")
print(f"輸出通道: {conv.out_channels}")
print(f"核大小: {conv.kernel_size}")
print(f"in_features (Cin*k*k): {conv.in_features}")

# ============= 前向傳播 =============
batch = 8
H, W = 28, 28
X_bits = np.random.randint(0, 2, (batch, 1, H, W), dtype=np.uint8)

scores = conv.forward_bits(X_bits)
print(f"輸出形狀: {scores.shape}")  # (8, 16, 28, 28) with padding=1

# 轉換為位元
output_bits = conv.forward_to_bits(X_bits, threshold=0)
print(f"位元輸出形狀: {output_bits.shape}")

# ============= 權重操作 =============
W = conv.get_weights_bits()
print(f"權重形狀: {W.shape}")  # (16, 1, 3, 3)

# 設定自定義權重 (例如邊緣檢測濾波器)
edge_filter = np.zeros((16, 1, 3, 3), dtype=np.uint8)
edge_filter[0, 0] = [[1, 1, 0], [1, 1, 0], [1, 1, 0]]  # 垂直邊緣
edge_filter[1, 0] = [[0, 1, 1], [0, 1, 1], [0, 1, 1]]  # 垂直邊緣
conv.set_weights_bits(edge_filter)

# 單一權重位元翻轉
conv.weight_bit_flip(out_ch=0, in_ch=0, ky=1, kx=1)
```

### 7.3 BitPooling2D (池化層)

```python
from ASMbitSpaceML import BitPooling2D
import numpy as np

# ============= 建立池化層 =============
# mode: "max" (OR), "min" (AND), "xor" (XOR)
pool = BitPooling2D(pool=2, stride=2, mode="max")

# ============= 前向傳播 =============
x = np.array([
    [[[1, 0, 1, 0],
      [0, 1, 0, 1],
      [1, 1, 0, 0],
      [0, 0, 1, 1]]]
], dtype=np.uint8)  # (1, 1, 4, 4)

y = pool.forward(x)
print(f"輸入形狀: {x.shape}")   # (1, 1, 4, 4)
print(f"輸出形狀: {y.shape}")   # (1, 1, 2, 2)
print(f"池化結果:\n{y[0, 0]}")

# Max pooling (OR): 2x2 區域有任何 1 則輸出 1

# ============= 不同模式 =============
pool_min = BitPooling2D(pool=2, stride=2, mode="min")  # AND: 全 1 才輸出 1
pool_xor = BitPooling2D(pool=2, stride=2, mode="xor")  # XOR: 奇偶性
```

### 7.4 BitNormLUT (正規化查找表)

```python
from ASMbitSpaceML import BitNormLUT
import numpy as np

# ============= 建立 LUT =============
norm = BitNormLUT(
    min_val=-100,
    max_val=100,
    out_bits=8  # 輸出範圍 [0, 255]
)

# ============= 前向傳播 =============
x = np.array([-100, -50, 0, 50, 100], dtype=np.int32)
y = norm.forward(x)
print(f"輸入: {x}")
print(f"輸出: {y}")  # 線性映射到 [0, 255]

# ============= 從直方圖更新 LUT =============
# 這會根據資料分布進行直方圖均衡化
data = np.random.randint(-100, 101, 10000, dtype=np.int32)
norm.update_from_histogram(data)

# 更新後的前向傳播
y_eq = norm.forward(x)
print(f"均衡化後輸出: {y_eq}")
```

### 7.5 BitAttentionTop1 (Top-1 注意力)

```python
from ASMbitSpaceML import BitAttentionTop1, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立注意力層 =============
attn = BitAttentionTop1(
    dim=64,
    residual_xor=True,  # 輸出 XOR 殘差
    backend=backend
)

# ============= 前向傳播 =============
# 輸入: (Batch, SeqLen, Dim)
B, T, D = 2, 8, 64
x_bits = np.random.randint(0, 2, (B, T, D), dtype=np.uint8)

output = attn.forward(x_bits)
print(f"輸入形狀: {x_bits.shape}")   # (2, 8, 64)
print(f"輸出形狀: {output.shape}")   # (2, 8, 64)

# 機制說明:
# 1. 對每個 token，計算與所有 token 的相似度 (XNOR-popcount)
# 2. 選擇最相似的 token (argmax)
# 3. 輸出 = selected_token XOR input (如果 residual_xor=True)
```

---

## 8. 分類模型

### 8.1 BitMLPClassifier (多層感知器分類器)

```python
from ASMbitSpaceML import BitMLPClassifier, BitMLBackend, SBGDOptimizer, BitLoss
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立模型 =============
model = BitMLPClassifier(
    input_dim=784,
    hidden_dims=[256, 128],
    num_classes=10,
    seed=42,
    backend=backend,
    name="MNIST_MLP"
)

print(model.summary())

# ============= 前向傳播 =============
batch = 32
X = np.random.randint(0, 2, (batch, 784), dtype=np.uint8)
y = np.random.randint(0, 10, batch, dtype=np.int64)

scores = model.forward_bits(X)
print(f"分數形狀: {scores.shape}")  # (32, 10)

# ============= 預測 =============
predictions = model.predict(X)
print(f"預測形狀: {predictions.shape}")  # (32,)
print(f"預測範例: {predictions[:5]}")

# ============= 計算準確率 =============
acc = model.accuracy(X, y)
print(f"準確率: {acc:.4f}")

# ============= 儲存/載入 =============
model.save_npz("mnist_mlp.npz")
loaded_model = BitMLPClassifier.load_npz("mnist_mlp.npz", backend=backend)

# ============= 訓練 (SBGD) =============
optimizer = SBGDOptimizer(sample_rate=0.01, seed=0)

for epoch in range(10):
    loss = optimizer.step(
        model=model,
        X_bits=X,
        y=y,
        loss_fn=lambda s, y: BitLoss.zero_one(s, y),
        max_flips_per_layer=256
    )
    print(f"Epoch {epoch}: Loss={loss}")
```

### 8.2 HDCPrototypeClassifier (高維計算原型分類器)

```python
from ASMbitSpaceML import HDCPrototypeClassifier, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立分類器 =============
clf = HDCPrototypeClassifier(
    input_dim=784,
    num_classes=10,
    backend=backend,
    name="MNIST_HDC"
)

# ============= 訓練 (一次性多數投票) =============
# 每個類別的原型 = 該類別所有樣本的逐位多數投票
X_train = np.random.randint(0, 2, (1000, 784), dtype=np.uint8)
y_train = np.random.randint(0, 10, 1000, dtype=np.int64)

clf.fit(X_train, y_train)

# ============= 推論 =============
X_test = np.random.randint(0, 2, (200, 784), dtype=np.uint8)
y_test = np.random.randint(0, 10, 200, dtype=np.int64)

predictions = clf.predict(X_test)
scores = clf.forward_bits(X_test)
accuracy = clf.accuracy(X_test, y_test)

print(f"準確率: {accuracy:.4f}")

# ============= 儲存/載入 =============
clf.save_npz("hdc_mnist.npz")
loaded = HDCPrototypeClassifier.load_npz("hdc_mnist.npz", backend=backend)
```

### 8.3 HNNConvClassifier (卷積分類器)

```python
from ASMbitSpaceML import (
    HNNConvClassifier, BitConv2DPackedLayer, 
    BitPooling2D, BitDensePackedLayer, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建構元件 =============
# 卷積層
conv = BitConv2DPackedLayer(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1,
    seed=42,
    backend=backend
)

# 池化層
pool = BitPooling2D(pool=2, stride=2, mode="max")

# 計算全連接層輸入維度
# 輸入: 28x28 -> 卷積後: 28x28 -> 池化後: 14x14
# 全連接輸入: 8 * 14 * 14 = 1568
head = BitDensePackedLayer(
    in_features=8 * 14 * 14,
    out_features=10,
    seed=42,
    backend=backend
)

# ============= 建立分類器 =============
classifier = HNNConvClassifier(
    image_hw=(28, 28),
    conv=conv,
    pool=pool,
    head=head,
    conv_threshold=0,
    name="ConvClassifier"
)

# ============= 前向傳播 =============
batch = 16
X = np.random.randint(0, 2, (batch, 1, 28, 28), dtype=np.uint8)
# 或 X = np.random.randint(0, 2, (batch, 28, 28), dtype=np.uint8)

scores = classifier.forward_bits(X)
print(f"分數形狀: {scores.shape}")  # (16, 10)

# ============= 預測與準確率 =============
y = np.random.randint(0, 10, batch)
predictions = classifier.predict(X)
accuracy = classifier.accuracy(X, y)

# ============= 儲存/載入 =============
classifier.save_npz("conv_classifier.npz")
loaded = HNNConvClassifier.load_npz("conv_classifier.npz", backend=backend)
```

---

## 9. HNN-Transformer 語言模型

### 9.1 ByteTokenizer

```python
from ASMbitSpaceML import ByteTokenizer

tokenizer = ByteTokenizer()

# 詞彙表大小
print(f"詞彙表大小: {tokenizer.vocab_size}")  # 256

# 編碼
text = "Hello 你好!"
tokens = tokenizer.encode(text)
print(f"Token 形狀: {tokens.shape}")
print(f"Tokens: {tokens.tolist()}")

# 解碼
decoded = tokenizer.decode(tokens)
print(f"解碼: {decoded}")
```

### 9.2 TokenEmbeddingPacked & PositionEncodingPacked

```python
from ASMbitSpaceML import TokenEmbeddingPacked, PositionEncodingPacked
import numpy as np

# ============= Token 嵌入 =============
embed = TokenEmbeddingPacked(
    vocab_size=256,
    dim=128,
    seed=42
)

# 查詢嵌入 (打包形式)
token_ids = np.array([[65, 66, 67], [68, 69, 70]], dtype=np.int64)  # (2, 3)
embeddings = embed.lookup_packed(token_ids)
print(f"嵌入形狀: {embeddings.shape}")  # (2, 3, row_bytes)

# ============= 位置編碼 =============
pos_enc = PositionEncodingPacked(
    max_len=256,
    dim=128,
    seed=1
)

# 應用位置編碼 (XOR)
x_with_pos = pos_enc.apply(embeddings)
print(f"加位置後形狀: {x_with_pos.shape}")
```

### 9.3 BitFFNPacked (前饋網路)

```python
from ASMbitSpaceML import BitFFNPacked, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立 FFN =============
ffn = BitFFNPacked(
    dim=128,
    hidden=256,
    backend=backend,
    seed=42,
    residual_xor=True  # 輸出 = FFN(x) XOR x
)

# ============= 前向傳播 =============
# 輸入: (B, T, row_bytes)
B, T = 2, 8
row_bytes = (128 + 7) // 8
x_packed = np.random.randint(0, 256, (B, T, row_bytes), dtype=np.uint8)

y_packed = ffn.forward(x_packed)
print(f"輸出形狀: {y_packed.shape}")
```

### 9.4 HNNTransformerLM (完整 Transformer)

```python
from ASMbitSpaceML import HNNTransformerLM, ByteTokenizer, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立模型 =============
lm = HNNTransformerLM(
    dim=256,           # 嵌入維度
    num_layers=2,      # Transformer 層數
    ff_hidden=512,     # FFN 隱藏層大小
    max_len=256,       # 最大序列長度
    seed=42,
    backend=backend,
    name="BitLM"
)

# ============= 前向傳播 =============
# 輸入: token IDs (B, T)
token_ids = np.array([[65, 66, 67, 68]], dtype=np.int64)
scores = lm.forward_tokens(token_ids)
print(f"分數形狀: {scores.shape}")  # (1, 4, 256) - (B, T, vocab_size)

# ============= 預測下一個 token =============
next_token = lm.predict_next(token_ids)
print(f"下一個 token: {next_token}")  # shape: (1,)

# ============= 文字生成 =============
prompt = "Hello"
generated = lm.generate_text(prompt, max_new_tokens=32)
print(f"生成文字: {repr(generated)}")

# 使用 token 陣列生成
prompt_tokens = ByteTokenizer.encode(prompt)
generated_tokens = lm.generate(
    prompt_tokens,
    max_new_tokens=32,
    stop_byte=None  # 可設定停止 byte (例如換行)
)
print(f"生成 tokens: {generated_tokens.shape}")

# ============= 儲存/載入 =============
lm.save_npz("transformer_lm.npz")
loaded_lm = HNNTransformerLM.load_npz("transformer_lm.npz", backend=backend)
```

---

## 10. HNN-GAN 圖像生成

### 10.1 BitMLPBitGenerator (生成器)

```python
from ASMbitSpaceML import BitMLPBitGenerator, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立生成器 =============
generator = BitMLPBitGenerator(
    latent_dim=128,
    hidden_dims=[256, 256],
    out_dim=784,  # 28x28
    seed=42,
    backend=backend,
    name="Generator"
)

# ============= 生成 =============
# 輸入: 隨機位元向量
z = np.random.randint(0, 2, (16, 128), dtype=np.uint8)
generated = generator.generate_bits(z)
print(f"生成形狀: {generated.shape}")  # (16, 784)

# 重塑為圖像
images = generated.reshape(16, 28, 28)
```

### 10.2 HNNGAN (完整 GAN)

```python
from ASMbitSpaceML import HNNGAN, BitMLBackend, PNGEncoder
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立 GAN =============
gan = HNNGAN(
    latent_dim=128,
    image_shape=(28, 28),
    gen_hidden=(256, 256),
    disc_hidden=(256, 128),
    seed=42,
    backend=backend,
    name="MNIST_GAN"
)

# ============= 生成圖像 =============
n_samples = 16
images = gan.generate_images_bits(n_samples, seed=123)
print(f"生成圖像形狀: {images.shape}")  # (16, 28, 28)
print(f"值範圍: {images.min()}, {images.max()}")  # 0, 1

# 儲存為 PNG
for i in range(min(4, n_samples)):
    img_u8 = (images[i] * 255).astype(np.uint8)
    PNGEncoder.encode_grayscale(img_u8, f"generated_{i}.png")

# ============= 鑑別器 =============
# 判斷圖像是真還是假
scores = gan.discriminate(images)
print(f"鑑別分數形狀: {scores.shape}")  # (16, 2) - [fake_score, real_score]

# ============= 採樣潛在向量 =============
z = gan.sample_latent(n=8, seed=0)
print(f"潛在向量形狀: {z.shape}")  # (8, 128)

# ============= 儲存/載入 =============
gan.save_npz("hnn_gan.npz")
loaded_gan = HNNGAN.load_npz("hnn_gan.npz", backend=backend)
```

---

## 11. 訓練優化器

### 11.1 SBGDOptimizer (標準 SBGD)

```python
from ASMbitSpaceML import (
    BitMLPClassifier, SBGDOptimizer, BitLoss, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)

# 建立模型與資料
model = BitMLPClassifier(784, [128], 10, seed=42, backend=backend)
X = np.random.randint(0, 2, (100, 784), dtype=np.uint8)
y = np.random.randint(0, 10, 100, dtype=np.int64)

# ============= 建立優化器 =============
optimizer = SBGDOptimizer(
    sample_rate=0.01,  # 每層隨機採樣比例
    seed=0
)

# ============= 訓練迴圈 =============
for step in range(50):
    loss = optimizer.step(
        model=model,
        X_bits=X,
        y=y,
        loss_fn=BitLoss.zero_one,      # 0-1 損失 (錯誤數)
        max_flips_per_layer=256        # 每層最大嘗試翻轉次數
    )
    if step % 10 == 0:
        acc = model.accuracy(X, y)
        print(f"Step {step}: Loss={loss}, Acc={acc:.4f}")
```

### 11.2 SBGDOneLayerIncremental (快速單層 SBGD)

```python
from ASMbitSpaceML import (
    HDCPrototypeClassifier, SBGDOneLayerIncremental, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)

# 建立並訓練原型分類器
clf = HDCPrototypeClassifier(784, 10, backend=backend)
X_train = np.random.randint(0, 2, (5000, 784), dtype=np.uint8)
y_train = np.random.randint(0, 10, 5000, dtype=np.int64)
clf.fit(X_train, y_train)

print(f"訓練後準確率: {clf.accuracy(X_train[:1000], y_train[:1000]):.4f}")

# ============= 增量精煉 =============
refiner = SBGDOneLayerIncremental(seed=0)

result = refiner.refine(
    layer=clf.layer,
    X_bits=X_train[:2000],
    y=y_train[:2000],
    steps=2000,
    max_bit_index=784  # 限制搜尋範圍 (可選)
)

print(f"精煉結果:")
print(f"  步數: {result['steps']}")
print(f"  接受數: {result['accepted']}")
print(f"  接受率: {result['accept_rate']:.3f}")
print(f"  最終損失: {result['loss_end']}")
print(f"  最佳損失: {result['best_loss']}")

# 關鍵優勢:
# - 增量更新: 不需重新計算整個前向傳播
# - 只更新受影響的分數列
# - 適合快速精煉原型分類器
```

### 11.3 BiasPulseSBGD (偏置脈衝優化)

```python
from ASMbitSpaceML import (
    BitConv2DPackedLayer, BiasPulseSBGD, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)

# 建立卷積層
conv = BitConv2DPackedLayer(
    in_channels=1, out_channels=8, kernel_size=3,
    stride=1, padding=1, seed=42, backend=backend
)

# 輸入資料
X = np.random.randint(0, 2, (1, 1, 64, 64), dtype=np.uint8)

# ============= 定義目標函數 =============
target_density = 0.1

def objective(scores, edge_bits):
    """目標: 保持邊緣密度接近目標值"""
    density = float(np.mean(edge_bits))
    return abs(density - target_density)

# ============= 優化偏置 =============
optimizer = BiasPulseSBGD(seed=0)

result = optimizer.optimize_conv_bias(
    conv=conv,
    x_bits=X,
    objective_fn=objective,
    steps=200,
    bit_positions=(0, 1, 2, 3, 4, 5, 6, 7),  # 優化的偏置位元位置
    threshold=0
)

print(f"優化結果:")
print(f"  步數: {result['steps']}")
print(f"  接受數: {result['accepted']}")
print(f"  接受率: {result['accept_rate']:.3f}")
print(f"  最佳目標值: {result['best_obj']:.4f}")
```

### 11.4 BitLoss (損失函數)

```python
from ASMbitSpaceML import BitLoss
import numpy as np

# 模擬分數輸出
scores = np.array([
    [3, -1, 2],   # 預測 class 0
    [-2, 5, 1],   # 預測 class 1
    [1, 1, 3],    # 預測 class 2
], dtype=np.int32)
y = np.array([0, 1, 0], dtype=np.int64)  # 真實標籤

# ============= 0-1 損失 =============
loss_01 = BitLoss.zero_one(scores, y)
print(f"0-1 損失 (錯誤數): {loss_01}")  # 1 (第3個錯)

# ============= Hinge Margin 損失 =============
loss_hinge = BitLoss.hinge_margin(scores, y, margin=1)
print(f"Hinge 損失 (margin=1): {loss_hinge}")
# 計算: max(0, margin + max_other_score - true_score)
```

---

## 12. 視覺化工具

### 12.1 顏色工具

```python
from ASMbitSpaceML import hsv_to_rgb_u8, palette_hsv, palette_heat256
import numpy as np

# ============= HSV 轉 RGB =============
r, g, b = hsv_to_rgb_u8(h=0.0, s=1.0, v=1.0)   # 紅色
print(f"紅色: ({r}, {g}, {b})")

r, g, b = hsv_to_rgb_u8(h=0.33, s=1.0, v=1.0)  # 綠色
print(f"綠色: ({r}, {g}, {b})")

# ============= HSV 調色盤 =============
palette = palette_hsv(n=16, s=1.0, v=1.0)
print(f"調色盤形狀: {palette.shape}")  # (16, 3)

# ============= 熱圖調色盤 (256色) =============
heat = palette_heat256()
print(f"熱圖調色盤形狀: {heat.shape}")  # (256, 3)
```

### 12.2 PNGEncoder (純 Python PNG 編碼)

```python
from ASMbitSpaceML import PNGEncoder
import numpy as np

# ============= 灰階圖像 =============
gray_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
PNGEncoder.encode_grayscale(gray_img, "gray_image.png")

# ============= RGB 圖像 =============
rgb_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
PNGEncoder.encode_rgb(rgb_img, "rgb_image.png")

# ============= 生成漸變圖 =============
height, width = 256, 256
gradient = np.zeros((height, width, 3), dtype=np.uint8)
for y in range(height):
    for x in range(width):
        gradient[y, x] = [y, x, 128]
PNGEncoder.encode_rgb(gradient, "gradient.png")
```

### 12.3 HyperSpaceProjector2D (超空間投影)

```python
from ASMbitSpaceML import HyperSpaceProjector2D, BitMLBackend
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立投影器 =============
n_bits = 256
projector = HyperSpaceProjector2D(
    n_bits=n_bits,
    seed1=1,
    seed2=2,
    backend=backend
)

# ============= 投影打包向量 =============
n_vectors = 100
row_bytes = (n_bits + 7) // 8
packed_matrix = np.random.randint(0, 256, (n_vectors, row_bytes), dtype=np.uint8)

coords = projector.project(packed_matrix)
print(f"座標形狀: {coords.shape}")  # (100, 2)
print(f"座標範圍: X=[{coords[:,0].min()}, {coords[:,0].max()}], "
      f"Y=[{coords[:,1].min()}, {coords[:,1].max()}]")

# 投影原理:
# - 使用兩個固定的參考向量 r1, r2
# - 每個向量 v 投影到 (matches(v, r1), matches(v, r2))
```

### 12.4 HyperspaceMLVisualizer (完整視覺化器)

```python
from ASMbitSpaceML import (
    HyperspaceMLVisualizer, BitDensePackedLayer, 
    BitConv2DPackedLayer, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)
viz = HyperspaceMLVisualizer(output_dir="./visualizations")

# 建立層
dense = BitDensePackedLayer(256, 64, seed=42, backend=backend)
conv = BitConv2DPackedLayer(1, 8, kernel_size=3, seed=42, backend=backend)

# ============= 權重矩陣視覺化 =============
path = viz.visualize_weights_matrix(
    layer=dense,
    name="dense_weights",
    max_width=1024,
    max_height=512,
    scale=2
)
print(f"權重矩陣圖: {path}")

# ============= 相似度矩陣視覺化 =============
path = viz.visualize_similarity_matrix(
    layer=dense,
    name="dense_similarity",
    max_neurons=64
)
print(f"相似度矩陣圖: {path}")

# ============= 嵌入空間視覺化 =============
path = viz.visualize_layer_embedding(
    layer=dense,
    name="dense_embedding",
    canvas_size=512,
    point_radius=3,
    seed1=1,
    seed2=2
)
print(f"嵌入空間圖: {path}")

# ============= 單一神經元局部結構 =============
path = viz.visualize_neuron_local(
    layer=dense,
    neuron_index=0,
    name="neuron_0_local",
    layout="morton",  # 或 "row"
    scale=4
)
print(f"神經元局部結構圖: {path}")

# ============= 決策貢獻視覺化 =============
x = np.random.randint(0, 2, 256, dtype=np.uint8)
path = viz.visualize_dense_contribution(
    layer=dense,
    x_bits=x,
    class_index=0,
    name="contribution_class0",
    image_shape=(16, 16),  # 可選, 若 in_features 可重塑
    scale=8
)
print(f"貢獻圖: {path}")
# 紅色 = 正貢獻 (匹配), 藍色 = 負貢獻 (不匹配)

# ============= 特徵圖視覺化 =============
feature_maps = np.random.randint(0, 2, (8, 16, 16), dtype=np.uint8)  # (C, H, W)
path = viz.visualize_feature_maps(
    feature_bits=feature_maps,
    name="feature_maps",
    scale=2,
    pad=2
)
print(f"特徵圖: {path}")
```

---

## 13. MNIST 整合

### 13.1 載入 MNIST

```python
from ASMbitSpaceML import (
    load_mnist, mnist_binarize, find_best_mnist_threshold, BitMLBackend
)
import os

# 確保目錄存在
mnist_dir = "./bitSpaceData/mnist"
os.makedirs(mnist_dir, exist_ok=True)

# ============= 載入資料 =============
train_images, train_labels, test_images, test_labels = load_mnist(
    mnist_dir=mnist_dir,
    download=True  # 自動下載 (如果不存在)
)

print(f"訓練集: {train_images.shape}, {train_labels.shape}")  # (60000, 28, 28), (60000,)
print(f"測試集: {test_images.shape}, {test_labels.shape}")    # (10000, 28, 28), (10000,)
print(f"像素值範圍: [{train_images.min()}, {train_images.max()}]")  # [0, 255]

# ============= 二值化 =============
threshold = 96
train_bits = mnist_binarize(train_images, threshold=threshold)
test_bits = mnist_binarize(test_images, threshold=threshold)

print(f"二值化後形狀: {train_bits.shape}")  # (60000, 784)
print(f"二值化範圍: [{train_bits.min()}, {train_bits.max()}]")  # [0, 1]

# ============= 找最佳閾值 =============
backend = BitMLBackend(enable_jit=True)
best_thr, best_acc = find_best_mnist_threshold(
    trX=train_images,
    trY=train_labels,
    teX=test_images,
    teY=test_labels,
    candidates=(32, 64, 96, 128, 160),
    train_n=5000,
    test_n=1000,
    backend=backend
)
print(f"最佳閾值: {best_thr}, 準確率: {best_acc:.4f}")
```

### 13.2 訓練與評估

```python
from ASMbitSpaceML import (
    load_mnist, mnist_binarize, HDCPrototypeClassifier,
    SBGDOneLayerIncremental, HyperspaceMLVisualizer, BitMLBackend
)
import time

backend = BitMLBackend(enable_jit=True)

# 載入資料
trX, trY, teX, teY = load_mnist(mnist_dir="./bitSpaceData/mnist")
Xtr = mnist_binarize(trX[:10000], threshold=96)
ytr = trY[:10000]
Xte = mnist_binarize(teX[:2000], threshold=96)
yte = teY[:2000]

# ============= HDC 原型分類器 =============
clf = HDCPrototypeClassifier(784, 10, backend=backend)

# 訓練
t0 = time.perf_counter()
clf.fit(Xtr, ytr)
t1 = time.perf_counter()

# 評估
acc = clf.accuracy(Xte, yte)
print(f"HDC 準確率: {acc:.4f}, 訓練時間: {t1-t0:.3f}s")

# ============= SBGD 精煉 =============
refiner = SBGDOneLayerIncremental(seed=0)
result = refiner.refine(clf.layer, Xtr[:2000], ytr[:2000], steps=2000)
print(f"精煉後準確率: {clf.accuracy(Xte, yte):.4f}")
print(f"接受率: {result['accept_rate']:.3f}")

# ============= 視覺化 =============
viz = HyperspaceMLVisualizer(output_dir="./mnist_viz")

# 權重熱圖
viz.visualize_weights_matrix(clf.layer, "mnist_weights", scale=1)

# 決策解釋
sample_idx = 0
x = Xte[sample_idx]
pred = int(clf.predict(x.reshape(1, -1))[0])
true_label = int(yte[sample_idx])
viz.visualize_dense_contribution(
    clf.layer, x, pred, 
    f"explain_pred{pred}_true{true_label}",
    image_shape=(28, 28), scale=8
)

# 儲存模型
clf.save_npz("mnist_hdc.npz")
```

---

## 14. 相機處理

### 14.1 影像工具

```python
from ASMbitSpaceML import rgb_to_gray_u8, resize_nn_u8, binarize_u8
import numpy as np

# ============= RGB 轉灰階 =============
rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
gray = rgb_to_gray_u8(rgb)
print(f"灰階形狀: {gray.shape}")  # (480, 640)

# ============= 最近鄰縮放 =============
small = resize_nn_u8(gray, new_hw=(128, 128))
print(f"縮放後形狀: {small.shape}")  # (128, 128)

# ============= 二值化 =============
binary = binarize_u8(gray, threshold=128)
print(f"二值化範圍: [{binary.min()}, {binary.max()}]")  # [0, 1]
```

### 14.2 幀源 (Frame Source)

```python
from ASMbitSpaceML import SyntheticFrameSource, OpenCVCameraSource, CV2_AVAILABLE

# ============= 合成幀源 (測試用) =============
src = SyntheticFrameSource(size=128, frames=30)
for i in range(5):
    frame = src.read()
    if frame is None:
        break
    print(f"幀 {i}: 形狀={frame.shape}")

# ============= 真實相機 (需要 cv2) =============
if CV2_AVAILABLE:
    try:
        cam = OpenCVCameraSource(index=0)
        frame = cam.read()
        if frame is not None:
            print(f"相機幀: {frame.shape}")
        cam.close()
    except Exception as e:
        print(f"相機錯誤: {e}")
```

### 14.3 CameraHNNProcessor

```python
from ASMbitSpaceML import (
    CameraHNNProcessor, SyntheticFrameSource, BitMLBackend
)

backend = BitMLBackend(enable_jit=True)

# ============= 建立處理器 =============
processor = CameraHNNProcessor(
    size=128,
    threshold_u8=128,
    conv_threshold=0,
    target_edge_density=0.08,
    backend=backend,
    output_dir="./camera_output"
)

# ============= 處理幀 =============
src = SyntheticFrameSource(size=128, frames=10)
for step in range(10):
    frame = src.read()
    if frame is None:
        break
    
    info = processor.process_frame(
        frame=frame,
        optimize_bias=True,  # 使用 SBGD 優化偏置
        step=step
    )
    
    print(f"幀 {step}:")
    print(f"  灰階圖: {info['gray_path']}")
    print(f"  二值圖: {info['bin_path']}")
    print(f"  邊緣密度: {info['edge_density']:.4f}")
    print(f"  分數均值: {info['scores_mean']:.2f}")
```

### 14.4 邊緣檢測核

```python
from ASMbitSpaceML import default_edge_kernels_3x3

kernels = default_edge_kernels_3x3()
print(f"核形狀: {kernels.shape}")  # (8, 1, 3, 3)

# 視覺化各核
for i in range(8):
    print(f"核 {i}:")
    print(kernels[i, 0])
    print()

# 核說明:
# 0, 1: 垂直邊緣 (左/右)
# 2, 3: 水平邊緣 (上/下)  
# 4, 5: 對角邊緣
# 6: 點
# 7: 十字
```

---

## 15. 音樂生成

### 15.1 BitMusicGenerator

```python
from ASMbitSpaceML import (
    BitMusicGenerator, wav_write_pcm16, wav_bytes_from_pcm16, UnifiedMediaCodec
)

# ============= 建立生成器 =============
music = BitMusicGenerator(seed=12345)

# ============= 生成 PCM16 波形 =============
pcm = music.generate_pcm16(
    seconds=3.0,
    sample_rate=16000,
    amp=8000
)
print(f"PCM 形狀: {pcm.shape}")  # (48000,)
print(f"PCM 範圍: [{pcm.min()}, {pcm.max()}]")

# ============= 儲存為 WAV =============
wav_write_pcm16("bitspace_music.wav", pcm, sample_rate=16000)

# ============= 取得 WAV bytes =============
wav_data = wav_bytes_from_pcm16(pcm, sample_rate=16000)
print(f"WAV 大小: {len(wav_data)} bytes")

# ============= 使用 UnifiedPacket =============
pkt = music.generate_wav_packet(seconds=2.0, sample_rate=8000)
print(f"封包類型: {pkt.media_type}")
print(f"封包元資料: {pkt.meta}")

# 解碼並使用
decoded_pcm, sr = UnifiedMediaCodec.decode_audio_pcm16(pkt)
wav_write_pcm16("music_from_packet.wav", decoded_pcm, sr)
```

---

## 16. 應用層整合 (BitSpaceApplication)

```python
from ASMbitSpaceML import (
    BitSpaceApplication, UnifiedMediaCodec, 
    PNGEncoder, wav_write_pcm16, BitMLBackend
)
import numpy as np

backend = BitMLBackend(enable_jit=True)

# ============= 建立應用 =============
app = BitSpaceApplication(
    backend=backend,
    output_dir="./app_output"
)

# ============= 描述封包 =============
codec = UnifiedMediaCodec()

# 文字封包
text_pkt = codec.encode_text("Hello BitSpace!")
desc = app.describe(text_pkt)
print(f"文字描述: {desc}")

# 圖像封包
img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
img_pkt = codec.encode_image_u8(img)
desc = app.describe(img_pkt)
print(f"圖像描述: {desc}")

# 音訊封包
pcm = np.random.randint(-1000, 1000, 8000, dtype=np.int16)
audio_pkt = codec.encode_audio_pcm16(pcm, sample_rate=8000)
desc = app.describe(audio_pkt)
print(f"音訊描述: {desc}")

# ============= 文字生成 =============
generated_text = app.generate_text("The meaning of life is", max_new_tokens=50)
print(f"生成文字: {repr(generated_text[:100])}")

# ============= 圖像生成 (GAN) =============
images = app.generate_image_bits(n=4, seed=42)
print(f"生成圖像形狀: {images.shape}")  # (4, 28, 28)

for i in range(4):
    PNGEncoder.encode_grayscale(
        (images[i] * 255).astype(np.uint8),
        f"./app_output/gan_image_{i}.png"
    )

# ============= 音樂生成 =============
music_pkt = app.generate_music_packet(seconds=2.0, sample_rate=16000)
decoded_pcm, sr = UnifiedMediaCodec.decode_audio_pcm16(music_pkt)
wav_write_pcm16("./app_output/generated_music.wav", decoded_pcm, sr)

# ============= MNIST 分類 (需先訓練) =============
# 若已載入 MNIST 並訓練分類器
# app.mnist_classifier = clf  # 設定分類器
# 
# test_img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
# pred = app.classify_mnist_image(test_img, threshold=96)
# print(f"MNIST 預測: {pred}")
```

---

## 17. 測試與基準測試

### 17.1 執行所有測試

```python
from ASMbitSpaceML import testAll

# 執行完整測試套件
# 包含:
# - 編解碼器測試
# - 後端測試
# - 層正確性測試
# - 模型測試
# - 視覺化測試
# - MNIST 整合測試
# - 效能基準測試

success = testAll()
print(f"測試{'通過' if success else '失敗'}")
```

### 17.2 獨立效能基準

```python
from ASMbitSpaceML import BitSpaceMLPerformanceBenchmark

# 執行所有基準測試
BitSpaceMLPerformanceBenchmark.run_all()

# 或執行單一基準
BitSpaceMLPerformanceBenchmark.bench_dense_forward()
BitSpaceMLPerformanceBenchmark.bench_conv_forward()
BitSpaceMLPerformanceBenchmark.bench_transformer_forward()
BitSpaceMLPerformanceBenchmark.bench_png_encode()
```

### 17.3 自定義測試

```python
from ASMbitSpaceML import TestResult, assert_true, assert_equal, assert_raises
import numpy as np

def my_custom_test():
    tr = TestResult()
    
    # 測試相等
    assert_equal(tr, "array_equal", np.array([1,2,3]), np.array([1,2,3]))
    
    # 測試條件
    assert_true(tr, "positive", 5 > 0)
    
    # 測試例外
    def div_by_zero():
        return 1 / 0
    assert_raises(tr, "div_zero", ZeroDivisionError, div_by_zero)
    
    # 顯示摘要
    return tr.summary("my_custom_test")

my_custom_test()
```

---

## 附錄: 完整使用範例

### A. MNIST 完整工作流程

```python
from ASMbitSpaceML import (
    load_mnist, mnist_binarize, find_best_mnist_threshold,
    HDCPrototypeClassifier, SBGDOneLayerIncremental,
    HyperspaceMLVisualizer, BitMLBackend, PNGEncoder
)
import os
import time

# 設定
os.makedirs("./mnist_demo", exist_ok=True)
backend = BitMLBackend(enable_jit=True)

# 1. 載入資料
print("載入 MNIST...")
trX, trY, teX, teY = load_mnist(mnist_dir="./bitSpaceData/mnist")

# 2. 找最佳閾值
print("尋找最佳二值化閾值...")
best_thr, _ = find_best_mnist_threshold(trX, trY, teX, teY, backend=backend)
print(f"最佳閾值: {best_thr}")

# 3. 二值化
Xtr = mnist_binarize(trX[:10000], threshold=best_thr)
ytr = trY[:10000]
Xte = mnist_binarize(teX[:2000], threshold=best_thr)
yte = teY[:2000]

# 4. 訓練 HDC 分類器
print("訓練 HDC 分類器...")
clf = HDCPrototypeClassifier(784, 10, backend=backend)
t0 = time.perf_counter()
clf.fit(Xtr, ytr)
t1 = time.perf_counter()
acc1 = clf.accuracy(Xte, yte)
print(f"初始準確率: {acc1:.4f}, 訓練時間: {t1-t0:.3f}s")

# 5. SBGD 精煉
print("SBGD 精煉...")
refiner = SBGDOneLayerIncremental(seed=0)
result = refiner.refine(clf.layer, Xtr[:2000], ytr[:2000], steps=3000)
acc2 = clf.accuracy(Xte, yte)
print(f"精煉後準確率: {acc2:.4f}")
print(f"接受率: {result['accept_rate']:.3f}")

# 6. 視覺化
print("生成視覺化...")
viz = HyperspaceMLVisualizer(output_dir="./mnist_demo")
viz.visualize_weights_matrix(clf.layer, "weights", scale=1)
viz.visualize_similarity_matrix(clf.layer, "similarity", max_neurons=10)

# 視覺化一些預測
for i in range(5):
    x = Xte[i]
    pred = int(clf.predict(x.reshape(1, -1))[0])
    true_label = int(yte[i])
    viz.visualize_dense_contribution(
        clf.layer, x, pred,
        f"sample{i}_pred{pred}_true{true_label}",
        image_shape=(28, 28), scale=8
    )
    PNGEncoder.encode_grayscale(teX[i], f"./mnist_demo/sample{i}_input.png")

# 7. 儲存模型
clf.save_npz("./mnist_demo/mnist_classifier.npz")
print("完成! 輸出在 ./mnist_demo/")
```

### B. 文字生成與分析

```python
from ASMbitSpaceML import (
    HNNTransformerLM, ByteTokenizer, 
    UnifiedMediaCodec, BitMLBackend, u8_entropy
)

backend = BitMLBackend(enable_jit=True)
codec = UnifiedMediaCodec()

# 建立 LM
lm = HNNTransformerLM(dim=256, num_layers=2, ff_hidden=512, max_len=128, seed=42, backend=backend)

# 多個提示詞生成
prompts = [
    "The weather today is",
    "Machine learning is",
    "In the year 2050,",
]

for prompt in prompts:
    # 生成文字
    generated = lm.generate_text(prompt, max_new_tokens=50)
    
    # 分析
    pkt = codec.encode_text(generated)
    bits = pkt.to_bits()
    ent = u8_entropy(pkt.payload_u8)
    
    print(f"提示詞: {prompt}")
    print(f"生成: {repr(generated[:80])}")
    print(f"熵: {ent:.3f} bits, 長度: {len(pkt.payload_u8)} bytes")
    print()
```

### C. 相機邊緣檢測 (完整)

```python
from ASMbitSpaceML import (
    CameraHNNProcessor, OpenCVCameraSource, SyntheticFrameSource,
    CV2_AVAILABLE, BitMLBackend, HyperspaceMLVisualizer
)
import os

os.makedirs("./camera_demo", exist_ok=True)
backend = BitMLBackend(enable_jit=True)

# 建立處理器
processor = CameraHNNProcessor(
    size=128,
    threshold_u8=128,
    conv_threshold=0,
    target_edge_density=0.1,
    backend=backend,
    output_dir="./camera_demo"
)

# 選擇幀源
if CV2_AVAILABLE:
    try:
        src = OpenCVCameraSource(0)
        print("使用真實相機")
    except:
        src = SyntheticFrameSource(size=128, frames=30)
        print("相機不可用, 使用合成幀")
else:
    src = SyntheticFrameSource(size=128, frames=30)
    print("cv2 不可用, 使用合成幀")

# 處理
for i in range(30):
    frame = src.read()
    if frame is None:
        break
    
    info = processor.process_frame(frame, optimize_bias=True, step=i)
    print(f"幀 {i:03d}: 邊緣密度={info['edge_density']:.4f}")

src.close()

# 視覺化卷積核
viz = HyperspaceMLVisualizer(output_dir="./camera_demo")
viz.visualize_weights_matrix(processor.conv, "conv_weights", scale=8)

print("完成! 輸出在 ./camera_demo/")
```

---

這份文件涵蓋了 `ASMbitSpaceML.py` 的所有主要功能，包括：

1. **環境配置** - 環境變數與配置類別
2. **工具函數** - 位元操作、打包/解包、相似度計算
3. **統一媒體編解碼** - 文字、圖像、音訊的可逆編碼
4. **神經網路層** - Dense、Conv2D、Pooling、Attention
5. **分類模型** - MLP、HDC原型、卷積分類器
6. **生成模型** - Transformer LM、GAN、音樂生成
7. **訓練優化** - SBGD 及其變體
8. **視覺化** - 權重、相似度、嵌入、決策解釋
9. **整合** - MNIST、相機處理、完整應用層