
# ASMbitBCMD.py 完整使用說明文檔

## 目錄

1. [概述](#1-概述)
2. [系統需求與依賴](#2-系統需求與依賴)
3. [環境變數配置](#3-環境變數配置)
4. [核心資料結構](#4-核心資料結構)
5. [底層 API](#5-底層-api)
6. [高階易用介面](#6-高階易用介面)
7. [串流 API](#7-串流-api)
8. [命令列介面 (CLI)](#8-命令列介面-cli)
9. [視覺化與追蹤](#9-視覺化與追蹤)
10. [測試資料生成器](#10-測試資料生成器)
11. [完整應用範例](#11-完整應用範例)

---

## 1. 概述

**ASMbitBCMD.py** 是一個基於 **BCMD (Bitwise Complementary Mask Decomposition)** 演算法的無損壓縮庫，使用 JIT 編譯的 C++ 核心進行高效位元運算。

### 核心特點

- **純位元運算遮罩分解**：使用三種遮罩族群 (F0/F1/F2) 進行資料分割
- **JIT 編譯加速**：透過 `ASMbitJIT.py` 即時編譯 C++ 計算核心
- **自適應學習**：Policy Net 和 Patterns Cache 自動學習最佳壓縮策略
- **多容器格式**：支援檔案容器 (ASMBCMD2) 和串流容器 (ASMBCMDS)
- **資料類型特化**：針對文字、圖片、影音等不同類型優化
- **算力等級控制**：Low/Medium/High 三種等級平衡速度與壓縮率

### 遮罩族群說明

| 族群 | 公式 | 說明 |
|------|------|------|
| F0_SHIFT_BIT | `m(i) = (i >> p) & 1` | 位移位元遮罩 |
| F1_XOR_SHIFT | `m(i) = ((i ^ c) >> p) & 1` | 異或位移遮罩 |
| F2_PARITY_AND | `m(i) = parity(i & c)` | 奇偶校驗遮罩 |

---

## 2. 系統需求與依賴

### 必要依賴

```python
# 必要
import numpy as np          # 數值運算
import ASMbitJIT            # JIT 編譯核心 (必須存在於同目錄或 PYTHONPATH)
```

### 可選依賴

```python
# 選用：影像 I/O
import cv2                  # OpenCV - MP4 輸出、JPG 讀取
from PIL import Image       # Pillow - JPG 讀取 (cv2 替代方案)
```

### 安裝方式

```bash
# 必要
pip install numpy

# 可選
pip install opencv-python   # 或 opencv-python-headless
pip install Pillow
```

---

## 3. 環境變數配置

### 基本設定

```bash
# 詳細日誌輸出
export ASMBCMD_VERBOSE=1

# 重度效能測試 (更大資料量)
export ASMBCMD_HEAVY_BENCH=1

# Policy 檔案路徑
export ASMBCMD_POLICY_PATH=./ASMbitBCMD_policy.json

# 輸出目錄
export ASMBCMD_OUTPUT_DIR=./asmbcmd_out

# 禁用影像 I/O
export ASMBCMD_DISABLE_IMAGE_IO=1

# C++ 優化等級 (0-3)
export ASMBCMD_OPT=3

# 是否強制要求 ASMbitJIT (預設 1)
export ASMBCMD_REQUIRE_ASMBITJIT=1
```

### Patterns Cache 設定

```bash
# 啟用 patterns 快取
export ASMBCMD_USE_PATTERNS=1

# Patterns 檔案路徑
export ASMBCMD_PATTERNS_PATH=./BCMDpatterns.bzipz

# 每種類型保留的 pattern 數量
export ASMBCMD_PATTERNS_PER_TYPE=256

# 最大類型群組數
export ASMBCMD_PATTERNS_MAX_GROUPS=512
```

### 易用介面設定

```bash
# 介面層預設目錄
export ASMBCMD_EASY_DIR=./asmbcmd_easy

# 預設算力等級
export ASMBCMD_DEFAULT_TIER=medium  # low|medium|high
```

---

## 4. 核心資料結構

### 4.1 PackedBitStream - 打包位元流

```python
from ASMbitBCMD import PackedBitStream
import numpy as np

# 建立位元流
data = np.array([0xAA, 0x55, 0xFF, 0x00], dtype=np.uint8)
stream = PackedBitStream(data_u8=data, n_bits=32)

# 屬性
print(f"位元數: {stream.n_bits}")      # 32
print(f"位元組數: {stream.n_bytes}")    # 4

# 轉換為 bytes
raw_bytes = stream.to_bytes_exact()
print(f"原始資料: {raw_bytes.hex()}")   # aa55ff00
```

### 4.2 MaskSpec - 遮罩規格

```python
from ASMbitBCMD import MaskSpec, MaskFamily

# F0: 位移位元遮罩
mask_f0 = MaskSpec(MaskFamily.F0_SHIFT_BIT, p=3, c=0)
print(mask_f0.to_str())  # "F0:p=3"

# F1: 異或位移遮罩
mask_f1 = MaskSpec(MaskFamily.F1_XOR_SHIFT, p=2, c=0x0F)
print(mask_f1.to_str())  # "F1:p=2,c=15"

# F2: 奇偶校驗遮罩
mask_f2 = MaskSpec(MaskFamily.F2_PARITY_AND, p=0, c=0x55)
print(mask_f2.to_str())  # "F2:c=85"

# 從字串解析
mask = MaskSpec.from_str("F1:p=2,c=15")
print(f"Family: {mask.family}, p={mask.p}, c={mask.c}")

# 檢查是否使用 c 參數
print(f"F0 uses c: {mask_f0.uses_c()}")  # False
print(f"F1 uses c: {mask_f1.uses_c()}")  # True
```

### 4.3 MarkovCounts - 馬可夫轉移統計

```python
from ASMbitBCMD import BCMDComputeCore, PackedBitStream
import numpy as np

core = BCMDComputeCore()
data = np.array([0b10101010, 0b01010101], dtype=np.uint8)
stream = PackedBitStream(data, n_bits=16)

counts = core.markov_counts(stream)
print(f"1 的數量: {counts.ones}")
print(f"0→0 轉移: {counts.c00}")
print(f"0→1 轉移: {counts.c01}")
print(f"1→0 轉移: {counts.c10}")
print(f"1→1 轉移: {counts.c11}")
print(f"邊界次數: {counts.boundary}")  # c01 + c10
```

### 4.4 LeafPayload - 葉節點負載

```python
from ASMbitBCMD import compress_leaf, decompress_leaf, LeafCoderId

# 原始資料
raw_data = b"Hello World! " * 100

# 壓縮 (自動選擇最佳編碼器)
payload = compress_leaf(raw_data, coder=LeafCoderId.AUTO)
print(f"編碼器: {payload.coder.name}")
print(f"壓縮後大小: {payload.nbytes()}")
print(f"CRC32: {payload.payload_crc32:#x}")

# 解壓縮
decompressed = decompress_leaf(payload)
assert decompressed == raw_data

# 指定編碼器
payload_bz2 = compress_leaf(raw_data, coder=LeafCoderId.BZ2, bz2_level=9)
payload_zlib = compress_leaf(raw_data, coder=LeafCoderId.ZLIB, zlib_level=6)
payload_raw = compress_leaf(raw_data, coder=LeafCoderId.RAW)
```

---

## 5. 底層 API

### 5.1 BCMDComputeCore - JIT 計算核心

```python
from ASMbitBCMD import BCMDComputeCore, PackedBitStream, MaskSpec, MaskFamily
import numpy as np

# 初始化核心 (自動 JIT 編譯 C++)
core = BCMDComputeCore(opt_level="3")

# 準備測試資料
data = np.random.randint(0, 256, size=1000, dtype=np.uint8)
stream = PackedBitStream(data, n_bits=8000)

# === markov_counts: 計算馬可夫轉移統計 ===
mc = core.markov_counts(stream)
print(f"統計: ones={mc.ones}, boundary={mc.boundary}")

# === eval_mask: 評估遮罩效果 ===
spec = MaskSpec(MaskFamily.F0_SHIFT_BIT, p=3)
ev = core.eval_mask(stream, i0_bits=0, spec=spec)
print(f"分割後: n1={ev.n1} bits, switches={ev.switches}")
print(f"A 子流: c00={ev.a_c00}, c01={ev.a_c01}, c10={ev.a_c10}, c11={ev.a_c11}")
print(f"B 子流: c00={ev.b_c00}, c01={ev.b_c01}, c10={ev.b_c10}, c11={ev.b_c11}")

# === partition: 依遮罩分割位元流 ===
A, B = core.partition(stream, spec=spec, n1_bits=ev.n1)
print(f"A: {A.n_bits} bits, B: {B.n_bits} bits")
assert A.n_bits + B.n_bits == stream.n_bits

# === interleave: 合併位元流 (partition 的逆運算) ===
reconstructed = core.interleave(A, B, spec=spec, n_bits=stream.n_bits)

# 驗證無損
orig_bits = np.unpackbits(stream.data_u8[:stream.n_bytes], bitorder="little")[:stream.n_bits]
rec_bits = np.unpackbits(reconstructed.data_u8[:reconstructed.n_bytes], bitorder="little")[:stream.n_bits]
assert np.array_equal(orig_bits, rec_bits), "Lossless verification failed!"
```

### 5.2 BCMDScorer - 成本評估器

```python
from ASMbitBCMD import BCMDScorer, BCMDScoringConfig, MaskSpec, MaskFamily

# 自訂評分配置
config = BCMDScoringConfig(
    lambda_switch=0.20,      # 切換成本權重
    switch_cost_div=8,       # 切換成本除數
    ratio_min=0.05,          # 最小分割比例
    gain_min_bits=8.0,       # 最小增益位元數
    leaf_header_base_bits=48,
    split_header_base_bits=16
)
scorer = BCMDScorer(config)

# 計算葉節點成本
n_bits = 10000
leaf_cost = scorer.leaf_cost_from_counts(n_bits, c00=2000, c01=500, c10=500, c11=2000)
print(f"葉節點成本: {leaf_cost:.2f} bits")

# 計算分割標頭成本
spec = MaskSpec(MaskFamily.F1_XOR_SHIFT, p=3, c=0xFF)
header_cost = scorer.split_header_cost(spec, n1=5000)
print(f"分割標頭成本: {header_cost:.2f} bits")

# 計算切換成本
switches_cost = scorer.switches_cost(switches=100)
print(f"切換成本: {switches_cost:.2f} bits")
```

### 5.3 BCMDSearchEngine - 搜尋引擎

```python
from ASMbitBCMD import (
    BCMDSearchEngine, BCMDSearchConfig, BCMDScorer, BCMDScoringConfig,
    BCMDComputeCore, PackedBitStream, AuxMaskPolicyNet
)
import numpy as np
import random

# 配置搜尋引擎
search_cfg = BCMDSearchConfig(
    max_p=12,                    # 最大位移參數
    sample_bits=16384,           # 取樣位元數
    sample_candidates=256,       # 取樣候選數
    full_candidates=32,          # 完整評估候選數
    sample_windows=2,            # 取樣視窗數
    mutate_candidates=96,        # 變異候選數
    mutate_elites=8,             # 精英數量
    random_candidates=64,        # 隨機候選數
    policy_candidates=32,        # Policy 候選數
    pattern_candidates=64,       # Pattern 候選數
    full_exploration_frac=0.25,  # 完整評估探索比例
    random_like_per_bit_threshold=0.995,  # 隨機資料閾值
)

core = BCMDComputeCore()
scorer = BCMDScorer(BCMDScoringConfig())
engine = BCMDSearchEngine(search_cfg, scorer, core, rng=random.Random(42))

# 準備資料
data = np.zeros(10000, dtype=np.uint8)
data[::2] = 0xFF  # 交錯模式
stream = PackedBitStream(data, n_bits=80000)

# 搜尋最佳遮罩
policy = AuxMaskPolicyNet("./policy.json")
policy.load()

best_mask, best_gain, detail = engine.choose_best_mask(
    stream,
    policy=policy,
    patterns=None,
    type_key="test"
)

if best_mask:
    print(f"最佳遮罩: {best_mask.to_str()}")
    print(f"預期增益: {best_gain:.2f} bits")
    print(f"評估詳情:")
    print(f"  - 基礎成本: {detail['base_cost']:.2f} bits")
    print(f"  - 分割成本: {detail['split_cost']:.2f} bits")
    print(f"  - n1 位元數: {detail['n1']}")
    print(f"  - 切換次數: {detail['switches']}")
else:
    print(f"未找到有效分割, 原因: {detail['reason']}")
```

### 5.4 BCMDCodec - 完整編解碼器

```python
from ASMbitBCMD import BCMDCodec, BCMDCodecConfig, BCMDTrace, LeafCoderId
import os

# 詳細配置
config = BCMDCodecConfig(
    # 區塊設定
    block_bytes=256 * 1024,    # 區塊大小 (256KB)
    max_depth=3,               # 最大遞迴深度
    min_bits=2048,             # 最小位元數閾值
    
    # 葉節點編碼
    leaf_coder=LeafCoderId.AUTO,  # 自動選擇編碼器
    bz2_level=9,               # BZ2 壓縮等級
    zlib_level=9,              # ZLIB 壓縮等級
    
    # 學習設定
    auto_train=True,           # 自動訓練 Policy
    policy_path="./policy.json",
    
    # Patterns Cache
    use_patterns_cache=True,
    auto_patterns=True,
    patterns_path="./patterns.bzipz",
    patterns_per_type=256,
    patterns_max_groups=512,
    
    # 其他
    random_seed=0,
    opt_level="3"
)
config.validate()  # 驗證配置

codec = BCMDCodec(config)

# === 記憶體內壓縮/解壓 ===
original_data = b"Hello World! " * 10000
meta = {"type": "text", "subtype": "utf8", "author": "demo"}
trace = BCMDTrace(max_events=1000)

compressed = codec.encode_bytes(original_data, meta=meta, trace=trace)
print(f"原始: {len(original_data):,} bytes")
print(f"壓縮: {len(compressed):,} bytes")
print(f"比例: {len(compressed)/len(original_data):.4f}")

decompressed, meta_out = codec.decode_bytes(compressed)
assert decompressed == original_data
print(f"元資料: {meta_out}")

# === 檔案壓縮/解壓 (傳統方式) ===
with open("input.bin", "wb") as f:
    f.write(original_data)

codec.encode_file("input.bin", "output.bzip", meta=meta, trace=trace)
meta_file = codec.decode_file("output.bzip", "decoded.bin")

with open("decoded.bin", "rb") as f:
    assert f.read() == original_data

# === 低記憶體串流壓縮/解壓 ===
info_enc = codec.encode_file_streaming(
    "input.bin", 
    "output_stream.bzip",
    meta={"mode": "streaming"},
    trace=None
)
print(f"串流壓縮結果: {info_enc}")

info_dec = codec.decode_file_streaming("output_stream.bzip", "decoded_stream.bin")
print(f"串流解壓結果: {info_dec}")
```

---

## 6. 高階易用介面

### 6.1 資料類型與算力等級

```python
from ASMbitBCMD import DataKind, ComputeTier

# 資料類型枚舉
print(DataKind.AUTO)     # 自動偵測
print(DataKind.GENERIC)  # 通用二進位
print(DataKind.TEXT)     # 文字
print(DataKind.IMAGE)    # 圖片
print(DataKind.VIDEO)    # 影片
print(DataKind.AUDIO)    # 音訊

# 算力等級枚舉
print(ComputeTier.AUTO)    # 自動選擇
print(ComputeTier.LOW)     # 低算力 (快速)
print(ComputeTier.MEDIUM)  # 中等算力 (平衡)
print(ComputeTier.HIGH)    # 高算力 (高壓縮率)
```

### 6.2 自動類型偵測

```python
from ASMbitBCMD import guess_data_kind_from_path, guess_subtype_from_path

# 根據副檔名猜測資料類型
print(guess_data_kind_from_path("document.txt"))   # DataKind.TEXT
print(guess_data_kind_from_path("photo.jpg"))      # DataKind.IMAGE
print(guess_data_kind_from_path("movie.mp4"))      # DataKind.VIDEO
print(guess_data_kind_from_path("music.wav"))      # DataKind.AUDIO
print(guess_data_kind_from_path("data.bin"))       # DataKind.GENERIC

# 取得子類型 (副檔名)
print(guess_subtype_from_path("photo.jpg"))        # "jpg"
print(guess_subtype_from_path("archive.tar.gz"))   # "gz"
```

### 6.3 一鍵壓縮/解壓縮

```python
from ASMbitBCMD import compress_file_easy, decompress_file_easy, info_file_easy

# === 一鍵壓縮 ===
result = compress_file_easy(
    "large_document.txt",
    "compressed.bzip",          # 可選，預設為 input + ".bzip"
    kind="auto",                # auto|text|image|video|audio|generic
    tier="medium",              # auto|low|medium|high
    realtime=False,             # 即時模式 (低延遲)
    container="auto",           # auto|file|stream
    meta_extra={"project": "demo"},
    enable_training=True,       # 啟用 Policy 學習
    enable_patterns=True,       # 啟用 Patterns Cache
    learning_dir="./models",    # 學習檔案目錄
    trace_out_prefix="./traces/doc",  # 追蹤輸出前綴
    trace_max_events=10000
)

print(f"輸入: {result['in_path']}")
print(f"輸出: {result['out_path']}")
print(f"原始: {result['raw_bytes']:,} bytes")
print(f"壓縮: {result['compressed_bytes']:,} bytes")
print(f"比例: {result['ratio']:.4f}")
print(f"耗時: {result['seconds']:.2f}s")

# === 一鍵解壓縮 (自動偵測容器格式) ===
result = decompress_file_easy(
    "compressed.bzip",
    "restored.txt"              # 可選，預設為 input + ".out"
)
print(f"解壓結果: {result}")

# === 查看檔案資訊 ===
info = info_file_easy("compressed.bzip")
print(f"容器類型: {info['container']}")
print(f"版本: {info.get('version')}")
print(f"區塊大小: {info.get('block_bytes')}")
print(f"原始大小: {info.get('raw_bytes')}")
print(f"區塊數: {info.get('n_blocks')}")
print(f"元資料: {info.get('meta')}")
```

### 6.4 建立特化配置

```python
from ASMbitBCMD import build_codec_config, BCMDCodec, DataKind, ComputeTier

# 建立針對文字、高算力的配置
config = build_codec_config(
    kind=DataKind.TEXT,
    tier=ComputeTier.HIGH,
    subtype="json",
    realtime=False,
    learning_dir="./my_models",
    enable_training=True,
    enable_patterns=True,
    opt_level="3"
)

print(f"區塊大小: {config.block_bytes}")
print(f"最大深度: {config.max_depth}")
print(f"最小位元: {config.min_bits}")
print(f"葉編碼器: {config.leaf_coder.name}")
print(f"Policy 路徑: {config.policy_path}")
print(f"Patterns 路徑: {config.patterns_path}")

# 使用配置建立編解碼器
codec = BCMDCodec(config)
```

### 6.5 學習路徑管理

```python
from ASMbitBCMD import build_learning_paths, ComputeTier

# 為不同類型/等級建立獨立的學習檔案路徑
policy_path, patterns_path = build_learning_paths(
    base_dir="./learning_data",
    type_key="image:png",
    tier=ComputeTier.HIGH
)

print(f"Policy: {policy_path}")
# 輸出: ./learning_data/ASMbitBCMD_policy_image_png_high.json

print(f"Patterns: {patterns_path}")
# 輸出: ./learning_data/BCMDpatterns_image_png_high.bzipz
```

---

## 7. 串流 API

### 7.1 BCMDStreamWriter - 串流寫入器

```python
from ASMbitBCMD import BCMDStreamWriter, BCMDCodec, BCMDCodecConfig

# 建立編解碼器
codec = BCMDCodec(BCMDCodecConfig(
    block_bytes=64 * 1024,  # 串流模式建議較小區塊
    auto_train=True,
    use_patterns_cache=True
))

# === 方法一：使用 context manager ===
with BCMDStreamWriter.open(
    "stream_output.bzs",
    codec=codec,
    meta={"type": "realtime_data"},
    autosave_interval_blocks=256,   # 每 256 區塊自動儲存學習資料
    autosave_interval_sec=60.0      # 或每 60 秒
) as writer:
    # 逐塊寫入資料
    for chunk in data_generator():
        writer.write(chunk)
    
    # 強制刷新緩衝區
    writer.flush()

# === 方法二：手動管理 ===
import io

output_buffer = io.BytesIO()
writer = BCMDStreamWriter(
    output_buffer,
    codec=codec,
    meta={"source": "sensor"}
)

try:
    writer.write(b"sensor data chunk 1...")
    writer.write(b"sensor data chunk 2...")
    writer.flush()
finally:
    writer.close()

compressed_data = output_buffer.getvalue()
```

### 7.2 BCMDStreamReader - 串流讀取器

```python
from ASMbitBCMD import BCMDStreamReader

# === 方法一：使用 context manager ===
with BCMDStreamReader.open("stream_output.bzs") as reader:
    # 讀取標頭資訊
    print(f"版本: {reader.header.version}")
    print(f"區塊大小: {reader.header.block_bytes}")
    print(f"元資料: {reader.header.meta}")
    
    # 逐塊解壓
    for block in reader.iter_blocks():
        process(block)

# === 方法二：一次讀取全部 ===
with BCMDStreamReader.open("stream_output.bzs") as reader:
    all_data = reader.readall()
    print(f"總大小: {len(all_data)} bytes")

# === 方法三：從記憶體緩衝區讀取 ===
import io

buffer = io.BytesIO(compressed_data)
reader = BCMDStreamReader(buffer)
try:
    for block in reader.iter_blocks():
        print(f"Block size: {len(block)}")
finally:
    reader.close()
```

### 7.3 即時管線範例

```python
from ASMbitBCMD import BCMDStreamWriter, BCMDStreamReader, BCMDCodec, BCMDCodecConfig
import socket
import threading

# 編解碼器配置 (針對即時通訊優化)
config = BCMDCodecConfig(
    block_bytes=32 * 1024,   # 小區塊降低延遲
    max_depth=2,             # 減少遞迴降低延遲
    auto_train=False,        # 即時模式可關閉學習
    use_patterns_cache=False
)
codec = BCMDCodec(config)

# === 發送端 ===
def sender(sock):
    class SocketWriter:
        def __init__(self, sock):
            self.sock = sock
        def write(self, data):
            self.sock.sendall(data)
        def flush(self):
            pass
        def tell(self):
            return -1
    
    writer = BCMDStreamWriter(
        SocketWriter(sock),
        codec=codec,
        meta={"stream": "realtime"}
    )
    
    try:
        for data in generate_realtime_data():
            writer.write(data)
            writer.flush()  # 確保即時發送
    finally:
        writer.close()

# === 接收端 ===
def receiver(sock):
    class SocketReader:
        def __init__(self, sock):
            self.sock = sock
            self.buffer = b""
        
        def read(self, n):
            while len(self.buffer) < n:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise EOFError("Connection closed")
                self.buffer += chunk
            result = self.buffer[:n]
            self.buffer = self.buffer[n:]
            return result
    
    reader = BCMDStreamReader(SocketReader(sock))
    
    try:
        for block in reader.iter_blocks():
            process_realtime_data(block)
    finally:
        reader.close()
```

---

## 8. 命令列介面 (CLI)

### 基本用法

```bash
# 查看幫助
python ASMbitBCMD.py --help
python ASMbitBCMD.py compress --help

# 壓縮檔案
python ASMbitBCMD.py compress input.txt output.bzip

# 解壓縮檔案
python ASMbitBCMD.py decompress output.bzip restored.txt

# 查看檔案資訊
python ASMbitBCMD.py info output.bzip

# 執行測試
python ASMbitBCMD.py test

# 執行效能測試
python ASMbitBCMD.py bench
```

### 壓縮命令選項

```bash
python ASMbitBCMD.py compress input.bin output.bzip \
    --kind image \              # 資料類型: auto|text|image|video|audio|generic
    --tier high \               # 算力等級: auto|low|medium|high
    --realtime \                # 即時模式 (低延遲)
    --container file \          # 容器格式: auto|file|stream
    --no-train \                # 禁用 Policy 學習
    --no-patterns \             # 禁用 Patterns Cache
    --learning-dir ./models \   # 學習檔案目錄
    --trace-prefix ./traces/run1  # 追蹤輸出前綴
```

### 輸出範例

```json
{
  "compressed_bytes": 45678,
  "in_path": "input.txt",
  "out_path": "output.bzip",
  "ratio": 0.3456,
  "raw_bytes": 132145,
  "seconds": 1.234,
  "tier": "high",
  "type": "text"
}
```

---

## 9. 視覺化與追蹤

### 9.1 BCMDTrace - 追蹤記錄

```python
from ASMbitBCMD import BCMDTrace, BCMDTraceEvent, TraceExporter, BCMDCodec
import numpy as np

# 建立追蹤物件
trace = BCMDTrace(max_events=10000)

# 手動添加事件 (通常由 codec 自動添加)
trace.add(BCMDTraceEvent(
    block_index=0,
    node_index=0,
    depth=0,
    n_bits=80000,
    action="split",        # "leaf" 或 "split"
    best_gain_bits=1234.5,
    leaf_cost_bits=5000.0,
    split_cost_bits=3765.5,
    mask="F0:p=3",
    n1=40000,
    switches=50,
    t_sec=0.015
))

# 使用 codec 自動追蹤
codec = BCMDCodec()
data = np.random.randint(0, 256, size=100000, dtype=np.uint8).tobytes()
trace = BCMDTrace()
compressed = codec.encode_bytes(data, meta={"test": True}, trace=trace)

print(f"追蹤事件數: {len(trace.events)}")
for ev in trace.events[:5]:
    print(f"  [{ev.action}] depth={ev.depth}, gain={ev.best_gain_bits:.1f}")
```

### 9.2 匯出追蹤視覺化

```python
from ASMbitBCMD import TraceExporter, BCMDTrace

# 匯出所有格式
exports = TraceExporter.export_all(trace, "./output/trace_viz")

print("匯出的檔案:")
print(f"  JSON: {exports['json']}")      # 原始追蹤資料
print(f"  PNG:  {exports['png']}")       # 增益熱圖
print(f"  GIF:  {exports['gif']}")       # 動態演進圖
if 'mp4' in exports:
    print(f"  MP4:  {exports['mp4']}")   # 影片 (需要 cv2)
elif 'mp4_error' in exports:
    print(f"  MP4 錯誤: {exports['mp4_error']}")
```

### 9.3 增益熱圖

```python
from ASMbitBCMD import BCMDTrace, PNGEncoder, scalar_to_heat_u8
import numpy as np

# 從追蹤生成增益矩陣
gain_map = trace.to_gain_map(max_depth=5)  # shape: (depth, events)
print(f"增益矩陣形狀: {gain_map.shape}")

# 轉換為熱圖
idx, rgb = scalar_to_heat_u8(gain_map)

# 儲存 PNG
PNGEncoder.save_rgb("./gain_heatmap.png", rgb)
```

---

## 10. 測試資料生成器

### 10.1 重複區塊資料

```python
from ASMbitBCMD import gen_repeated_block_bytes

# 生成重複區塊資料 (適合測試區塊去重)
data = gen_repeated_block_bytes(
    n_bytes=1000000,      # 總位元組數
    block_bytes=4096,     # 區塊大小
    seed=42,              # 隨機種子
    noise_frac=0.01       # 雜訊比例 (0-1)
)

print(f"資料大小: {len(data)} bytes")
# 結構: 重複的 4KB 區塊 + 1% 隨機雜訊
```

### 10.2 文字語料庫

```python
from ASMbitBCMD import gen_text_corpus_bytes

# 生成結構化文字資料
text_data = gen_text_corpus_bytes(
    n_bytes=500000,
    seed=123,
    noise_frac=0.003      # 0.3% 雜訊
)

print(text_data[:200].decode("utf-8", errors="replace"))
# 輸出包含多語言文字、JSON 片段等結構化內容
```

### 10.3 音訊 PCM 資料

```python
from ASMbitBCMD import gen_audio_pcm16_bytes

# 生成模擬音訊資料 (16-bit PCM)
audio_data = gen_audio_pcm16_bytes(
    n_bytes=200000,
    sample_rate=8000,     # 取樣率
    freq_hz=440,          # 頻率 (A4 音高)
    amp=12000,            # 振幅
    seed=0,
    noise_frac=0.001      # 0.1% 雜訊
)

# 結構: 方波音訊 + 週期性靜音 + 微量雜訊
```

### 10.4 交錯零/隨機位元

```python
from ASMbitBCMD import gen_interleaved_zero_random_bits_bytes

# 生成可穩定觸發 BCMD split 的資料
# bit[i] 偶數 => 0, 奇數 => random(0/1)
data = gen_interleaved_zero_random_bits_bytes(
    n_bytes=100000,
    seed=7
)

# 這種結構對 F0:p=0 遮罩特別有效
```

---

## 11. 完整應用範例

### 11.1 批次壓縮目錄

```python
from ASMbitBCMD import compress_file_easy, decompress_file_easy
import os
import json

def compress_directory(input_dir, output_dir, tier="medium"):
    """批次壓縮目錄中的所有檔案"""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            in_path = os.path.join(root, filename)
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path + ".bzip")
            
            # 建立輸出目錄
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            try:
                result = compress_file_easy(
                    in_path, out_path,
                    kind="auto",
                    tier=tier,
                    enable_training=True
                )
                results.append({
                    "file": rel_path,
                    "status": "ok",
                    "ratio": result["ratio"],
                    "seconds": result["seconds"]
                })
                print(f"✓ {rel_path}: {result['ratio']:.4f}")
            except Exception as e:
                results.append({
                    "file": rel_path,
                    "status": "error",
                    "error": str(e)
                })
                print(f"✗ {rel_path}: {e}")
    
    # 儲存報告
    with open(os.path.join(output_dir, "report.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# 使用範例
results = compress_directory("./documents", "./compressed", tier="high")
```

### 11.2 即時視訊壓縮管線

```python
from ASMbitBCMD import BCMDStreamWriter, BCMDCodec, BCMDCodecConfig, DataKind, ComputeTier, build_codec_config
import cv2
import numpy as np

def video_compression_pipeline(input_video, output_stream):
    """即時視訊壓縮範例"""
    
    # 建立針對視訊優化的配置
    config = build_codec_config(
        kind=DataKind.VIDEO,
        tier=ComputeTier.LOW,  # 視訊使用低算力以保持即時性
        subtype="yuv",
        realtime=True
    )
    codec = BCMDCodec(config)
    
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    with BCMDStreamWriter.open(
        output_stream,
        codec=codec,
        meta={
            "type": "video",
            "fps": fps,
            "width": width,
            "height": height
        }
    ) as writer:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 轉換為 YUV420 格式 (更適合壓縮)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
            
            # 寫入壓縮流
            writer.write(yuv.tobytes())
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
        
        writer.flush()
    
    cap.release()
    print(f"Total frames: {frame_count}")

# 使用範例
video_compression_pipeline("input.mp4", "output.bzs")
```

### 11.3 自訂學習策略

```python
from ASMbitBCMD import (
    AuxMaskPolicyNet, BCMDPatternTable, MaskSpec, MaskFamily,
    BCMDCodec, BCMDCodecConfig
)

# === Policy Net 自訂學習 ===
policy = AuxMaskPolicyNet("./custom_policy.json")
policy.load()

# 手動更新學習資料
for i in range(100):
    mask = MaskSpec(MaskFamily.F1_XOR_SHIFT, p=i % 8, c=i * 17)
    policy.update(
        n_bits=100000,
        ones=50000,
        boundary=10000,
        mask=mask,
        gain=float(100 - i),  # 模擬增益
        type_key="custom_data"
    )

# 取得建議
suggestions = policy.suggest(
    n_bits=100000,
    ones=50000,
    boundary=10000,
    k=10,
    type_key="custom_data"
)
print("Policy 建議的遮罩:")
for mask in suggestions:
    print(f"  {mask.to_str()}")

policy.save()

# === Patterns Cache 自訂管理 ===
patterns = BCMDPatternTable(
    "./custom_patterns.bzipz",
    per_type_limit=512,
    max_groups=64
)
patterns.load()

# 手動添加 pattern
for i in range(300):
    mask = MaskSpec(MaskFamily.F0_SHIFT_BIT, p=i % 12)
    patterns.update(
        type_key="sensor_data",
        mask=mask,
        n_bits=50000,
        gain_bits=float(50 + i % 50)
    )

patterns.save()

# 查看統計
summary = patterns.debug_summary(top_k=5)
print(f"總群組數: {summary['n_groups']}")
for group in summary['groups'][:3]:
    print(f"  類型: {group['type_key']}, 數量: {group['count']}")
    for top in group['top']:
        print(f"    {top['mask']}: seen={top['seen']}, avg_gain={top['avg_gain_per_bit']:.4f}")
```

### 11.4 影像壓縮完整流程

```python
from ASMbitBCMD import (
    compress_testjpg_to_bzip_and_bmp,
    BCMDCodec, BCMDCodecConfig, BCMDTrace,
    PNGEncoder, TraceExporter
)
from PIL import Image
import numpy as np

def image_compression_demo(input_jpg, output_dir):
    """影像壓縮完整示範"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 配置高壓縮率編解碼器
    config = BCMDCodecConfig(
        block_bytes=512 * 1024,
        max_depth=4,
        min_bits=1024,
        auto_train=True,
        use_patterns_cache=True,
        patterns_per_type=512
    )
    codec = BCMDCodec(config)
    
    # 使用內建 JPG -> BMP -> BZIP 管線
    result = compress_testjpg_to_bzip_and_bmp(
        codec=codec,
        jpg_path=input_jpg,
        bzip_path=os.path.join(output_dir, "image.bzip"),
        bmp_path=os.path.join(output_dir, "image.bmp"),
        trace_out_prefix=os.path.join(output_dir, "trace"),
        also_write_png_preview=True
    )
    
    print(f"壓縮結果:")
    print(f"  原始 BMP: {result['raw_bytes']:,} bytes")
    print(f"  壓縮後:   {result['compressed_bytes']:,} bytes")
    print(f"  壓縮率:   {result['ratio']:.4f}")
    print(f"  SHA256:   {result['sha256_bmp'][:32]}...")
    
    if result.get('trace_exports'):
        print(f"追蹤輸出:")
        for fmt, path in result['trace_exports'].items():
            print(f"  {fmt}: {path}")
    
    return result

# 使用範例
result = image_compression_demo("photo.jpg", "./image_demo")
```

### 11.5 完整測試套件

```python
from ASMbitBCMD import testAll

# 執行完整測試套件
# 包含：
# - 單元測試 (資料結構、編解碼、遮罩運算)
# - 整合測試 (檔案 I/O、串流、學習系統)
# - 效能測試 (大規模資料壓縮)

success = testAll()

if success:
    print("所有測試通過！")
else:
    print("部分測試失敗")
    exit(1)
```

---

## 附錄：類別與函數速查表

### 核心類別

| 類別 | 說明 |
|------|------|
| `PackedBitStream` | 打包位元流容器 |
| `MaskSpec` | 遮罩規格定義 |
| `BCMDComputeCore` | JIT 計算核心 |
| `BCMDScorer` | 成本評估器 |
| `BCMDSearchEngine` | 遮罩搜尋引擎 |
| `BCMDCodec` | 完整編解碼器 |
| `BCMDStreamWriter` | 串流寫入器 |
| `BCMDStreamReader` | 串流讀取器 |

### 學習系統

| 類別 | 說明 |
|------|------|
| `AuxMaskPolicyNet` | 特徵導向策略網路 |
| `BCMDPatternTable` | 類型導向 Pattern 快取 |

### 易用函數

| 函數 | 說明 |
|------|------|
| `compress_file_easy()` | 一鍵壓縮檔案 |
| `decompress_file_easy()` | 一鍵解壓縮 |
| `info_file_easy()` | 查看容器資訊 |
| `build_codec_config()` | 建立特化配置 |
| `guess_data_kind_from_path()` | 猜測資料類型 |

### 視覺化

| 類別/函數 | 說明 |
|------|------|
| `BCMDTrace` | 追蹤記錄 |
| `TraceExporter` | 追蹤匯出 (JSON/PNG/GIF/MP4) |
| `PNGEncoder` | 純 Python PNG 編碼器 |
| `GIFEncoder` | 純 Python GIF 編碼器 |

### 測試資料生成

| 函數 | 說明 |
|------|------|
| `gen_repeated_block_bytes()` | 重複區塊 + 雜訊 |
| `gen_text_corpus_bytes()` | 結構化文字語料 |
| `gen_audio_pcm16_bytes()` | 模擬音訊 PCM |
| `gen_interleaved_zero_random_bits_bytes()` | 交錯零/隨機位元 |