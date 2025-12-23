
# ASMbitSpaceMathJIT.py 完整使用說明

## 目錄

1. [概述](#概述)
2. [安裝與配置](#安裝與配置)
3. [核心常量與配置](#核心常量與配置)
4. [超空間定義 (第二章)](#超空間定義-第二章)
5. [距離度量 (第三章)](#距離度量-第三章)
6. [超空間幾何 (第三章)](#超空間幾何-第三章)
7. [代數結構 (第四章)](#代數結構-第四章)
8. [數論變換 (第四章)](#數論變換-第四章)
9. [正向變換 (第五章)](#正向變換-第五章)
10. [高效演算法 (第六章)](#高效演算法-第六章)
11. [逆向變換 (第七章)](#逆向變換-第七章)
12. [視覺化工具](#視覺化工具)
13. [測試與性能基準](#測試與性能基準)
14. [完整應用範例](#完整應用範例)

---

## 概述

`ASMbitSpaceMathJIT.py` 是一個基於位元運算的超空間計算理論實現庫。它提供了：

- **純位元運算實現**：所有數學操作基於位元層級
- **C++ JIT 動態編譯**：可選的高性能後端（透過 ASMbitJIT.py）
- **視覺化輸出**：PNG 格式輸出，不依賴 matplotlib
- **完整測試框架**：包含單元測試與性能基準

### 理論基礎

本模組實現了位元超空間數學理論的七大公理：

| 公理 | 名稱 | 描述 |
|------|------|------|
| I | 閉包性 | 所有位元運算結果仍在超空間內 |
| II | 環面邊界 | 邊界條件形成環面拓撲 |
| III | 原子性 | 最小單位為單一位元 |
| IV | 有限性 | 空間大小為 2^n |
| V | 確定性 | 相同輸入產生相同輸出 |
| VI | 可逆性 | 存在加法逆元 |
| VII | 位元獨立性 | 各位元獨立運算 |

---

## 安裝與配置

### 基本需求

```python
import numpy as np
from ASMbitSpaceMathJIT import *
```

### 可選：高性能後端

```python
# 若有 ASMbitJIT.py，將自動啟用 JIT 加速
# 可透過環境變數控制詳細輸出
import os
os.environ["BITSPACE_VERBOSE"] = "1"
```

### 配置輸出目錄

```python
# 所有視覺化輸出預設存放於 ./bitSpace
BitSpaceConfig.OUTPUT_DIR = "./my_output"
```

---

## 核心常量與配置

### BitSpaceConstants

提供位元超空間常用常量和工具函數。

```python
from ASMbitSpaceMathJIT import BitSpaceConstants

# ===== 預定義位元寬度 =====
print(BitSpaceConstants.BIT_WIDTH_8)   # 8
print(BitSpaceConstants.BIT_WIDTH_16)  # 16
print(BitSpaceConstants.BIT_WIDTH_32)  # 32
print(BitSpaceConstants.BIT_WIDTH_64)  # 64

# ===== 預計算遮罩 =====
print(hex(BitSpaceConstants.MASK_8))   # 0xff
print(hex(BitSpaceConstants.MASK_16))  # 0xffff
print(hex(BitSpaceConstants.MASK_32))  # 0xffffffff

# ===== NTT 相關常量 =====
print(BitSpaceConstants.FERMAT_16)     # 65537 (費馬數 F_4 = 2^16 + 1)
print(BitSpaceConstants.PRIMITIVE_ROOT) # 3 (原根)

# ===== 動態取得遮罩與模數 =====
n = 12
mask = BitSpaceConstants.get_mask(n)      # 2^12 - 1 = 4095
modulus = BitSpaceConstants.get_modulus(n) # 2^12 = 4096

print(f"n={n}: mask={mask}, modulus={modulus}")
```

**輸出：**
```
8
16
32
64
0xff
0xffff
0xffffffff
65537
3
n=12: mask=4095, modulus=4096
```

### BitSpaceConfig

全域配置選項。

```python
from ASMbitSpaceMathJIT import BitSpaceConfig

# 預設位元寬度
BitSpaceConfig.DEFAULT_BIT_WIDTH = 16

# 輸出目錄
BitSpaceConfig.OUTPUT_DIR = "./bitSpace"

# 是否啟用 JIT 加速
BitSpaceConfig.ENABLE_JIT = True

# 詳細輸出模式
BitSpaceConfig.VERBOSE = False
```

---

## 超空間定義 (第二章)

### TorusArithmetic

環面算術運算，實現公理 II（環面邊界）與公理 VI（可逆性）。

```python
from ASMbitSpaceMathJIT import TorusArithmetic

# 創建 8 位元環面算術器
ta = TorusArithmetic(n=8)
print(f"模數 M = 2^8 = {ta.M}")      # 256
print(f"遮罩 MASK = {ta.MASK}")      # 255

# ===== 基本算術運算 =====

# 加法：(a + b) mod 2^n
result = ta.add(255, 1)  # 255 + 1 = 256 → 0 (環繞)
print(f"255 + 1 = {result}")  # 0

result = ta.add(100, 50)
print(f"100 + 50 = {result}")  # 150

# 減法：(a - b) mod 2^n
result = ta.sub(0, 1)  # 0 - 1 = -1 → 255 (環繞)
print(f"0 - 1 = {result}")  # 255

# 乘法：(a × b) mod 2^n
result = ta.mul(16, 16)  # 256 mod 256 = 0
print(f"16 × 16 = {result}")  # 0

# ===== 加法逆元（公理 VI）=====
a = 100
neg_a = ta.neg(a)  # -100 mod 256 = 156
print(f"-{a} = {neg_a}")  # 156
print(f"{a} + {neg_a} = {ta.add(a, neg_a)}")  # 0（驗證逆元）

# ===== 遞增/遞減（公理 II）=====
x = 255
print(f"{x} + 1 = {ta.increment(x)}")  # 0（環繞）
print(f"0 - 1 = {ta.decrement(0)}")    # 255（環繞）

# ===== 環面距離 =====
# 考慮環繞的最短距離
d = ta.distance(10, 250)  # min(|10-250|, 256-|10-250|) = min(240, 16) = 16
print(f"distance(10, 250) = {d}")  # 16

# ===== 包裹函數 =====
print(f"wrap(300) = {ta.wrap(300)}")   # 300 mod 256 = 44
print(f"wrap(-5) = {ta.wrap(-5)}")     # -5 mod 256 = 251
```

### HyperSpace1D

一維位元超空間 H_n^(1) = Z_{2^n}。

```python
from ASMbitSpaceMathJIT import HyperSpace1D

# 創建 8 位元一維超空間
h = HyperSpace1D(n=8)

# ===== 空間屬性 =====
print(f"維度: {h.dimension()}")  # 1
print(f"大小: {h.size()}")       # 256 (= 2^8)
print(f"位元寬度: {h.n}")        # 8

# ===== 成員檢查 =====
print(f"0 in h: {0 in h}")       # True
print(f"255 in h: {255 in h}")   # True
print(f"256 in h: {256 in h}")   # False（超出範圍）

# ===== 座標包裹 =====
print(f"wrap(256) = {h.wrap(256)}")  # 0
print(f"wrap(-1) = {h.wrap(-1)}")    # 255

# ===== 迭代所有元素 =====
# 適用於小空間
count = 0
for x in h.iterate():
    count += 1
    if count <= 5:
        print(f"元素: {x}")
print(f"總共 {count} 個元素")

# ===== 使用內建算術器 =====
result = h.arithmetic.add(200, 100)
print(f"200 + 100 (mod 256) = {result}")  # 44
```

### HyperSpace2D

二維位元超空間 H_n^(2) = Z_{2^n} × Z_{2^n}。

```python
from ASMbitSpaceMathJIT import HyperSpace2D

# 創建 4 位元二維超空間
h = HyperSpace2D(n=4)

# ===== 空間屬性 =====
print(f"維度: {h.dimension()}")  # 2
print(f"大小: {h.size()}")       # 256 (= 16 × 16 = 2^8)

# ===== 成員檢查 =====
print(f"(0, 0) in h: {(0, 0) in h}")      # True
print(f"(15, 15) in h: {(15, 15) in h}")  # True
print(f"(16, 0) in h: {(16, 0) in h}")    # False

# ===== 座標包裹 =====
x, y = h.wrap(20, -3)
print(f"wrap(20, -3) = ({x}, {y})")  # (4, 13)

# ===== 環面曼哈頓距離（定義3.3）=====
# 考慮環繞的最短曼哈頓距離
p1 = (0, 0)
p2 = (15, 15)
d = h.torus_manhattan_distance(p1, p2)
print(f"torus_manhattan_distance{p1, p2} = {d}")  # 2（走環繞更近）

p3 = (5, 5)
d = h.torus_manhattan_distance(p1, p3)
print(f"torus_manhattan_distance{p1, p3} = {d}")  # 10（直接走）

# ===== 迭代所有元素 =====
count = 0
for x, y in h.iterate():
    count += 1
print(f"總共 {count} 個元素")  # 256
```

### HyperSpaceND

k 維位元超空間 H_n^(k)。

```python
from ASMbitSpaceMathJIT import HyperSpaceND

# 創建 4 位元 3 維超空間
h = HyperSpaceND(n=4, k=3)

# ===== 空間屬性 =====
print(f"維度: {h.dimension()}")  # 3
print(f"位元寬度: {h.n}")        # 4
print(f"大小: {h.size()}")       # 4096 (= 16^3 = 2^12)

# ===== 成員檢查 =====
print(f"(0, 0, 0) in h: {(0, 0, 0) in h}")        # True
print(f"(15, 15, 15) in h: {(15, 15, 15) in h}")  # True
print(f"(16, 0, 0) in h: {(16, 0, 0) in h}")      # False

# ===== 座標包裹 =====
wrapped = h.wrap(20, -3, 100)
print(f"wrap(20, -3, 100) = {wrapped}")  # (4, 13, 4)
```

---

## 距離度量 (第三章)

### BitMetrics

位元度量空間的各種距離與相似度計算。

```python
from ASMbitSpaceMathJIT import BitMetrics
import numpy as np

# ===== XOR 度量（定義3.1）=====
# d_XOR(a, b) = a ⊕ b
a, b = 0b1100, 0b1010
d = BitMetrics.xor_distance(a, b)
print(f"XOR distance: {a:04b} ⊕ {b:04b} = {d:04b} ({d})")  # 0110 (6)

# ===== 漢明度量（定義3.2）=====
# d_H(a, b) = popcount(a ⊕ b)
d = BitMetrics.hamming_distance_int(a, b)
print(f"Hamming distance: {d}")  # 2（兩個位元不同）

# ===== Popcount =====
x = 0b11110000
pc = BitMetrics.popcount(x)
print(f"popcount({x:08b}) = {pc}")  # 4

# ===== 位元組陣列操作（高性能）=====
arr_a = np.array([0xFF, 0x00, 0xAA], dtype=np.uint8)
arr_b = np.array([0xFF, 0xFF, 0x55], dtype=np.uint8)

# 漢明距離
hd = BitMetrics.hamming_distance_bytes(arr_a, arr_b)
print(f"Hamming distance (bytes): {hd}")  # 8 + 8 = 16

# 陣列總 popcount
pc = BitMetrics.popcount_array(arr_a)
print(f"Total popcount: {pc}")  # 8 + 0 + 4 = 12

# XNOR 相似度（匹配位元數）
sim = BitMetrics.xnor_similarity(arr_a, arr_b)
print(f"XNOR similarity: {sim}")  # 24 - 16 = 8
```

### HammingCircle

漢明圓 C_H(c, r) = {x : d_H(x, c) = r}，即與圓心漢明距離恰為 r 的所有點。

```python
from ASMbitSpaceMathJIT import HammingCircle

# 創建圓心為 0，半徑為 2，位元寬度為 4 的漢明圓
circle = HammingCircle(center=0, radius=2, n=4)

# ===== 圓的大小（定理3.2）=====
# |C_H(c, r)| = C(n, r) = C(4, 2) = 6
print(f"圓的大小: {circle.size()}")  # 6

# ===== 成員檢查 =====
# 與 0 有 2 個位元不同的數
print(f"0b0011 (3) 在圓上: {circle.contains(0b0011)}")   # True
print(f"0b0101 (5) 在圓上: {circle.contains(0b0101)}")   # True
print(f"0b1001 (9) 在圓上: {circle.contains(0b1001)}")   # True
print(f"0b0001 (1) 在圓上: {circle.contains(0b0001)}")   # False（只有 1 個位元不同）
print(f"0b0111 (7) 在圓上: {circle.contains(0b0111)}")   # False（有 3 個位元不同）

# ===== 枚舉圓上所有點 =====
print("圓上所有點:")
for point in circle.enumerate():
    print(f"  {point:04b} ({point})")
# 輸出: 0011(3), 0101(5), 0110(6), 1001(9), 1010(10), 1100(12)
```

### HammingBall

漢明球 B_H(c, r) = {x : d_H(x, c) ≤ r}，即與圓心漢明距離不超過 r 的所有點。

```python
from ASMbitSpaceMathJIT import HammingBall

# 創建圓心為 0，半徑為 2，位元寬度為 4 的漢明球
ball = HammingBall(center=0, radius=2, n=4)

# ===== 球的大小 =====
# |B_H(c, r)| = Σ_{i=0}^{r} C(n, i) = C(4,0) + C(4,1) + C(4,2) = 1 + 4 + 6 = 11
print(f"球的大小: {ball.size()}")  # 11

# ===== 成員檢查 =====
print(f"0b0000 (0) 在球內: {ball.contains(0b0000)}")   # True (距離 0)
print(f"0b0001 (1) 在球內: {ball.contains(0b0001)}")   # True (距離 1)
print(f"0b0011 (3) 在球內: {ball.contains(0b0011)}")   # True (距離 2)
print(f"0b0111 (7) 在球內: {ball.contains(0b0111)}")   # False (距離 3)
```

---

## 超空間幾何 (第三章)

### BitLine

位元直線 L(p, v)，由起點 p 和方向向量 v 定義。

```python
from ASMbitSpaceMathJIT import BitLine

# 創建起點為 0，方向為 0b0111，位元寬度為 4 的位元直線
line = BitLine(start=0, direction=0b0111, n=4)

# ===== 直線大小（定理3.3）=====
# |L(p, v)| = 2^w，其中 w = popcount(v)
# direction = 0b0111 有 3 個非零位元，所以 size = 2^3 = 8
print(f"直線大小: {line.size()}")  # 8

# ===== 成員檢查 =====
# 直線上的點 = start XOR (direction 的任意子集)
print(f"0b0000 在直線上: {line.contains(0b0000)}")  # True (start)
print(f"0b0001 在直線上: {line.contains(0b0001)}")  # True
print(f"0b0011 在直線上: {line.contains(0b0011)}")  # True
print(f"0b0111 在直線上: {line.contains(0b0111)}")  # True
print(f"0b1000 在直線上: {line.contains(0b1000)}")  # False

# ===== 枚舉直線上所有點 =====
print("直線上所有點:")
for point in line.enumerate():
    print(f"  {point:04b} ({point})")
# 輸出: 0000, 0001, 0010, 0011, 0100, 0101, 0110, 0111
```

### BitGeometryOps

位元幾何操作工具類。

```python
from ASMbitSpaceMathJIT import BitGeometryOps
import numpy as np

# ===== 謝爾賓斯基條件（定理3.4）=====
# f(x, y) = 1 if (x AND y) = 0
print(f"(0, 0): {BitGeometryOps.sierpinski_check(0, 0)}")  # True
print(f"(1, 0): {BitGeometryOps.sierpinski_check(1, 0)}")  # True
print(f"(1, 1): {BitGeometryOps.sierpinski_check(1, 1)}")  # False
print(f"(2, 1): {BitGeometryOps.sierpinski_check(2, 1)}")  # True (0b10 & 0b01 = 0)
print(f"(3, 1): {BitGeometryOps.sierpinski_check(3, 1)}")  # False (0b11 & 0b01 = 1)

# ===== 生成謝爾賓斯基三角形 =====
pattern = BitGeometryOps.generate_sierpinski(size=8)
print("謝爾賓斯基三角形 (8x8):")
for row in pattern:
    print(''.join(['█' if x else ' ' for x in row]))
```

**輸出：**
```
謝爾賓斯基三角形 (8x8):
████████
█ █ █ █ 
██  ██  
█   █   
████    
█ █     
██      
█       
```

### SierpinskiGenerator

謝爾賓斯基碎形生成器，提供更多分析功能。

```python
from ASMbitSpaceMathJIT import SierpinskiGenerator

# 創建 256x256 的謝爾賓斯基生成器
gen = SierpinskiGenerator(size=256)

# ===== 獲取圖案 =====
pattern = gen.pattern  # 惰性計算，首次訪問時生成
print(f"圖案形狀: {pattern.shape}")  # (256, 256)
print(f"填充像素數: {np.sum(pattern)}")

# ===== 填充密度 =====
density = gen.get_density()
print(f"填充密度: {density:.6f}")  # 約 0.190

# ===== 碎形維度 =====
# 謝爾賓斯基三角形的理論維度 = log(3)/log(2) ≈ 1.585
dim = gen.fractal_dimension()
print(f"碎形維度: {dim:.6f}")  # 1.584963
```

---

## 代數結構 (第四章)

### TorusRing

環結構 (H_n, +_H, ×_H)，含單位元的交換環。

```python
from ASMbitSpaceMathJIT import TorusRing

ring = TorusRing(n=8)

# ===== 單位元 =====
print(f"加法單位元: {ring.zero()}")  # 0
print(f"乘法單位元: {ring.one()}")   # 1

# ===== 環運算 =====
a, b, c = 100, 150, 200

# 加法
print(f"{a} + {b} = {ring.add(a, b)}")  # 250
print(f"{a} + {c} = {ring.add(a, c)}")  # 44 (300 mod 256)

# 乘法
print(f"16 × 16 = {ring.mul(16, 16)}")  # 0 (256 mod 256)
print(f"10 × 20 = {ring.mul(10, 20)}")  # 200

# 加法逆元
neg_a = ring.additive_inverse(a)
print(f"-{a} = {neg_a}")  # 156
print(f"{a} + (-{a}) = {ring.add(a, neg_a)}")  # 0

# ===== 驗證環公理 =====
import random

# 結合律
for _ in range(10):
    x, y, z = [random.randint(0, 255) for _ in range(3)]
    assert ring.add(ring.add(x, y), z) == ring.add(x, ring.add(y, z))
    assert ring.mul(ring.mul(x, y), z) == ring.mul(x, ring.mul(y, z))

# 交換律
for _ in range(10):
    x, y = random.randint(0, 255), random.randint(0, 255)
    assert ring.add(x, y) == ring.add(y, x)
    assert ring.mul(x, y) == ring.mul(y, x)

# 分配律
for _ in range(10):
    x, y, z = [random.randint(0, 255) for _ in range(3)]
    assert ring.mul(x, ring.add(y, z)) == ring.add(ring.mul(x, y), ring.mul(x, z))

print("所有環公理驗證通過！")
```

### BooleanRing

布林環 (H_n, ⊕, ∧)，XOR 為加法，AND 為乘法。

```python
from ASMbitSpaceMathJIT import BooleanRing

ring = BooleanRing(n=8)

# ===== 單位元 =====
print(f"加法單位元 (XOR): {ring.zero()}")  # 0
print(f"乘法單位元 (AND): {ring.one()}")   # 255 (全 1)

# ===== 布林環運算 =====
a = 0b11001100
b = 0b10101010

# 加法 (XOR)
result = ring.add(a, b)
print(f"{a:08b} ⊕ {b:08b} = {result:08b}")  # 01100110

# 乘法 (AND)
result = ring.mul(a, b)
print(f"{a:08b} ∧ {b:08b} = {result:08b}")  # 10001000

# ===== 布林環特性 =====
# 每個元素都是自己的加法逆元
print(f"a = {a:08b}")
print(f"-a = {ring.additive_inverse(a):08b}")  # 同樣是 11001100
print(f"a ⊕ a = {ring.add(a, a)}")  # 0

# 冪等性: a ∧ a = a
print(f"a ∧ a = {ring.mul(a, a):08b}")  # 11001100

# 與全 1 做 AND 得到自己
print(f"a ∧ 1 = {ring.mul(a, ring.one()):08b}")  # 11001100
```

### BitCalculus

位元微積分，實現離散微分與積分運算。

```python
from ASMbitSpaceMathJIT import BitCalculus
import numpy as np

# ===== 一維時間位元微分（定義4.2）=====
# ∇_t f(t) = f(t) ⊕ f(t-1)
f = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.uint8)
print(f"原函數:   {list(f)}")

deriv = BitCalculus.derivative_1d(f)
print(f"導數:     {list(deriv)}")  # 顯示相鄰元素的變化

# ===== 一維位元積分（定義4.6）=====
# ∫_0^x f = f(0) ⊕ f(1) ⊕ ... ⊕ f(x)
integ = BitCalculus.integral_1d(f)
print(f"積分:     {list(integ)}")

# ===== 驗證位元微積分基本定理（定理4.3）=====
# ∇_t(∫_0^t f) = f(t)
is_valid = BitCalculus.verify_fundamental_theorem(f)
print(f"基本定理驗證: {is_valid}")  # True

# ===== 二維空間位元微分（定義4.3）=====
F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1]
], dtype=np.uint8)

# x 方向導數: ∇_x F(x,y) = F(x,y) ⊕ F(x-1,y)
dx = BitCalculus.derivative_2d_x(F)
print("x 方向導數:")
print(dx)

# y 方向導數: ∇_y F(x,y) = F(x,y) ⊕ F(x,y-1)
dy = BitCalculus.derivative_2d_y(F)
print("y 方向導數:")
print(dy)

# ===== 位元梯度（定義4.4）=====
grad_x, grad_y = BitCalculus.gradient_2d(F)
print(f"梯度 x 分量形狀: {grad_x.shape}")
print(f"梯度 y 分量形狀: {grad_y.shape}")

# ===== 位元拉普拉斯算子（定義4.5）=====
# Δ_⊕ F = F(x+1,y) ⊕ F(x-1,y) ⊕ F(x,y+1) ⊕ F(x,y-1)
laplacian = BitCalculus.laplacian_2d(F)
print("拉普拉斯算子結果:")
print(laplacian)
```

---

## 數論變換 (第四章)

### NTT

數論變換 (Number Theoretic Transform)，用於整數域的頻譜分析。

```python
from ASMbitSpaceMathJIT import NTT
import numpy as np

# 創建 16 位元 NTT（費馬數 p = 2^16 + 1 = 65537）
ntt = NTT(n=16)
print(f"費馬模數 p = {ntt.p}")     # 65537
print(f"原根 g = {ntt.g}")         # 3

# ===== 基本變換 =====
a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
print(f"原始序列: {list(a)}")

# 正變換
A = ntt.forward(a)
print(f"NTT 頻譜: {list(A)}")

# 逆變換
a_recovered = ntt.inverse(A)
print(f"重建序列: {list(a_recovered)}")

# 驗證完美重建
assert np.array_equal(a, a_recovered)
print("完美重建驗證通過！")

# ===== 使用 NTT 計算多項式乘法（快速卷積）=====
def polynomial_multiply_ntt(p1, p2, ntt):
    """使用 NTT 計算多項式乘法"""
    n = len(p1) + len(p2) - 1
    # 找到最接近的 2 的冪次
    size = 1
    while size < n:
        size *= 2
    
    # 零填充
    p1_padded = np.zeros(size, dtype=np.int64)
    p2_padded = np.zeros(size, dtype=np.int64)
    p1_padded[:len(p1)] = p1
    p2_padded[:len(p2)] = p2
    
    # NTT 變換
    P1 = ntt.forward(p1_padded)
    P2 = ntt.forward(p2_padded)
    
    # 逐點相乘
    C = (P1 * P2) % ntt.p
    
    # 逆變換
    result = ntt.inverse(C)
    
    return result[:n]

# 計算 (1 + 2x + 3x²) × (4 + 5x + 6x²)
p1 = np.array([1, 2, 3], dtype=np.int64)
p2 = np.array([4, 5, 6], dtype=np.int64)
product = polynomial_multiply_ntt(p1, p2, ntt)
print(f"多項式乘積係數: {list(product)}")
# 預期: [4, 13, 28, 27, 18] 代表 4 + 13x + 28x² + 27x³ + 18x⁴
```

### WHT

沃爾什-阿達馬變換 (Walsh-Hadamard Transform)，用於 XOR 卷積。

```python
from ASMbitSpaceMathJIT import WHT
import numpy as np

# ===== 基本變換 =====
x = np.array([1, 2, 3, 4], dtype=np.int64)
print(f"原始序列: {list(x)}")

# 正變換
X = WHT.transform(x.copy())
print(f"WHT 頻譜: {list(X)}")

# 逆變換
x_recovered = WHT.transform(X.copy(), inverse=True)
print(f"重建序列: {list(x_recovered)}")

assert np.array_equal(x, x_recovered)
print("完美重建驗證通過！")

# ===== XOR 卷積（定義6.1）=====
# (f ⊛ g)(k) = Σ_{i⊕j=k} f(i)·g(j)
a = np.array([1, 2, 3, 4], dtype=np.int64)
b = np.array([4, 3, 2, 1], dtype=np.int64)

# 使用 WHT 計算 XOR 卷積
c = WHT.xor_convolution(a, b)
print(f"XOR 卷積結果: {list(c)}")

# 手動驗證
expected = np.zeros(4, dtype=np.int64)
for i in range(4):
    for j in range(4):
        k = i ^ j
        expected[k] += a[i] * b[j]
print(f"手動計算結果: {list(expected)}")

assert np.array_equal(c, expected)
print("XOR 卷積驗證通過！")
```

---

## 正向變換 (第五章)

### MortonCodec

莫頓編碼 (Z-order Curve)，將二維座標交錯編碼為一維索引。

```python
from ASMbitSpaceMathJIT import MortonCodec
import numpy as np

# ===== 基本編碼與解碼 =====
x, y = 5, 7
z = MortonCodec.encode(x, y, n=4)
print(f"Morton encode ({x}, {y}) = {z} (0b{z:08b})")

x2, y2 = MortonCodec.decode(z, n=4)
print(f"Morton decode {z} = ({x2}, {y2})")

# ===== 理解莫頓編碼原理 =====
# x = 5 = 0b0101
# y = 7 = 0b0111
# 交錯: y3 x3 y2 x2 y1 x1 y0 x0
#       1  0  1  1  1  1  1  1 = 0b10111111 = 191
print("\n莫頓編碼原理:")
print(f"x = {x} = 0b{x:04b}")
print(f"y = {y} = 0b{y:04b}")
print("交錯後: y₃x₃y₂x₂y₁x₁y₀x₀")

# ===== 批量編碼 =====
coords = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 0],
    [3, 0]
], dtype=np.int32)

morton_codes = MortonCodec.encode_array(coords, n=4)
print("\n批量莫頓編碼:")
for i, (x, y) in enumerate(coords):
    print(f"  ({x}, {y}) → {morton_codes[i]}")

# ===== 莫頓編碼的空間局部性 =====
# 莫頓曲線保持空間局部性
print("\n莫頓曲線順序 (4x4):")
for z in range(16):
    x, y = MortonCodec.decode(z, n=2)
    print(f"  z={z:2d} → ({x}, {y})")
```

### ImageTransform

影像變換 Φ_I。

```python
from ASMbitSpaceMathJIT import ImageTransform
import numpy as np

# ===== 灰階影像編碼（定義5.2）=====
gray_image = np.array([
    [100, 150, 200],
    [50,  100, 150],
    [0,   50,  100]
], dtype=np.uint8)

encoded = ImageTransform.encode_grayscale(gray_image)
print("灰階編碼結果:")
print(encoded)

# ===== RGB 影像編碼（定義5.3）=====
# Φ_I^RGB(r, g, b) = (r << 16) | (g << 8) | b
r = np.array([[255, 0], [128, 64]], dtype=np.uint8)
g = np.array([[0, 255], [128, 64]], dtype=np.uint8)
b = np.array([[0, 0], [128, 192]], dtype=np.uint8)

packed = ImageTransform.encode_rgb(r, g, b)
print("\nRGB 編碼結果:")
print(f"形狀: {packed.shape}")
for y in range(2):
    for x in range(2):
        print(f"  ({y},{x}): 0x{packed[y,x]:06X} → R={r[y,x]:3d}, G={g[y,x]:3d}, B={b[y,x]:3d}")

# ===== RGB 影像解碼 =====
r2, g2, b2 = ImageTransform.decode_rgb(packed)
print("\nRGB 解碼驗證:")
print(f"  R 通道相同: {np.array_equal(r, r2)}")
print(f"  G 通道相同: {np.array_equal(g, g2)}")
print(f"  B 通道相同: {np.array_equal(b, b2)}")

# ===== 莫頓編碼的影像變換 =====
small_img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
morton_encoded = ImageTransform.encode_with_morton(small_img, n=8)
print("\n莫頓編碼影像變換:")
print(morton_encoded)
```

### TextTransform

文字變換 Φ_T。

```python
from ASMbitSpaceMathJIT import TextTransform
import numpy as np

# ===== 字元編碼（定義5.5）=====
char = 'A'
code = TextTransform.encode_char(char)
print(f"encode_char('{char}') = {code}")  # 65

decoded = TextTransform.decode_char(code)
print(f"decode_char({code}) = '{decoded}'")  # 'A'

# ===== 字串編碼（定義5.6）=====
text = "Hello, 世界!"
encoded = TextTransform.encode_string(text)
print(f"\n字串 '{text}' 編碼:")
print(f"  長度: {len(encoded)}")
print(f"  編碼: {list(encoded)}")

decoded = TextTransform.decode_string(encoded)
print(f"  解碼: '{decoded}'")

# ===== 滾動雜湊（定義5.7）=====
# Φ_T^hash(S) = Σ_{i=0}^{m-1} c_i · b^{m-1-i} mod 2^n
s1 = "hello"
s2 = "hello"
s3 = "world"

h1 = TextTransform.rolling_hash(s1, base=257, n=32)
h2 = TextTransform.rolling_hash(s2, base=257, n=32)
h3 = TextTransform.rolling_hash(s3, base=257, n=32)

print(f"\n滾動雜湊:")
print(f"  hash('{s1}') = {h1}")
print(f"  hash('{s2}') = {h2}")
print(f"  hash('{s3}') = {h3}")
print(f"  '{s1}' == '{s2}': {h1 == h2}")  # True
print(f"  '{s1}' == '{s3}': {h1 == h3}")  # False

# ===== N-gram 位元簽名（定義5.8）=====
text1 = "the quick brown fox"
text2 = "the quick brown dog"
text3 = "completely different"

sig1 = TextTransform.ngram_signature(text1, gram_size=2, n=64)
sig2 = TextTransform.ngram_signature(text2, gram_size=2, n=64)
sig3 = TextTransform.ngram_signature(text3, gram_size=2, n=64)

# 計算簽名相似度（共同位元數）
def signature_similarity(a, b, n=64):
    common = bin(a & b).count('1')
    total = bin(a | b).count('1')
    return common / total if total > 0 else 0

print(f"\nN-gram 簽名相似度:")
print(f"  '{text1}' vs '{text2}': {signature_similarity(sig1, sig2):.3f}")
print(f"  '{text1}' vs '{text3}': {signature_similarity(sig1, sig3):.3f}")
```

### LogicTransform

邏輯變換 Φ_L。

```python
from ASMbitSpaceMathJIT import LogicTransform

# ===== 布林函數編碼（定義5.9）=====
# 將布林函數編碼為真值表的整數表示

# AND 閘
and_tt = LogicTransform.and_gate_encoding()
print(f"AND 閘真值表: {and_tt:04b} ({and_tt})")  # 1000 (8)

# OR 閘
or_tt = LogicTransform.or_gate_encoding()
print(f"OR 閘真值表:  {or_tt:04b} ({or_tt})")   # 1110 (14)

# XOR 閘
xor_tt = LogicTransform.xor_gate_encoding()
print(f"XOR 閘真值表: {xor_tt:04b} ({xor_tt})")  # 0110 (6)

# ===== 自訂布林函數編碼 =====
# 編碼 NAND 閘
def nand(a, b):
    return not (a and b)

nand_tt = LogicTransform.encode_boolean_function(nand, num_vars=2)
print(f"\nNAND 閘真值表: {nand_tt:04b} ({nand_tt})")  # 0111 (7)

# 編碼 3 輸入多數函數
def majority3(a, b, c):
    return (a + b + c) >= 2

maj_tt = LogicTransform.encode_boolean_function(majority3, num_vars=3)
print(f"3輸入多數函數: {maj_tt:08b} ({maj_tt})")  # 11101000 (232)

# ===== 布林函數解碼 =====
# 從真值表重建函數
decoded_and = LogicTransform.decode_boolean_function(and_tt, num_vars=2)

print("\n驗證解碼的 AND 函數:")
for a in [0, 1]:
    for b in [0, 1]:
        result = decoded_and(a, b)
        expected = a and b
        print(f"  AND({a}, {b}) = {result} (預期: {expected})")

# ===== 真值表的位元表示說明 =====
print("\n真值表位元表示說明:")
print("  對於 2 輸入函數 f(a, b):")
print("  位元位置 i = 2*b + a 對應輸入 (a, b)")
print("  AND: f(0,0)=0, f(1,0)=0, f(0,1)=0, f(1,1)=1 → 0b1000 = 8")
print("  OR:  f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=1 → 0b1110 = 14")
print("  XOR: f(0,0)=0, f(1,0)=1, f(0,1)=1, f(1,1)=0 → 0b0110 = 6")
```

### UnifiedTransform

統一變換框架。

```python
from ASMbitSpaceMathJIT import UnifiedTransform

# ===== 類型標籤 =====
print("類型標籤:")
for tag in UnifiedTransform.TypeTag:
    print(f"  {tag.name}: {tag.value}")

# ===== 統一編碼與解碼 =====
# 整數
int_data = 12345
encoded_int = UnifiedTransform.encode(int_data, UnifiedTransform.TypeTag.INTEGER)
print(f"\n整數編碼: {encoded_int}")

# 文字
text_data = "Hello"
encoded_text = UnifiedTransform.encode(text_data, UnifiedTransform.TypeTag.TEXT)
print(f"文字編碼: {encoded_text}")

# 解碼
decoded = UnifiedTransform.decode(encoded_int)
print(f"解碼整數: {decoded}")
```

---

## 高效演算法 (第六章)

### LookupTableEngine

查找表引擎，實現 O(1) 查詢。

```python
from ASMbitSpaceMathJIT import LookupTableEngine
import numpy as np

# 創建 8 位元查找表引擎
engine = LookupTableEngine(n=8)
print(f"表大小: {engine.size}")  # 256

# ===== 創建平方表 =====
sq_table = engine.create_square_table()
print("\n平方查找:")
for x in [0, 5, 10, 15, 16]:
    result = engine.lookup('square', x)
    print(f"  {x}² = {result}")

# ===== 創建平方根表 =====
sqrt_table = engine.create_sqrt_table()
print("\n平方根查找 (整數部分):")
for x in [0, 4, 9, 16, 25, 100]:
    result = engine.lookup('sqrt', x)
    print(f"  √{x} = {result}")

# ===== 創建 popcount 表 =====
pc_table = engine.create_popcount_table()
print("\nPopcount 查找:")
for x in [0, 0x0F, 0xAA, 0xFF]:
    result = engine.lookup('popcount', x)
    print(f"  popcount(0x{x:02X}) = {result}")

# ===== 創建位元反轉表 =====
rev_table = engine.create_bit_reverse_table()
print("\n位元反轉查找:")
for x in [0b00000001, 0b10000000, 0b11110000, 0b10101010]:
    result = engine.lookup('bit_reverse', x)
    print(f"  reverse(0b{x:08b}) = 0b{result:08b}")

# ===== 自訂查找表 =====
# 創建 Gray code 表
gray_table = engine.create_table('gray', lambda x: x ^ (x >> 1))
print("\nGray code 查找:")
for x in range(8):
    result = engine.lookup('gray', x)
    print(f"  gray({x}) = {result} (0b{result:04b})")
```

### BitConvolution

位元卷積運算。

```python
from ASMbitSpaceMathJIT import BitConvolution
import numpy as np

# 測試數據
a = np.array([1, 2, 3, 4], dtype=np.int64)
b = np.array([4, 3, 2, 1], dtype=np.int64)

# ===== XOR 卷積 =====
# (f ⊛ g)(k) = Σ_{i⊕j=k} f(i)·g(j)
c_xor = BitConvolution.xor_convolution(a, b)
print("XOR 卷積:")
print(f"  a = {list(a)}")
print(f"  b = {list(b)}")
print(f"  a ⊛_XOR b = {list(c_xor)}")

# 手動驗證
print("\n驗證 XOR 卷積 c[k] = Σ_{i⊕j=k} a[i]·b[j]:")
for k in range(4):
    terms = []
    for i in range(4):
        for j in range(4):
            if i ^ j == k:
                terms.append(f"a[{i}]·b[{j}]={a[i]*b[j]}")
    print(f"  c[{k}] = {' + '.join(terms)} = {c_xor[k]}")

# ===== AND 卷積 =====
c_and = BitConvolution.and_convolution(a, b)
print(f"\nAND 卷積: {list(c_and)}")

# ===== OR 卷積 =====
c_or = BitConvolution.or_convolution(a, b)
print(f"OR 卷積: {list(c_or)}")

# ===== 實際應用：集合函數卷積 =====
print("\n應用範例：集合函數")
# 假設 a[S] = 集合 S 的某個值，S 用位元表示
# AND 卷積用於計算子集相關的求和
# OR 卷積用於計算超集相關的求和
# XOR 卷積用於計算對稱差相關的求和
```

### BitGraphAlgorithms

位元圖演算法。

```python
from ASMbitSpaceMathJIT import BitGraphAlgorithms
import numpy as np

# ===== 位元並行 BFS（演算法6.2）=====
# 圖的鄰接矩陣以位元遮罩表示
# 節點 i 的鄰居 = adj[i] 的設置位元

# 範例圖：0 -- 1 -- 2 -- 3 (鏈狀)
#         0 -> 1: adj[0] = 0b0010
#         1 -> 0, 2: adj[1] = 0b0101
#         2 -> 1, 3: adj[2] = 0b1010
#         3 -> 2: adj[3] = 0b0100

adj = [
    0b0010,  # 節點 0 連接到節點 1
    0b0101,  # 節點 1 連接到節點 0, 2
    0b1010,  # 節點 2 連接到節點 1, 3
    0b0100,  # 節點 3 連接到節點 2
]

reachable = BitGraphAlgorithms.bit_bfs(adj, start=0, num_nodes=4)
print("位元 BFS 結果:")
print(f"  從節點 0 可達的節點: {reachable:04b}")
print(f"  可達節點列表: {[i for i in range(4) if reachable & (1 << i)]}")

# 更複雜的圖
#     0 --- 1
#     |     |
#     3 --- 2
#     
#     4 --- 5 (獨立的連通分量)

adj2 = [
    0b1010,  # 0 -> 1, 3
    0b0101,  # 1 -> 0, 2
    0b1010,  # 2 -> 1, 3
    0b0101,  # 3 -> 0, 2
    0b100000,  # 4 -> 5
    0b010000,  # 5 -> 4
]

reachable_from_0 = BitGraphAlgorithms.bit_bfs(adj2, start=0, num_nodes=6)
reachable_from_4 = BitGraphAlgorithms.bit_bfs(adj2, start=4, num_nodes=6)

print(f"\n從節點 0 可達: {[i for i in range(6) if reachable_from_0 & (1 << i)]}")
print(f"從節點 4 可達: {[i for i in range(6) if reachable_from_4 & (1 << i)]}")

# ===== 距離場波傳播（演算法6.3）=====
# 在 2D 網格上計算到起點的最短距離

grid = np.ones((10, 10), dtype=np.uint8)
# 添加障礙物
grid[3:7, 5] = 0  # 垂直牆

start = (2, 5)
dist = BitGraphAlgorithms.distance_field_2d(grid, start)

print("\n距離場結果 (起點=(2,5), -1=障礙或不可達):")
print(dist)

# 檢查特定點的距離
print(f"\n起點 {start} 的距離: {dist[start[1], start[0]]}")
print(f"點 (8, 5) 的距離: {dist[5, 8]}")  # 需要繞過牆壁
```

### BitImageFilters

位元影像濾波器。

```python
from ASMbitSpaceMathJIT import BitImageFilters
import numpy as np

# 創建測試影像
img = np.array([
    [255, 255, 255, 0, 0, 0, 0, 0],
    [255, 255, 255, 0, 0, 0, 0, 0],
    [255, 255, 255, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 255, 255, 0, 0],
    [0, 0, 0, 255, 255, 255, 0, 0],
    [0, 0, 0, 255, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 255, 255],
    [0, 0, 0, 0, 0, 0, 255, 255],
], dtype=np.uint8)

print("原始影像:")
print(img)

# ===== XOR 模糊濾波器（定義6.2）=====
# 產生邊緣檢測效果
xor_result = BitImageFilters.xor_blur(img)
print("\nXOR 模糊 (邊緣檢測):")
print(xor_result)

# ===== AND 濾波器（定義6.3）=====
# 保留鄰近像素共有的特徵
and_result = BitImageFilters.and_filter(img)
print("\nAND 濾波 (侵蝕效果):")
print(and_result)

# ===== 形態學操作 =====
binary_img = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0],
    [0, 255, 255, 255, 255, 255, 0, 0],
    [0, 255, 255, 255, 255, 255, 0, 0],
    [0, 255, 255, 255, 255, 255, 0, 0],
    [0, 0, 255, 255, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.uint8)

print("\n二值影像:")
print((binary_img > 0).astype(int))

# 膨脹
dilated = BitImageFilters.dilate(binary_img)
print("\n膨脹後:")
print((dilated > 0).astype(int))

# 侵蝕
eroded = BitImageFilters.erode(binary_img)
print("\n侵蝕後:")
print((eroded > 0).astype(int))

# ===== 開運算 (侵蝕後膨脹) =====
opened = BitImageFilters.dilate(BitImageFilters.erode(binary_img))
print("\n開運算後:")
print((opened > 0).astype(int))

# ===== 閉運算 (膨脹後侵蝕) =====
closed = BitImageFilters.erode(BitImageFilters.dilate(binary_img))
print("\n閉運算後:")
print((closed > 0).astype(int))
```

---

## 逆向變換 (第七章)

### InverseTransform

實現各種正向變換的逆操作。

```python
from ASMbitSpaceMathJIT import InverseTransform, MortonCodec, LogicTransform

# ===== RGB 解包（7.2.1）=====
packed_color = 0xFF8040  # R=255, G=128, B=64
r, g, b = InverseTransform.rgb_unpack(packed_color)
print(f"RGB 解包 0x{packed_color:06X}:")
print(f"  R = {r} (0x{r:02X})")
print(f"  G = {g} (0x{g:02X})")
print(f"  B = {b} (0x{b:02X})")

# ===== 莫頓解碼（7.2.1）=====
x, y = 5, 7
z = MortonCodec.encode(x, y, n=8)
x2, y2 = InverseTransform.morton_decode(z, n=8)
print(f"\n莫頓解碼:")
print(f"  原始座標: ({x}, {y})")
print(f"  莫頓碼: {z}")
print(f"  解碼座標: ({x2}, {y2})")
print(f"  驗證: {(x, y) == (x2, y2)}")

# ===== 字元解碼（7.2.2）=====
codepoints = [72, 101, 108, 108, 111]  # "Hello"
decoded_chars = [InverseTransform.char_decode(cp) for cp in codepoints]
print(f"\n字元解碼:")
print(f"  碼點: {codepoints}")
print(f"  字元: {''.join(decoded_chars)}")

# ===== 布林函數解碼（7.2.3）=====
# 解碼 AND 閘 (真值表 = 8 = 0b1000)
and_func = InverseTransform.boolean_function_decode(8, num_vars=2)
print(f"\n布林函數解碼 (真值表=8):")
for a in [0, 1]:
    for b in [0, 1]:
        result = and_func(a, b)
        print(f"  f({a}, {b}) = {result}")

# 解碼 XOR 閘 (真值表 = 6 = 0b0110)
xor_func = InverseTransform.boolean_function_decode(6, num_vars=2)
print(f"\n布林函數解碼 (真值表=6):")
for a in [0, 1]:
    for b in [0, 1]:
        result = xor_func(a, b)
        print(f"  f({a}, {b}) = {result}")

# ===== 完整的編碼-解碼往返驗證 =====
print("\n=== 完整往返驗證 ===")

# RGB
original_rgb = (200, 150, 100)
packed = (original_rgb[0] << 16) | (original_rgb[1] << 8) | original_rgb[2]
decoded_rgb = InverseTransform.rgb_unpack(packed)
print(f"RGB 往返: {original_rgb} → 0x{packed:06X} → {decoded_rgb}")
assert original_rgb == decoded_rgb

# Morton
original_xy = (123, 456)
morton = MortonCodec.encode(*original_xy, n=16)
decoded_xy = InverseTransform.morton_decode(morton, n=16)
print(f"Morton 往返: {original_xy} → {morton} → {decoded_xy}")
assert original_xy == decoded_xy

# 布林函數
def my_func(a, b, c):
    return (a and b) or (not a and c)

original_tt = LogicTransform.encode_boolean_function(my_func, 3)
decoded_func = InverseTransform.boolean_function_decode(original_tt, 3)
for a in [0, 1]:
    for b in [0, 1]:
        for c in [0, 1]:
            assert my_func(a, b, c) == decoded_func(a, b, c)
print(f"布林函數往返: 真值表 {original_tt:08b} 驗證通過")
```

---

## 視覺化工具

### PNGEncoder

純 Python PNG 編碼器，不依賴外部圖像庫。

```python
from ASMbitSpaceMathJIT import PNGEncoder
import numpy as np
import os

# 確保輸出目錄存在
os.makedirs("./bitSpace", exist_ok=True)

# ===== 灰階影像 =====
gray_img = np.zeros((256, 256), dtype=np.uint8)
for y in range(256):
    for x in range(256):
        gray_img[y, x] = (x + y) % 256  # 漸層

PNGEncoder.encode_grayscale(gray_img, "./bitSpace/demo_gray.png")
print("已儲存: ./bitSpace/demo_gray.png")

# ===== RGB 影像 =====
rgb_img = np.zeros((256, 256, 3), dtype=np.uint8)
rgb_img[:, :, 0] = np.arange(256).reshape(1, 256)  # R 通道
rgb_img[:, :, 1] = np.arange(256).reshape(256, 1)  # G 通道
rgb_img[:, :, 2] = 128  # B 通道固定

PNGEncoder.encode_rgb(rgb_img, "./bitSpace/demo_rgb.png")
print("已儲存: ./bitSpace/demo_rgb.png")

# ===== 調色盤影像 =====
# 創建 4 色調色盤
palette = np.array([
    [0, 0, 0],       # 黑
    [255, 0, 0],     # 紅
    [0, 255, 0],     # 綠
    [0, 0, 255],     # 藍
], dtype=np.uint8)

# 創建索引影像
indexed_img = np.zeros((64, 64), dtype=np.uint8)
indexed_img[:32, :32] = 0  # 左上黑
indexed_img[:32, 32:] = 1  # 右上紅
indexed_img[32:, :32] = 2  # 左下綠
indexed_img[32:, 32:] = 3  # 右下藍

PNGEncoder.encode_palette(indexed_img, palette, "./bitSpace/demo_palette.png")
print("已儲存: ./bitSpace/demo_palette.png")

# ===== 生成謝爾賓斯基三角形 =====
from ASMbitSpaceMathJIT import BitGeometryOps

sierpinski = BitGeometryOps.generate_sierpinski(256) * 255
PNGEncoder.encode_grayscale(sierpinski.astype(np.uint8), "./bitSpace/demo_sierpinski.png")
print("已儲存: ./bitSpace/demo_sierpinski.png")
```

### HyperSpaceVisualizer

高階視覺化工具。

```python
from ASMbitSpaceMathJIT import HyperSpaceVisualizer
import numpy as np

# 創建視覺化器
viz = HyperSpaceVisualizer(output_dir="./bitSpace")

# ===== 謝爾賓斯基三角形 =====
path = viz.visualize_sierpinski(size=256, name="vis_sierpinski")
print(f"謝爾賓斯基三角形: {path}")

# ===== 漢明圓 =====
# 中心為 0，半徑為 3，位元寬度為 8
path = viz.visualize_hamming_circle(center=0, radius=3, n=8, name="vis_hamming_r3")
print(f"漢明圓 (r=3): {path}")

# 不同半徑的漢明圓
for r in [1, 2, 3, 4]:
    path = viz.visualize_hamming_circle(0, r, 8, f"vis_hamming_r{r}")
    print(f"漢明圓 (r={r}): {path}")

# ===== 位元直線 =====
path = viz.visualize_bit_line(start=0, direction=0b00001111, n=8, name="vis_bitline_1")
print(f"位元直線 (d=0x0F): {path}")

path = viz.visualize_bit_line(0, 0b01010101, 8, "vis_bitline_2")
print(f"位元直線 (d=0x55): {path}")

# ===== 莫頓空間 =====
path = viz.visualize_morton_space(n=8, name="vis_morton")
print(f"莫頓編碼空間: {path}")

# ===== 距離場 =====
grid = np.ones((64, 64), dtype=np.uint8)
grid[20:44, 30:34] = 0  # 添加障礙物
path = viz.visualize_distance_field(grid, start=(32, 32), name="vis_distance")
print(f"距離場: {path}")

# ===== XOR 濾波效果 =====
random_img = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
path = viz.visualize_xor_filter(random_img, name="vis_xor_filter")
print(f"XOR 濾波: {path}")

# ===== 位元微積分可視化 =====
# 一維
f_1d = np.random.randint(0, 2, 128, dtype=np.uint8)
path = viz.visualize_bit_calculus(f_1d, name="vis_calculus_1d")
print(f"一維位元微積分: {path}")

# 二維
f_2d = np.random.randint(0, 2, (64, 64), dtype=np.uint8)
path = viz.visualize_bit_calculus(f_2d, name="vis_calculus_2d")
print(f"二維位元微積分 (拉普拉斯): {path}")

# ===== NTT 頻譜 =====
data = np.random.randint(0, 100, 256, dtype=np.int64)
path = viz.visualize_ntt_spectrum(data, name="vis_ntt_spectrum")
print(f"NTT 頻譜: {path}")

# ===== 一維演化歷史 =====
# 模擬細胞自動機
def rule30(state):
    """Rule 30 細胞自動機"""
    n = len(state)
    new_state = np.zeros_like(state)
    for i in range(n):
        left = state[(i - 1) % n]
        center = state[i]
        right = state[(i + 1) % n]
        pattern = (left << 2) | (center << 1) | right
        new_state[i] = (30 >> pattern) & 1
    return new_state

initial = np.zeros(100, dtype=np.uint8)
initial[50] = 1  # 中心起點

history = [initial.copy()]
state = initial.copy()
for _ in range(100):
    state = rule30(state)
    history.append(state.copy())

path = viz.visualize_1d_evolution(history, name="vis_rule30")
print(f"Rule 30 演化: {path}")

# ===== 二維狀態 =====
state_2d = np.random.randint(0, 256, (128, 128), dtype=np.uint8)
path = viz.visualize_2d_state(state_2d, name="vis_random_2d")
print(f"二維狀態: {path}")
```

---

## 測試與性能基準

### testAll

運行完整測試套件。

```python
from ASMbitSpaceMathJIT import testAll

# 運行所有測試
success = testAll()

# testAll() 會:
# 1. 運行所有單元測試
# 2. 執行性能基準測試
# 3. 生成視覺化輸出到 ./bitSpace 目錄
# 4. 返回 True 如果所有測試通過，否則 False

if success:
    print("所有測試通過！")
else:
    print("部分測試失敗")
```

### BitSpacePerformanceBenchmark

性能基準測試類。

```python
from ASMbitSpaceMathJIT import BitSpacePerformanceBenchmark

# ===== 運行所有性能測試 =====
BitSpacePerformanceBenchmark.run_all()

# ===== 或運行個別測試 =====

# 位元度量性能
BitSpacePerformanceBenchmark.bench_metrics()

# 變換性能
BitSpacePerformanceBenchmark.bench_transforms()

# NTT 性能
BitSpacePerformanceBenchmark.bench_ntt()

# 位元微積分性能
BitSpacePerformanceBenchmark.bench_calculus()

# 影像濾波性能
BitSpacePerformanceBenchmark.bench_filters()

# 卷積性能
BitSpacePerformanceBenchmark.bench_convolution()

# 圖演算法性能
BitSpacePerformanceBenchmark.bench_graph()

# 視覺化輸出性能
BitSpacePerformanceBenchmark.bench_visualization()
```

**輸出範例：**
```
============================================================
位元超空間性能基準測試
============================================================

後端信息: {'backend': 'numpy_fallback', 'available': False}
ASM JIT 可用: False

------------------------------------------------------------
性能測試: 位元度量
------------------------------------------------------------
  Hamming distance n=  1000: 45.23 µs
  Popcount n=  1000: 12.34 µs
  Hamming distance n= 10000: 234.56 µs
  Popcount n= 10000: 89.01 µs
  ...

------------------------------------------------------------
性能測試: NTT
------------------------------------------------------------
  NTT forward n=   64: 1.23 ms
  NTT inverse n=   64: 1.34 ms
  NTT forward n=  256: 5.67 ms
  ...
```

---

## 完整應用範例

### 範例 1：影像邊緣檢測

```python
from ASMbitSpaceMathJIT import (
    BitImageFilters, BitCalculus, PNGEncoder, 
    HyperSpaceVisualizer
)
import numpy as np

# 創建測試影像（簡單幾何形狀）
img = np.zeros((128, 128), dtype=np.uint8)
img[20:40, 20:80] = 255       # 矩形
img[50:100, 40:90] = 200      # 另一個矩形
img[60:90, 50:80] = 255       # 重疊區域

# 保存原始影像
PNGEncoder.encode_grayscale(img, "./bitSpace/edge_original.png")

# 方法 1：XOR 濾波器
edges_xor = BitImageFilters.xor_blur(img)
PNGEncoder.encode_grayscale(edges_xor.astype(np.uint8), "./bitSpace/edge_xor.png")

# 方法 2：位元拉普拉斯算子
laplacian = BitCalculus.laplacian_2d(img)
PNGEncoder.encode_grayscale(laplacian.astype(np.uint8), "./bitSpace/edge_laplacian.png")

# 方法 3：梯度幅值
grad_x, grad_y = BitCalculus.gradient_2d(img)
gradient_mag = np.sqrt(grad_x.astype(float)**2 + grad_y.astype(float)**2)
gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
PNGEncoder.encode_grayscale(gradient_mag, "./bitSpace/edge_gradient.png")

print("邊緣檢測結果已保存到 ./bitSpace/")
```

### 範例 2：文字相似度比較

```python
from ASMbitSpaceMathJIT import TextTransform

def compute_text_similarity(text1, text2, gram_size=2):
    """使用 n-gram 簽名計算文字相似度"""
    sig1 = TextTransform.ngram_signature(text1, gram_size)
    sig2 = TextTransform.ngram_signature(text2, gram_size)
    
    # Jaccard 相似度
    intersection = bin(sig1 & sig2).count('1')
    union = bin(sig1 | sig2).count('1')
    
    if union == 0:
        return 0.0
    return intersection / union

# 測試文字
texts = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox leaps over the lazy dog",  # 相似
    "a fast auburn fox hops above the sleepy hound",  # 語義相似
    "completely unrelated text about programming",  # 不相關
]

print("文字相似度比較：")
print("-" * 60)
for i, t1 in enumerate(texts):
    for j, t2 in enumerate(texts):
        if i < j:
            sim = compute_text_similarity(t1, t2)
            print(f"Text {i+1} vs Text {j+1}: {sim:.3f}")
```

### 範例 3：位元圖形生成

```python
from ASMbitSpaceMathJIT import (
    BitGeometryOps, HammingCircle, BitLine,
    PNGEncoder
)
import numpy as np

# 生成組合圖案
size = 256
canvas = np.zeros((size, size), dtype=np.uint8)

# 添加謝爾賓斯基三角形
sierpinski = BitGeometryOps.generate_sierpinski(size)
canvas = np.bitwise_or(canvas, sierpinski * 50)

# 添加漢明圓（放大顯示）
for center in [0, 85, 170, 255]:
    for radius in range(1, 5):
        try:
            circle = HammingCircle(center, radius, 8)
            for point in circle.enumerate():
                y = (point // 16) * 16 + 8
                x = (point % 16) * 16 + 8
                if 0 <= y < size and 0 <= x < size:
                    canvas[y:y+2, x:x+2] = 200
        except:
            pass

# 保存
PNGEncoder.encode_grayscale(canvas, "./bitSpace/combined_pattern.png")
print("組合圖案已保存到 ./bitSpace/combined_pattern.png")
```

### 範例 4：快速多項式乘法

```python
from ASMbitSpaceMathJIT import NTT
import numpy as np

def fast_polynomial_multiply(coeffs1, coeffs2):
    """使用 NTT 進行快速多項式乘法"""
    n1, n2 = len(coeffs1), len(coeffs2)
    result_len = n1 + n2 - 1
    
    # 找到足夠大的 2 的冪次
    size = 1
    while size < result_len:
        size *= 2
    
    # 零填充
    p1 = np.zeros(size, dtype=np.int64)
    p2 = np.zeros(size, dtype=np.int64)
    p1[:n1] = coeffs1
    p2[:n2] = coeffs2
    
    # NTT 變換與逐點乘法
    ntt = NTT(16)
    P1 = ntt.forward(p1)
    P2 = ntt.forward(p2)
    C = (P1 * P2) % ntt.p
    result = ntt.inverse(C)
    
    return result[:result_len]

# 範例：計算 (1 + 2x + 3x² + 4x³) × (5 + 6x + 7x² + 8x³)
poly1 = np.array([1, 2, 3, 4], dtype=np.int64)
poly2 = np.array([5, 6, 7, 8], dtype=np.int64)

result = fast_polynomial_multiply(poly1, poly2)
print(f"多項式 1: {list(poly1)}")
print(f"多項式 2: {list(poly2)}")
print(f"乘積係數: {list(result)}")

# 驗證（對比樸素 O(n²) 算法）
def naive_multiply(p1, p2):
    n = len(p1) + len(p2) - 1
    result = np.zeros(n, dtype=np.int64)
    for i in range(len(p1)):
        for j in range(len(p2)):
            result[i + j] += p1[i] * p2[j]
    return result

expected = naive_multiply(poly1, poly2)
print(f"樸素算法: {list(expected)}")
print(f"結果驗證: {np.array_equal(result, expected)}")
```

### 範例 5：數位邏輯電路模擬

```python
from ASMbitSpaceMathJIT import LogicTransform, BooleanRing

class BitCircuit:
    """簡單的位元邏輯電路模擬器"""
    
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.gates = []
    
    def add_gate(self, gate_func, *input_indices):
        """添加邏輯閘"""
        self.gates.append((gate_func, input_indices))
    
    def evaluate(self, inputs):
        """評估電路"""
        values = list(inputs)
        for gate_func, indices in self.gates:
            gate_inputs = [values[i] for i in indices]
            result = gate_func(*gate_inputs)
            values.append(result)
        return values[-1]
    
    def to_truth_table(self):
        """生成完整真值表"""
        return LogicTransform.encode_boolean_function(
            lambda *args: self.evaluate(args),
            self.num_vars
        )

# 建立半加器電路
circuit = BitCircuit(2)
# Sum = A XOR B
circuit.add_gate(lambda a, b: a != b, 0, 1)
# Carry = A AND B  
circuit.add_gate(lambda a, b: a and b, 0, 1)

# 測試
print("半加器真值表：")
print("A B | Sum Carry")
print("-" * 15)
for a in [0, 1]:
    for b in [0, 1]:
        sum_val = (a ^ b)  # XOR
        carry = (a & b)    # AND
        print(f"{a} {b} |  {sum_val}    {carry}")

# 生成 Sum 輸出的真值表編碼
sum_circuit = BitCircuit(2)
sum_circuit.add_gate(lambda a, b: a != b, 0, 1)
sum_tt = sum_circuit.to_truth_table()
print(f"\nSum 函數真值表編碼: {sum_tt:04b} ({sum_tt})")
```

---

## 附錄：API 快速參考

### 核心類別摘要

| 類別 | 功能 | 主要方法 |
|------|------|----------|
| `BitSpaceConstants` | 常量定義 | `get_mask()`, `get_modulus()` |
| `TorusArithmetic` | 環面算術 | `add()`, `sub()`, `mul()`, `neg()` |
| `HyperSpace1D/2D/ND` | 超空間定義 | `size()`, `dimension()`, `wrap()`, `iterate()` |
| `BitMetrics` | 距離度量 | `xor_distance()`, `hamming_distance_int()`, `popcount()` |
| `HammingCircle/Ball` | 漢明幾何 | `size()`, `contains()`, `enumerate()` |
| `BitLine` | 位元直線 | `size()`, `contains()`, `enumerate()` |
| `TorusRing` | 環結構 | `add()`, `mul()`, `zero()`, `one()` |
| `BooleanRing` | 布林環 | `add()` (XOR), `mul()` (AND) |
| `BitCalculus` | 位元微積分 | `derivative_1d()`, `laplacian_2d()`, `integral_1d()` |
| `NTT` | 數論變換 | `forward()`, `inverse()` |
| `WHT` | 沃爾什變換 | `transform()`, `xor_convolution()` |
| `MortonCodec` | 莫頓編碼 | `encode()`, `decode()` |
| `ImageTransform` | 影像變換 | `encode_rgb()`, `decode_rgb()` |
| `TextTransform` | 文字變換 | `encode_string()`, `rolling_hash()` |
| `LogicTransform` | 邏輯變換 | `encode_boolean_function()`, `decode_boolean_function()` |
| `LookupTableEngine` | 查找表 | `create_table()`, `lookup()` |
| `BitConvolution` | 卷積運算 | `xor_convolution()`, `and_convolution()` |
| `BitGraphAlgorithms` | 圖演算法 | `bit_bfs()`, `distance_field_2d()` |
| `BitImageFilters` | 影像濾波 | `xor_blur()`, `dilate()`, `erode()` |
| `PNGEncoder` | PNG 編碼 | `encode_grayscale()`, `encode_rgb()` |
| `HyperSpaceVisualizer` | 視覺化 | `visualize_sierpinski()`, `visualize_hamming_circle()` |

### 環境變數

| 變數 | 說明 | 預設值 |
|------|------|--------|
| `BITSPACE_VERBOSE` | 詳細輸出模式 | `"0"` |

### 輸出目錄

所有視覺化輸出預設保存至 `./bitSpace` 目錄，可透過 `BitSpaceConfig.OUTPUT_DIR` 修改。