# ASMdynamicGradio 動態知識與工具體系 — 完整使用說明

## 目錄

1. [系統概述](#1-系統概述)
2. [安裝與環境設置](#2-安裝與環境設置)
3. [核心概念](#3-核心概念)
4. [快速入門](#4-快速入門)
5. [高效率內容管理](#5-高效率內容管理)
6. [動態知識整合與查詢](#6-動態知識整合與查詢)
7. [動態執行開發](#7-動態執行開發)
8. [高效率查詢與整合](#8-高效率查詢與整合)
9. [能力分發與經驗回收](#9-能力分發與經驗回收)
10. [進階應用場景](#10-進階應用場景)
11. [最佳實踐與性能優化](#11-最佳實踐與性能優化)
12. [完整 API 參考](#12-完整-api-參考)

---

## 1. 系統概述

ASMdynamicGradio 是一個動態知識與工具體系，提供以下核心能力：

| 功能模組 | 說明 |
|---------|------|
| **動態代碼管理** | 保存、載入、執行 Python 代碼，支持熱更新 |
| **知識庫管理** | Markdown 格式的知識條目，支持標籤和附件 |
| **資料組管理** | 將多種類型內容（文字、圖片、代碼、HTML）組織為統一資料組 |
| **Embedding 校準** | 將不同類型內容映射到同一語義空間，實現跨類型關聯搜索 |
| **命名空間隔離** | 支持多項目、多環境的完全隔離 |
| **蒙特卡羅搜索** | 具有探索性的智能搜索，每次搜索都有驚喜 |
| **演化可視化** | 細胞自動機演化與可視化輸出 |

---

## 2. 安裝與環境設置

### 2.1 依賴安裝

```bash
# 核心依賴
pip install numpy

# Web 界面（可選）
pip install gradio

# 圖像處理（可選）
pip install pillow
```

### 2.2 必要文件

確保以下文件在同一目錄：

```
project/
├── ASMdynamicGradio.py      # 主應用文件
├── ASMsuperDynamicSystem.py # 核心系統模組
├── ASMFileDict3.py          # 存儲引擎（可選）
└── dynamic_app_data/        # 數據存儲目錄（自動創建）
```

### 2.3 啟動方式

```bash
# 啟動 Gradio Web 界面
python ASMdynamicGradio.py

# 指定端口和共享
python ASMdynamicGradio.py --port 8080 --share

# 運行測試套件
python ASMdynamicGradio.py --test

# 指定存儲目錄
python ASMdynamicGradio.py --storage-dir ./my_data --namespace my_project
```

---

## 3. 核心概念

### 3.1 命名空間 (Namespace)

命名空間是邏輯隔離的容器，用於組織不同項目或模組的內容。

```
root/
├── default/           # 默認命名空間
├── project_a/         # 項目 A
│   ├── utils/        # 子命名空間
│   └── models/
└── project_b/         # 項目 B
```

### 3.2 節點類型

| 節點類型 | 說明 | 用途 |
|---------|------|------|
| `code` | Python 代碼 | 可執行函數、類、模組 |
| `data` | 結構化數據 | JSON、字典、列表等 |
| `knowledge` | 知識條目 | Markdown 文檔、教程、筆記 |
| `file` | 二進制文件 | 圖片、視頻、PDF 等 |
| `data_group` | 資料組 | 多類型內容的組合 |

### 3.3 Embedding 校準

系統使用 HDC（Hyperdimensional Computing）向量將不同類型的內容編碼到統一的語義空間：

```
文字內容  ─┐
代碼內容  ─┼─→ HDC 編碼 ─→ 統一向量空間 ─→ 語義相似度計算
圖片元數據 ─┘
```

---

## 4. 快速入門

### 4.1 基本使用模式

```python
from ASMdynamicGradio import DynamicApp

# 創建應用實例
app = DynamicApp(
    storage_dir="./my_knowledge_base",
    namespace="default",
    auto_load=True
)

# 保存代碼
app.saveCode("hello", """
def main():
    return "Hello, World!"
""")

# 執行代碼
result = app.run("hello")
print(result.result)  # 輸出: Hello, World!

# 保存知識
app.saveKnowledge(
    "python_basics",
    "# Python 基礎\n\nPython 是一種高級編程語言...",
    tags=["python", "教程", "入門"]
)

# 搜索
results = app.search("python", mode="fuzzy")
for r in results:
    print(f"{r.name}: {r.score:.2%}")

# 關閉應用
app.close()
```

### 4.2 使用上下文管理器

```python
from ASMdynamicGradio import DynamicApp

with DynamicApp("./data") as app:
    app.saveCode("test", "def main(): return 42")
    result = app.run("test")
    print(result.result)
# 自動關閉，資源自動釋放
```

---

## 5. 高效率內容管理

### 5.1 代碼管理

#### 保存和更新代碼

```python
# 保存新代碼
app.saveCode("utils/string_helper", """
def reverse_string(s: str) -> str:
    '''反轉字符串'''
    return s[::-1]

def capitalize_words(s: str) -> str:
    '''每個單詞首字母大寫'''
    return ' '.join(word.capitalize() for word in s.split())

def main(text: str = "hello world"):
    return {
        "original": text,
        "reversed": reverse_string(text),
        "capitalized": capitalize_words(text)
    }
""", namespace="utils")

# 獲取代碼
code = app.getCode("string_helper", namespace="utils")

# 更新代碼（保留元數據）
app.updateCode("string_helper", new_code, namespace="utils")

# 刪除代碼
app.deleteCode("string_helper", namespace="utils")
```

#### 代碼組織最佳實踐

```python
# 按功能組織命名空間
app.createNamespace("core", description="核心功能模組")
app.createNamespace("utils", description="工具函數", parent="core")
app.createNamespace("models", description="數據模型", parent="core")
app.createNamespace("api", description="API 接口", parent="core")

# 保存代碼到對應命名空間
app.saveCode("base_model", model_code, namespace="models")
app.saveCode("validation", validation_code, namespace="utils")
app.saveCode("endpoints", api_code, namespace="api")
```

### 5.2 知識管理

#### 創建和組織知識條目

```python
# 創建知識條目
app.saveKnowledge(
    name="machine_learning_intro",
    content="""
# 機器學習入門指南

## 什麼是機器學習？

機器學習是人工智能的一個分支，它使計算機能夠從數據中學習，
而無需被明確編程。

## 主要類型

1. **監督學習** - 使用標記數據進行訓練
2. **非監督學習** - 發現數據中的隱藏模式
3. **強化學習** - 通過試錯來學習

## 常見算法

- 線性回歸
- 決策樹
- 神經網絡
- 支持向量機
""",
    namespace="knowledge",
    tags=["ML", "AI", "教程", "入門"],
    metadata={"author": "教學團隊", "difficulty": "beginner"}
)

# 更新知識條目
app.updateKnowledge(
    "machine_learning_intro",
    updated_content,
    tags=["ML", "AI", "教程", "入門", "更新"]
)

# 獲取知識條目
knowledge = app.getKnowledge("machine_learning_intro")
print(knowledge["content"])
print(knowledge["tags"])
```

### 5.3 資料組管理

資料組是將多種類型內容組織在一起的強大功能：

```python
# 創建資料組 - 組合多種內容類型
data_group = app.createDataGroup(
    name="python_tutorial_package",
    items=[
        # 文字說明
        ("Python 是一種高級編程語言，以其簡潔和可讀性著稱。", "text", {"section": "intro"}),
        
        # 代碼示例
        ("""
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
""", "code", {"language": "python", "level": "beginner"}),
        
        # Markdown 文檔
        ("## 變量和數據類型\n\nPython 支持多種數據類型...", "markdown", {"topic": "variables"}),
        
        # HTML 內容
        ("<div class='example'><code>x = 10</code></div>", "html", {"type": "interactive"}),
        
        # JSON 數據
        ('{"chapter": 1, "exercises": ["ex1", "ex2"]}', "json", {"metadata": True})
    ],
    namespace="tutorials",
    description="Python 入門教程完整資料包",
    tags=["python", "教程", "入門", "完整"],
    calibrate=True  # 啟用 Embedding 校準
)

# 查看資料組
print(f"資料組: {data_group.name}")
print(f"項目數: {len(data_group.items)}")
print(f"已校準: {data_group.calibrated}")

# 向現有資料組添加內容
app.addToDataGroup(
    "python_tutorial_package",
    content="# 進階主題\n\n## 裝飾器...",
    item_type="markdown",
    metadata={"topic": "advanced"},
    namespace="tutorials",
    recalibrate=True  # 重新校準以保持語義一致性
)

# 列出所有資料組
groups = app.listDataGroups(namespace="tutorials")
print(f"教程資料組: {groups}")
```

### 5.4 文件管理

```python
# 上傳圖片
with open("diagram.png", "rb") as f:
    image_data = f.read()

app.addFile(
    "architecture_diagram.png",
    image_data,
    namespace="docs",
    mime_type="image/png",
    metadata={"description": "系統架構圖", "version": "1.0"}
)

# 獲取文件
file_data = app.getFile("architecture_diagram.png", namespace="docs")

# 獲取文件信息
file_info = app.getFileInfo("architecture_diagram.png", namespace="docs")
print(f"文件大小: {file_info.size} bytes")
print(f"MIME 類型: {file_info.mime_type}")

# 刪除文件
app.deleteFile("architecture_diagram.png", namespace="docs")
```

---

## 6. 動態知識整合與查詢

### 6.1 Embedding 校準系統

Embedding 校準是實現跨類型語義搜索的關鍵：

```python
# 批量導入並校準
items = [
    ("data_processor", """
def process_data(data):
    '''處理數據的主函數'''
    cleaned = clean_data(data)
    transformed = transform_data(cleaned)
    return transformed
""", "code", {"category": "data"}),
    
    ("data_processing_guide", """
# 數據處理指南

本指南介紹如何使用 process_data 函數進行數據處理...
""", "text", {"category": "documentation"}),
    
    ("data_schema", {
        "input_format": "json",
        "output_format": "json",
        "fields": ["id", "name", "value"]
    }, "data", {"category": "schema"})
]

# 帶校準的批量導入
nodes = app.batchImportWithCalibration(
    items,
    namespace="data_processing",
    as_data_group=False
)
print(f"導入了 {len(nodes)} 個節點（已校準）")

# 作為資料組導入（更緊密的關聯）
data_group = app.batchImportWithCalibration(
    items,
    namespace="data_processing",
    as_data_group=True,
    group_name="data_processing_kit"
)
print(f"資料組 {data_group.name} 已創建，包含 {len(data_group.items)} 項")
```

### 6.2 命名空間校準

對整個命名空間進行統一校準：

```python
# 校準命名空間中的所有內容
result = app.calibrateNamespace("my_project")
print(f"校準了 {result['calibrated']} 個節點")

# 這使得命名空間內的所有內容都可以進行語義比較
# 例如：代碼和文檔之間的關聯搜索
```

### 6.3 跨類型關聯搜索

```python
# 關聯搜索 - 跨越代碼、文檔、資料組
results = app.searchRelated(
    query="數據處理",
    content_types=["code", "knowledge", "data_group"],
    namespace="my_project",
    threshold=0.5,
    limit=20
)

for r in results:
    print(f"\n[{r.content_type}] {r.name}")
    print(f"  相似度: {r.score:.2%}")
    print(f"  預覽: {r.preview[:100]}...")
    
    if r.related_items:
        print(f"  關聯項目: {', '.join(r.related_items)}")
```

---

## 7. 動態執行開發

### 7.1 代碼執行模式

```python
# 基本執行
app.saveCode("calculator", """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def main(operation="add", x=0, y=0):
    if operation == "add":
        return add(x, y)
    elif operation == "multiply":
        return multiply(x, y)
    else:
        raise ValueError(f"Unknown operation: {operation}")
""")

# 執行並傳遞參數
result = app.run(
    "calculator",
    entry_point="main",
    kwargs={"operation": "multiply", "x": 5, "y": 3}
)

if result.success:
    print(f"結果: {result.result}")  # 輸出: 15
    print(f"執行時間: {result.execution_time_ms:.2f} ms")
else:
    print(f"錯誤: {result.error}")
    print(f"堆棧: {result.stderr}")
```

### 7.2 動態模組導入

```python
# 保存可重用模組
app.saveCode("math_utils", """
import math

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True
""", namespace="utils")

# 動態導入為模組
math_utils = app.importCode("math_utils", namespace="utils")

# 使用導入的模組
print(math_utils.factorial(5))     # 120
print(math_utils.fibonacci(10))    # 55
print(math_utils.is_prime(17))     # True

# 在其他代碼中使用
app.saveCode("main_program", """
def main():
    # 這裡可以使用 math_utils
    from dynamic_modules import math_utils
    return {
        "factorial_10": math_utils.factorial(10),
        "fib_20": math_utils.fibonacci(20),
        "is_97_prime": math_utils.is_prime(97)
    }
""")
```

### 7.3 錯誤處理與調試

```python
# 執行可能出錯的代碼
result = app.run("risky_code")

if not result.success:
    # 詳細錯誤信息
    print(f"錯誤類型: {result.error}")
    print(f"標準錯誤輸出:\n{result.stderr}")
    
    # 記錄到調試日誌
    app._log_debug(
        "ERROR",
        f"代碼執行失敗: risky_code",
        details={"error": result.error},
        exc_info=True
    )

# 查看調試日誌
logs = app.get_debug_log(limit=50)
for log in logs:
    print(f"[{log['level']}] {log['timestamp']}: {log['message']}")

# 清空調試日誌
app.clear_debug_log()
```

### 7.4 執行結果處理

```python
# 執行返回複雜結果的代碼
app.saveCode("data_analysis", """
import numpy as np

def main(data_size=100):
    # 生成模擬數據
    data = np.random.randn(data_size)
    
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "percentiles": {
            "25": float(np.percentile(data, 25)),
            "50": float(np.percentile(data, 50)),
            "75": float(np.percentile(data, 75))
        },
        "sample_size": data_size
    }
""")

result = app.run("data_analysis", kwargs={"data_size": 1000})

if result.success:
    analysis = result.result
    print(f"平均值: {analysis['mean']:.4f}")
    print(f"標準差: {analysis['std']:.4f}")
    print(f"中位數: {analysis['percentiles']['50']:.4f}")
```

---

## 8. 高效率查詢與整合

### 8.1 多模式搜索

```python
# 1. 精確匹配
results = app.search("def process_data", mode="exact")

# 2. 模糊匹配（默認）
results = app.search("數據處理", mode="fuzzy", similarity_threshold=0.3)

# 3. 正則表達式
results = app.search(r"def\s+\w+_handler\(", mode="regex")

# 4. 語義搜索
results = app.search("如何處理 JSON 數據", mode="semantic", similarity_threshold=0.5)

# 5. 蒙特卡羅搜索（帶探索性）
results = app.search(
    "機器學習算法",
    mode="monte_carlo",
    monte_carlo_samples=100,
    similarity_threshold=0.3,
    result_limit=20
)

# 6. 關聯搜索
results = app.search("python 教程", mode="related", similarity_threshold=0.4)
```

### 8.2 高級搜索選項

```python
# 完整搜索配置
results = app.search(
    query="深度學習模型訓練",
    mode="monte_carlo",
    content_type="all",  # all, code, data, knowledge, file, data_group
    namespace=None,      # None 表示搜索所有命名空間
    similarity_threshold=0.3,
    monte_carlo_samples=150,
    fast_match_limit=2000,  # 快速匹配階段的候選數量限制
    result_limit=30,
    include_content=True,
    include_data_groups=True
)

# 處理結果
for r in results:
    print(f"""
節點 ID: {r.node_id}
名稱: {r.name}
命名空間: {r.namespace}
類型: {r.content_type}
相似度: {r.score:.2%}
預覽: {r.preview[:150]}...
關聯項目: {r.related_items}
元數據: {r.metadata}
""")
```

### 8.3 搜索結果整合

```python
def integrated_search(app, query, namespaces=None):
    """整合搜索 - 從多個來源收集和排序結果"""
    
    all_results = []
    
    # 從多個命名空間搜索
    target_namespaces = namespaces or app.listNamespaces()
    
    for ns in target_namespaces:
        # 語義搜索
        semantic_results = app.search(
            query, mode="semantic", 
            namespace=ns, 
            similarity_threshold=0.4
        )
        
        # 模糊搜索
        fuzzy_results = app.search(
            query, mode="fuzzy",
            namespace=ns,
            similarity_threshold=0.3
        )
        
        all_results.extend(semantic_results)
        all_results.extend(fuzzy_results)
    
    # 去重（按 node_id）
    seen = set()
    unique_results = []
    for r in all_results:
        if r.node_id not in seen:
            seen.add(r.node_id)
            unique_results.append(r)
    
    # 按分數排序
    unique_results.sort(key=lambda x: x.score, reverse=True)
    
    return unique_results[:50]

# 使用
results = integrated_search(app, "數據處理流程")
```

### 8.4 節點列表與過濾

```python
# 列出所有節點
all_nodes = app.listNodes()

# 按類型過濾
code_nodes = app.listNodes(content_type="code")
knowledge_nodes = app.listNodes(content_type="knowledge")
data_groups = app.listNodes(content_type="data_group")

# 按命名空間過濾
project_nodes = app.listNodes(namespace="my_project")

# 組合過濾
project_code = app.listNodes(content_type="code", namespace="my_project")

# 獲取節點詳細信息
for node in project_code[:5]:
    detail = app.getNode(node["name"], node["namespace"])
    print(f"""
名稱: {detail['name']}
類型: {detail['type']}
創建時間: {detail['created']}
修改時間: {detail['modified']}
內容預覽: {str(detail['content'])[:200]}...
""")
```

---

## 9. 能力分發與經驗回收

### 9.1 命名空間複製（能力分發）

將成熟的功能複製到新項目：

```python
# 場景：將核心工具庫分發到多個項目

# 1. 創建核心工具庫
app.createNamespace("core_tools", description="核心工具庫 v1.0")
app.saveCode("logger", logger_code, namespace="core_tools")
app.saveCode("config_parser", config_code, namespace="core_tools")
app.saveCode("data_validator", validator_code, namespace="core_tools")
app.createDataGroup("tool_docs", doc_items, namespace="core_tools")

# 2. 分發到項目 A
result = app.copyNamespace(
    source="core_tools",
    target="project_a_tools",
    include_codes=True,
    include_data=True,
    include_groups=True
)
print(f"分發到項目 A: 代碼 {result['codes']}, 數據 {result['data']}, 資料組 {result['groups']}")

# 3. 分發到項目 B（可選擇性分發）
result = app.copyNamespace(
    source="core_tools",
    target="project_b_tools",
    include_codes=True,
    include_data=False,  # 不包含數據
    include_groups=False  # 不包含資料組
)

# 4. 項目可以獨立修改各自的副本
app.updateCode("logger", custom_logger_code, namespace="project_a_tools")
```

### 9.2 導出與備份（經驗回收）

```python
# 導出到文件夾
export_count = app.toFolder(
    folder_path="./exported_project",
    namespace="my_project",
    include_codes=True,
    include_data=True
)
print(f"導出了 {export_count} 個節點")

# 導出目錄結構
# exported_project/
# ├── codes/
# │   ├── module1.py
# │   ├── module2.py
# │   └── utils/
# │       └── helpers.py
# ├── data/
# │   ├── config.json
# │   └── schema.yaml
# └── knowledge/
#     ├── readme.md
#     └── api_docs.md

# 從文件夾導入（經驗回收）
nodes = app.fromFolder(
    folder_path="./external_project",
    namespace="imported_project",
    recursive=True,
    calibrate=True,  # 導入時進行校準
    file_patterns=["*.py", "*.json", "*.md"]
)
print(f"導入了 {len(nodes)} 個節點")
```

### 9.3 跨系統遷移

```python
from ASMFileDict3 import FileDict

# 導出到 FileDict（可攜式存儲）
target_storage = FileDict("./portable_backup.db")
count = app.toFileDict(target_storage, namespace="production")
print(f"導出到 FileDict: {count} 個節點")
target_storage.close()

# 從 FileDict 導入
source_storage = FileDict("./external_backup.db")
nodes = app.fromFileDict(source_storage, namespace="imported")
print(f"從 FileDict 導入: {len(nodes)} 個節點")
source_storage.close()
```

### 9.4 版本管理與回滾

```python
# 創建版本快照
def create_snapshot(app, namespace, version):
    """創建命名空間的版本快照"""
    snapshot_ns = f"{namespace}_v{version}"
    
    result = app.copyNamespace(
        source=namespace,
        target=snapshot_ns,
        include_codes=True,
        include_data=True,
        include_groups=True
    )
    
    # 記錄版本信息
    app.saveData(
        "version_info",
        {
            "version": version,
            "source": namespace,
            "created": datetime.now().isoformat(),
            "stats": result
        },
        namespace=snapshot_ns
    )
    
    return snapshot_ns

# 回滾到特定版本
def rollback_to_version(app, namespace, version):
    """回滾命名空間到特定版本"""
    snapshot_ns = f"{namespace}_v{version}"
    
    # 刪除當前內容
    app.deleteNamespace(namespace, force=True)
    
    # 從快照恢復
    app.renameNamespace(snapshot_ns, namespace)
    
    return True

# 使用示例
create_snapshot(app, "production", "1.0")
# ... 進行開發 ...
create_snapshot(app, "production", "1.1")
# ... 發現問題 ...
rollback_to_version(app, "production", "1.0")
```

---

## 10. 進階應用場景

### 10.1 知識庫構建與查詢系統

```python
class KnowledgeBase:
    """知識庫管理類"""
    
    def __init__(self, app, namespace="knowledge_base"):
        self.app = app
        self.namespace = namespace
        app.createNamespace(namespace, description="知識庫系統")
    
    def add_article(self, title, content, category, tags=None):
        """添加文章"""
        name = self._normalize_name(title)
        
        self.app.saveKnowledge(
            name=name,
            content=content,
            namespace=self.namespace,
            tags=tags or [],
            metadata={
                "title": title,
                "category": category,
                "created": datetime.now().isoformat()
            }
        )
        
        return name
    
    def add_tutorial(self, name, sections):
        """添加教程（多部分內容）"""
        items = []
        for i, section in enumerate(sections):
            items.append((
                section["content"],
                section.get("type", "markdown"),
                {"section": i, "title": section.get("title", f"Section {i+1}")}
            ))
        
        return self.app.createDataGroup(
            name=name,
            items=items,
            namespace=self.namespace,
            description=f"教程: {name}",
            tags=["tutorial"],
            calibrate=True
        )
    
    def search(self, query, category=None):
        """搜索知識庫"""
        results = self.app.search(
            query,
            mode="semantic",
            namespace=self.namespace,
            similarity_threshold=0.4,
            result_limit=20
        )
        
        if category:
            results = [
                r for r in results
                if r.metadata.get("category") == category
            ]
        
        return results
    
    def get_related_articles(self, article_name):
        """獲取相關文章"""
        article = self.app.getKnowledge(article_name, self.namespace)
        if not article:
            return []
        
        content = article.get("content", "")
        results = self.app.searchRelated(
            content[:500],
            content_types=["knowledge", "data_group"],
            namespace=self.namespace,
            threshold=0.5
        )
        
        # 排除自身
        return [r for r in results if r.name != article_name]
    
    def _normalize_name(self, title):
        return title.lower().replace(" ", "_").replace("/", "_")

# 使用示例
kb = KnowledgeBase(app)

# 添加文章
kb.add_article(
    "Python 列表推導式",
    "# 列表推導式\n\n列表推導式是 Python 中創建列表的簡潔方式...",
    category="python",
    tags=["python", "列表", "技巧"]
)

# 添加教程
kb.add_tutorial("Python 入門", [
    {"title": "安裝 Python", "content": "# 安裝指南\n...", "type": "markdown"},
    {"title": "第一個程序", "content": "print('Hello')", "type": "code"},
    {"title": "變量", "content": "# 變量\n...", "type": "markdown"}
])

# 搜索
results = kb.search("列表操作", category="python")
related = kb.get_related_articles("python_列表推導式")
```

### 10.2 代碼模板系統

```python
class CodeTemplateSystem:
    """代碼模板系統"""
    
    def __init__(self, app, namespace="templates"):
        self.app = app
        self.namespace = namespace
        app.createNamespace(namespace, description="代碼模板庫")
    
    def add_template(self, name, template_code, variables, description=""):
        """添加代碼模板"""
        self.app.saveCode(
            name=name,
            code=template_code,
            namespace=self.namespace,
            metadata={
                "type": "template",
                "variables": variables,
                "description": description
            }
        )
    
    def generate_code(self, template_name, **variables):
        """根據模板生成代碼"""
        template = self.app.getCode(template_name, self.namespace)
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # 簡單的變量替換
        code = template
        for key, value in variables.items():
            code = code.replace(f"${{{key}}}", str(value))
            code = code.replace(f"$${key}$$", str(value))
        
        return code
    
    def save_generated(self, name, template_name, namespace=None, **variables):
        """生成並保存代碼"""
        code = self.generate_code(template_name, **variables)
        ns = namespace or "generated"
        
        self.app.saveCode(name, code, ns, metadata={
            "generated_from": template_name,
            "variables": variables
        })
        
        return code

# 使用示例
templates = CodeTemplateSystem(app)

# 添加 CRUD 模板
templates.add_template(
    "crud_handler",
    """
class $${model_name}$$Handler:
    '''$${description}$$'''
    
    def __init__(self, db):
        self.db = db
        self.collection = "$${collection}$$"
    
    def create(self, data):
        return self.db[self.collection].insert_one(data)
    
    def read(self, id):
        return self.db[self.collection].find_one({"_id": id})
    
    def update(self, id, data):
        return self.db[self.collection].update_one({"_id": id}, {"$set": data})
    
    def delete(self, id):
        return self.db[self.collection].delete_one({"_id": id})
""",
    variables=["model_name", "description", "collection"],
    description="CRUD 處理器模板"
)

# 生成代碼
templates.save_generated(
    "user_handler",
    "crud_handler",
    namespace="handlers",
    model_name="User",
    description="用戶數據處理器",
    collection="users"
)
```

### 10.3 演化可視化系統

```python
# 初始化演化
app.initEvolution(mode="random")  # random, center, gradient, noise

# 執行演化
frames = app.evolve(
    steps=200,
    rule="diffusion",  # diffusion, conway, wave, growth, erosion
    record_interval=10
)

print(f"生成了 {len(frames)} 幀")

# 處理演化幀
for i, frame in enumerate(frames):
    state = frame.state  # numpy array (128x128)
    metrics = frame.metrics
    
    print(f"幀 {i}: mean={metrics['mean']:.4f}, std={metrics['std']:.4f}")
    
    # 可以將 state 轉換為圖像
    # rgb = (state * 255).astype(np.uint8)

# 獲取當前演化狀態
current_state = app.getEvolutionState()

# 保存可視化輸出
outputs = app.saveVisualization(prefix="evolution", effect="glow")
print(f"保存的文件: {outputs}")
```

---

## 11. 最佳實踐與性能優化

### 11.1 命名空間組織策略

```
project/
├── core/                    # 核心功能
│   ├── utils/              # 通用工具
│   ├── models/             # 數據模型
│   └── services/           # 服務層
├── features/               # 功能模組
│   ├── user_management/
│   ├── data_processing/
│   └── reporting/
├── knowledge/              # 知識庫
│   ├── docs/               # 文檔
│   ├── tutorials/          # 教程
│   └── faqs/               # 常見問題
├── templates/              # 代碼模板
├── tests/                  # 測試代碼
└── snapshots/              # 版本快照
    ├── v1.0/
    └── v1.1/
```

### 11.2 內容組織建議

```python
# 1. 使用一致的命名規範
# 代碼：snake_case
app.saveCode("user_authentication", code)
app.saveCode("data_processor", code)

# 知識：描述性名稱
app.saveKnowledge("getting_started_guide", content)
app.saveKnowledge("api_reference_v2", content)

# 資料組：功能性名稱
app.createDataGroup("ml_training_dataset", items)
app.createDataGroup("user_onboarding_content", items)

# 2. 使用標籤進行分類
app.saveKnowledge(
    "python_async_guide",
    content,
    tags=["python", "async", "advanced", "v3.8+"]
)

# 3. 使用元數據記錄上下文
app.saveCode(
    "payment_processor",
    code,
    metadata={
        "author": "team_a",
        "version": "2.1",
        "dependencies": ["stripe", "requests"],
        "last_reviewed": "2024-01-15"
    }
)
```

### 11.3 性能優化技巧

```python
# 1. 批量操作優先於單個操作
# 不推薦
for item in items:
    app.saveCode(item["name"], item["code"])

# 推薦
app.batchImportWithCalibration([
    (item["name"], item["code"], "code", None)
    for item in items
])

# 2. 合理使用校準
# 對於需要語義搜索的內容，啟用校準
app.createDataGroup("searchable_docs", items, calibrate=True)

# 對於不需要搜索的內容，跳過校準
app.createDataGroup("raw_data", items, calibrate=False)

# 3. 限制搜索範圍
# 不推薦：搜索所有內容
results = app.search("query")

# 推薦：指定命名空間和類型
results = app.search(
    "query",
    namespace="my_project",
    content_type="code",
    fast_match_limit=500
)

# 4. 使用上下文管理器確保資源釋放
with DynamicApp("./data") as app:
    # 操作
    pass
# 自動釋放資源

# 5. 定期清理調試日誌
if len(app.get_debug_log()) > 500:
    app.clear_debug_log()
```

### 11.4 錯誤處理模式

```python
def safe_execute(app, code_name, namespace=None, **kwargs):
    """安全執行代碼，包含完整錯誤處理"""
    try:
        # 檢查代碼是否存在
        code = app.getCode(code_name, namespace)
        if code is None:
            return {
                "success": False,
                "error": f"Code not found: {code_name}",
                "result": None
            }
        
        # 執行代碼
        result = app.run(code_name, namespace, **kwargs)
        
        if result.success:
            return {
                "success": True,
                "result": result.result,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "stderr": result.stderr,
                "result": None
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}",
            "result": None
        }

# 使用
result = safe_execute(app, "my_function", namespace="utils", x=10)
if result["success"]:
    print(f"Result: {result['result']}")
else:
    print(f"Error: {result['error']}")
```

---

## 12. 完整 API 參考

### 12.1 DynamicApp 類

#### 初始化

```python
app = DynamicApp(
    storage_dir: str = "./dynamic_app_data",  # 存儲目錄
    namespace: str = "default",               # 默認命名空間
    auto_load: bool = True                    # 是否自動加載現有數據
)
```

#### 代碼管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `saveCode(name, code, namespace, metadata)` | 保存代碼 | `DynamicNode` |
| `getCode(name, namespace)` | 獲取代碼 | `Optional[str]` |
| `updateCode(name, code, namespace, metadata)` | 更新代碼 | `DynamicNode` |
| `deleteCode(name, namespace)` | 刪除代碼 | `bool` |
| `importCode(name, namespace, globals_dict)` | 動態導入 | `module` |
| `run(name, namespace, entry_point, args, kwargs)` | 執行代碼 | `ExecutionResult` |

#### 數據管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `saveData(name, data, namespace, data_type, compression, metadata)` | 保存數據 | `DynamicNode` |
| `getData(name, namespace)` | 獲取數據 | `Optional[Any]` |
| `deleteData(name, namespace)` | 刪除數據 | `bool` |

#### 知識管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `saveKnowledge(name, content, namespace, tags, attachments, metadata)` | 保存知識 | `DynamicNode` |
| `getKnowledge(name, namespace)` | 獲取知識 | `Optional[Dict]` |
| `updateKnowledge(name, content, namespace, tags)` | 更新知識 | `DynamicNode` |

#### 命名空間管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `createNamespace(name, description, parent)` | 創建命名空間 | `NamespaceInfo` |
| `copyNamespace(source, target, include_codes, include_data, include_groups)` | 複製命名空間 | `Dict[str, int]` |
| `renameNamespace(old_name, new_name)` | 重命名命名空間 | `bool` |
| `deleteNamespace(name, force)` | 刪除命名空間 | `bool` |
| `listNamespaces()` | 列出命名空間 | `List[str]` |
| `getNamespace(name)` | 獲取命名空間信息 | `Optional[NamespaceInfo]` |

#### 資料組管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `createDataGroup(name, items, namespace, description, tags, calibrate)` | 創建資料組 | `DataGroup` |
| `getDataGroup(name, namespace)` | 獲取資料組 | `Optional[DataGroup]` |
| `updateDataGroup(name, items, namespace, description, tags, calibrate)` | 更新資料組 | `Optional[DataGroup]` |
| `addToDataGroup(name, content, item_type, metadata, namespace, recalibrate)` | 添加內容 | `Optional[DataGroup]` |
| `deleteDataGroup(name, namespace)` | 刪除資料組 | `bool` |
| `listDataGroups(namespace)` | 列出資料組 | `List[str]` |

#### Embedding 校準

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `batchImportWithCalibration(items, namespace, as_data_group, group_name)` | 帶校準的批量導入 | `List[DynamicNode]` 或 `DataGroup` |
| `calibrateNamespace(namespace)` | 校準命名空間 | `Dict[str, Any]` |

#### 搜索功能

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `search(query, mode, content_type, namespace, similarity_threshold, ...)` | 搜索 | `List[SearchResult]` |
| `searchRelated(query, content_types, namespace, threshold, limit)` | 關聯搜索 | `List[SearchResult]` |

#### 導入導出

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `fromFolder(folder_path, namespace, recursive, calibrate, file_patterns)` | 從文件夾導入 | `List[DynamicNode]` |
| `toFolder(folder_path, namespace, include_codes, include_data)` | 導出到文件夾 | `int` |
| `fromFileDict(source_storage, namespace)` | 從 FileDict 導入 | `List[DynamicNode]` |
| `toFileDict(target_storage, namespace)` | 導出到 FileDict | `int` |

#### 節點管理

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `listNodes(content_type, namespace)` | 列出節點 | `List[Dict]` |
| `getNode(name, namespace)` | 獲取節點詳情 | `Optional[Dict]` |

#### 演化與可視化

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `initEvolution(mode, **kwargs)` | 初始化演化 | `None` |
| `evolve(steps, rule, record_interval)` | 執行演化 | `List[EvolutionFrame]` |
| `getEvolutionState()` | 獲取演化狀態 | `np.ndarray` |
| `saveVisualization(prefix, effect)` | 保存可視化 | `Dict[str, str]` |

#### 系統功能

| 方法 | 說明 | 返回值 |
|------|------|--------|
| `getStats()` | 獲取統計信息 | `Dict[str, Any]` |
| `getSystemInfo()` | 獲取系統信息 | `Dict[str, Any]` |
| `get_debug_log(limit)` | 獲取調試日誌 | `List[Dict]` |
| `clear_debug_log()` | 清空調試日誌 | `None` |
| `close()` | 關閉應用 | `None` |

### 12.2 搜索模式

| 模式 | 說明 | 適用場景 |
|------|------|----------|
| `exact` | 精確匹配 | 查找特定關鍵詞 |
| `fuzzy` | 模糊匹配 | 通用搜索（默認） |
| `regex` | 正則表達式 | 模式匹配 |
| `semantic` | 語義搜索 | 概念相關搜索 |
| `monte_carlo` | 蒙特卡羅搜索 | 探索性搜索 |
| `related` | 關聯搜索 | 查找相關內容 |

### 12.3 內容類型

| 類型 | 說明 |
|------|------|
| `all` | 所有類型 |
| `code` | Python 代碼 |
| `data` | 結構化數據 |
| `knowledge` | 知識條目 |
| `file` | 二進制文件 |
| `data_group` | 資料組 |

---

## 結語

ASMdynamicGradio 動態知識與工具體系提供了一個強大而靈活的平台，用於管理代碼、知識和數據。通過合理使用命名空間、Embedding 校準、資料組等功能，可以構建高效的知識管理系統、代碼庫、或任何需要動態內容管理的應用。

關鍵要點：

1. **使用命名空間組織內容** — 保持項目結構清晰
2. **啟用 Embedding 校準** — 實現跨類型語義搜索
3. **善用資料組** — 將相關內容組織在一起
4. **利用蒙特卡羅搜索** — 發現意外的關聯
5. **定期備份和版本控制** — 使用 copyNamespace 和 toFolder