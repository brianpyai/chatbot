# ASMdynamicGradio 動態知識與工具體系完整使用指南

## 📑 目錄

1. [系統概述](#1-系統概述)
2. [安裝與環境配置](#2-安裝與環境配置)
3. [CLI 命令行介面](#3-cli-命令行介面)
4. [代碼層 API 完整指南](#4-代碼層-api-完整指南)
5. [動態知識管理](#5-動態知識管理)
6. [動態工具開發](#6-動態工具開發)
7. [蒙特卡羅搜索引擎](#7-蒙特卡羅搜索引擎)
8. [演化可視化系統](#8-演化可視化系統)
9. [實際應用場景](#9-實際應用場景)
10. [系統優越性分析](#10-系統優越性分析)
11. [最佳實踐與設計模式](#11-最佳實踐與設計模式)

---

## 1. 系統概述

### 1.1 架構圖

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ASMdynamicGradio 應用層                           │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                      Gradio Web 介面                                 ││
│  │  📝 代碼開發 │ 📚 知識管理 │ 🔍 搜索 │ 📦 導入導出 │ 🌀 演化可視化   ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    ↕                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                     DynamicApp 代碼層 API                            ││
│  │  saveCode │ getCode │ run │ importCode │ search │ evolve            ││
│  │  saveData │ getData │ saveKnowledge │ fromFolder │ toFolder         ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                    ↕                                     │
│  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────────────┐ │
│  │ MonteCarloSearch │ │ EvolutionEngine  │ │ NamespaceManager         │ │
│  │ Engine           │ │                  │ │                          │ │
│  └──────────────────┘ └──────────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────────────┐
│                    ASMsuperDynamicSystem 核心層                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐│
│  │ CodeManager │ │ DataManager │ │ NodeIO      │ │ DynamicRenderer     ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
                                    ↕
┌─────────────────────────────────────────────────────────────────────────┐
│                       ASMFileDict3 存儲層                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────────────────┐│
│  │ FileDict        │ │ FileSQL3        │ │ HDCVectorOps               ││
│  │ (SQLite KV)     │ │ (Binary Store)  │ │ (向量編碼)                  ││
│  └─────────────────┘ └─────────────────┘ └─────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心特性

| 特性 | 說明 |
|------|------|
| **動態代碼管理** | 運行時保存、加載、導入、執行 Python 代碼 |
| **動態數據管理** | 支持 JSON、NumPy、二進制等多種格式 |
| **知識庫系統** | Markdown 格式的知識條目，支持標籤和附件 |
| **蒙特卡羅搜索** | 創新的隨機採樣搜索算法，結果具有驚喜性 |
| **命名空間隔離** | 層級化的命名空間管理，支持多項目 |
| **演化可視化** | 細胞自動機演化引擎，支持多種規則 |
| **導入導出** | 文件夾、FileDict 之間的雙向傳輸 |
| **Web 介面** | 現代化 Gradio 界面，語法高亮編輯 |

---

## 2. 安裝與環境配置

### 2.1 依賴安裝

```bash
# 必要依賴
pip install numpy

# Web 介面
pip install gradio

# 可選依賴（完整功能）
pip install pillow          # 圖像處理
pip install opencv-python   # 視頻處理
pip install scipy           # 高級數學運算
```

### 2.2 文件結構

確保以下文件在同一目錄：

```
your_project/
├── ASMdynamicGradio.py        # 應用層（本文件）
├── ASMsuperDynamicSystem.py   # 核心系統
├── ASMFileDict3.py            # 存儲層（可選）
└── dynamic_app_data/          # 默認存儲目錄（自動創建）
```

### 2.3 快速驗證安裝

```python
from ASMdynamicGradio import DynamicApp

# 測試初始化
with DynamicApp("./test_app", auto_load=False) as app:
    app.saveCode("hello", "def main(): return 'Hello, World!'")
    result = app.run("hello")
    print(result.result)  # Hello, World!
```

---

## 3. CLI 命令行介面

### 3.1 基本命令

```bash
# 啟動 Web 介面（默認配置）
python ASMdynamicGradio.py

# 自定義端口
python ASMdynamicGradio.py --port 8080

# 創建公共分享鏈接（需要網絡）
python ASMdynamicGradio.py --share

# 指定存儲目錄和命名空間
python ASMdynamicGradio.py --storage-dir ./my_project --namespace main

# 運行完整測試套件
python ASMdynamicGradio.py --test
```

### 3.2 完整參數說明

| 參數 | 類型 | 默認值 | 說明 |
|------|------|--------|------|
| `--test` | flag | - | 運行測試套件 |
| `--port` | int | 7860 | Gradio 服務端口 |
| `--share` | flag | - | 創建公共分享鏈接 |
| `--storage-dir` | str | `./dynamic_app_data` | 存儲目錄 |
| `--namespace` | str | `default` | 默認命名空間 |

### 3.3 CLI 使用範例

```bash
# 場景 1：開發環境
python ASMdynamicGradio.py --port 7860 --storage-dir ./dev_data

# 場景 2：生產環境分享
python ASMdynamicGradio.py --port 80 --share --storage-dir ./prod_data

# 場景 3：多項目隔離
python ASMdynamicGradio.py --storage-dir ./project_alpha --namespace alpha
python ASMdynamicGradio.py --storage-dir ./project_beta --namespace beta --port 7861

# 場景 4：CI/CD 測試
python ASMdynamicGradio.py --test && echo "Tests passed!"
```

---

## 4. 代碼層 API 完整指南

### 4.1 初始化與上下文管理

```python
from ASMdynamicGradio import DynamicApp

# 方式 1：標準初始化
app = DynamicApp(
    storage_dir="./my_app_data",   # 存儲目錄
    namespace="main",               # 默認命名空間
    auto_load=True                  # 自動加載已存儲的數據
)

# 使用完畢後關閉
app.close()

# 方式 2：上下文管理器（推薦）
with DynamicApp("./my_app_data", namespace="main") as app:
    # 所有操作...
    pass
# 自動關閉

# 方式 3：臨時/測試用途
with DynamicApp("./temp", auto_load=False) as app:
    # 不加載已有數據，適合測試
    pass
```

### 4.2 代碼管理 API

#### 4.2.1 保存代碼 (`saveCode`)

```python
# 基本用法
app.saveCode("my_function", """
def main():
    return "Hello, World!"
""")

# 完整用法
node = app.saveCode(
    name="advanced_function",
    code="""
import math

def calculate(x, y):
    '''計算兩數的平方和'''
    return math.sqrt(x**2 + y**2)

def main(x=3, y=4):
    return calculate(x, y)
""",
    namespace="math_utils",  # 指定命名空間
    metadata={               # 附加元數據
        "author": "developer",
        "version": "1.0.0",
        "tags": ["math", "geometry"]
    }
)

print(f"節點 ID: {node.node_id}")      # math_utils.advanced_function
print(f"創建時間: {node.created}")
```

#### 4.2.2 獲取代碼 (`getCode`)

```python
# 從默認命名空間獲取
code = app.getCode("my_function")
if code:
    print(code)

# 從指定命名空間獲取
code = app.getCode("advanced_function", namespace="math_utils")
```

#### 4.2.3 更新代碼 (`updateCode`)

```python
# 更新已存在的代碼（保留元數據）
app.updateCode("my_function", """
def main():
    return "Updated Hello!"
""")

# 更新並修改元數據
app.updateCode(
    "advanced_function",
    code="def main(): return 42",
    namespace="math_utils",
    metadata={"version": "2.0.0"}
)
```

#### 4.2.4 執行代碼 (`run`)

```python
# 基本執行（調用 main 函數）
result = app.run("my_function")

if result.success:
    print(f"結果: {result.result}")
    print(f"執行時間: {result.execution_time_ms:.2f} ms")
else:
    print(f"錯誤: {result.error}")
    print(f"詳情: {result.stderr}")

# 帶參數執行
app.saveCode("calculator", """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def main(operation, x, y):
    if operation == "add":
        return add(x, y)
    elif operation == "multiply":
        return multiply(x, y)
    else:
        raise ValueError(f"Unknown operation: {operation}")
""")

result = app.run(
    "calculator",
    entry_point="main",           # 指定入口函數
    kwargs={                       # 關鍵字參數
        "operation": "multiply",
        "x": 6,
        "y": 7
    }
)
print(result.result)  # 42

# 直接調用特定函數
result = app.run("calculator", entry_point="add", kwargs={"a": 10, "b": 20})
print(result.result)  # 30
```

#### 4.2.5 動態導入 (`importCode`)

```python
# 保存可複用模組
app.saveCode("utils", """
PI = 3.14159265359

def circle_area(radius):
    return PI * radius ** 2

def circle_circumference(radius):
    return 2 * PI * radius

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def magnitude(self):
        return (self.x**2 + self.y**2) ** 0.5
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
""")

# 動態導入為模組
utils = app.importCode("utils")

# 像使用普通模組一樣使用
print(utils.PI)                           # 3.14159265359
print(utils.circle_area(5))               # 78.539816...
print(utils.circle_circumference(5))      # 31.4159...

v = utils.Vector(3, 4)
print(v)                                  # Vector(3, 4)
print(v.magnitude())                      # 5.0
```

#### 4.2.6 刪除代碼 (`deleteCode`)

```python
# 刪除代碼
if app.deleteCode("my_function"):
    print("刪除成功")
else:
    print("刪除失敗（可能不存在）")

# 從指定命名空間刪除
app.deleteCode("advanced_function", namespace="math_utils")
```

### 4.3 數據管理 API

#### 4.3.1 保存數據 (`saveData`)

```python
import numpy as np

# JSON 數據（自動檢測類型）
app.saveData("config", {
    "app_name": "MyApp",
    "settings": {"theme": "dark", "language": "zh-TW"},
    "features": ["feature_a", "feature_b"]
})

# NumPy 數組
app.saveData(
    "training_data",
    np.random.rand(1000, 784),
    data_type="numpy",
    compression=True  # 啟用壓縮（大數據推薦）
)

# 二進制數據
with open("image.png", "rb") as f:
    app.saveData("my_image", f.read(), data_type="binary")

# 帶元數據
app.saveData(
    "experiment_result",
    {"accuracy": 0.95, "loss": 0.05},
    metadata={
        "experiment_id": "exp_001",
        "timestamp": "2024-01-01T00:00:00"
    }
)
```

#### 4.3.2 獲取數據 (`getData`)

```python
# 獲取 JSON 數據
config = app.getData("config")
print(config["settings"]["theme"])  # dark

# 獲取 NumPy 數據
data = app.getData("training_data")
print(data.shape)  # (1000, 784)

# 獲取二進制數據
image_bytes = app.getData("my_image")
```

#### 4.3.3 刪除數據 (`deleteData`)

```python
app.deleteData("config")
app.deleteData("training_data", namespace="ml_project")
```

### 4.4 知識管理 API

#### 4.4.1 保存知識 (`saveKnowledge`)

```python
# 創建知識條目（Markdown 格式）
app.saveKnowledge(
    name="python_best_practices",
    content="""
# Python 最佳實踐

## 1. 代碼風格

遵循 PEP 8 規範：
- 使用 4 空格縮進
- 每行不超過 79 字符
- 函數和類之間空兩行

## 2. 命名規範

```python
# 變量和函數：snake_case
my_variable = 42
def my_function():
    pass

# 類：PascalCase
class MyClass:
    pass

# 常量：UPPER_CASE
MAX_SIZE = 100
```

## 3. 文檔字符串

```python
def calculate_area(width, height):
    '''
    計算矩形面積
    
    Args:
        width: 寬度
        height: 高度
    
    Returns:
        面積值
    '''
    return width * height
```
""",
    tags=["python", "coding", "best-practices"],
    metadata={"author": "Team Lead", "reviewed": True}
)
```

#### 4.4.2 獲取知識 (`getKnowledge`)

```python
knowledge = app.getKnowledge("python_best_practices")

print(knowledge["content"])        # Markdown 內容
print(knowledge["tags"])           # ['python', 'coding', 'best-practices']
print(knowledge["created"])        # 創建時間
print(knowledge["modified"])       # 修改時間
```

#### 4.4.3 更新知識 (`updateKnowledge`)

```python
# 更新內容
app.updateKnowledge(
    "python_best_practices",
    content="# 更新後的內容\n\n...",
    tags=["python", "updated"]
)
```

### 4.5 文件管理 API

#### 4.5.1 添加文件 (`addFile`)

```python
# 從文件系統添加
with open("document.pdf", "rb") as f:
    app.addFile("user_manual.pdf", f.read())

# 添加圖片（自動檢測 MIME 類型）
with open("logo.png", "rb") as f:
    app.addFile(
        "company_logo.png",
        f.read(),
        metadata={"description": "Company logo", "version": "2.0"}
    )

# 手動指定 MIME 類型
app.addFile(
    "custom_data.bin",
    b"\x00\x01\x02\x03",
    mime_type="application/octet-stream"
)
```

#### 4.5.2 獲取文件 (`getFile`)

```python
# 獲取文件數據
pdf_data = app.getFile("user_manual.pdf")

# 保存到文件系統
with open("downloaded.pdf", "wb") as f:
    f.write(pdf_data)
```

#### 4.5.3 獲取文件信息 (`getFileInfo`)

```python
info = app.getFileInfo("company_logo.png")

print(info.name)          # company_logo.png
print(info.path)          # default.company_logo.png
print(info.size)          # 文件大小（字節）
print(info.mime_type)     # image/png
print(info.created)       # 創建時間
print(info.modified)      # 修改時間
```

### 4.6 節點管理 API

#### 4.6.1 列出節點 (`listNodes`)

```python
# 列出所有節點
all_nodes = app.listNodes()
for node in all_nodes:
    print(f"{node['type']}: {node['namespace']}.{node['name']}")

# 按類型過濾
code_nodes = app.listNodes(content_type="code")
data_nodes = app.listNodes(content_type="data")
knowledge_nodes = app.listNodes(content_type="knowledge")
file_nodes = app.listNodes(content_type="file")

# 按命名空間過濾
project_nodes = app.listNodes(namespace="my_project")

# 組合過濾
project_codes = app.listNodes(content_type="code", namespace="my_project")
```

#### 4.6.2 獲取節點詳情 (`getNode`)

```python
node = app.getNode("my_function")

print(node["id"])           # default.my_function
print(node["name"])         # my_function
print(node["namespace"])    # default
print(node["type"])         # code
print(node["content"])      # 代碼內容
print(node["metadata"])     # 元數據
print(node["created"])      # 創建時間
print(node["modified"])     # 修改時間
```

### 4.7 命名空間管理 API

```python
# 創建命名空間
app.createNamespace("project_a", description="Project A workspace")

# 創建子命名空間
app.createNamespace("models", description="ML models", parent="project_a")
app.createNamespace("data", description="Training data", parent="project_a")

# 列出所有命名空間
namespaces = app.listNamespaces()
print(namespaces)  # ['default', 'project_a', 'models', 'data', ...]

# 獲取命名空間信息
ns_info = app.getNamespace("project_a")
print(ns_info.name)         # project_a
print(ns_info.description)  # Project A workspace
print(ns_info.children)     # ['models', 'data']
print(ns_info.parent)       # None
```

### 4.8 導入導出 API

#### 4.8.1 從文件夾導入 (`fromFolder`)

```python
# 基本導入
nodes = app.fromFolder("./my_project")
print(f"導入了 {len(nodes)} 個節點")

# 完整選項
nodes = app.fromFolder(
    folder_path="./my_project",
    namespace="imported",
    recursive=True,                    # 遞歸子目錄
    file_patterns=["*.py", "*.json"]   # 文件模式過濾
)
```

#### 4.8.2 導出到文件夾 (`toFolder`)

```python
# 導出整個命名空間
count = app.toFolder("./backup", namespace="my_project")
print(f"導出了 {count} 個文件")

# 選擇性導出
count = app.toFolder(
    folder_path="./code_backup",
    namespace="my_project",
    include_codes=True,
    include_data=False
)
```

#### 4.8.3 FileDict 傳輸

```python
from ASMdynamicGradio import FileDict

# 導出到 FileDict
target = FileDict("./backup.db")
count = app.toFileDict(target, namespace="my_project")
target.close()

# 從 FileDict 導入
source = FileDict("./backup.db")
nodes = app.fromFileDict(source, namespace="restored")
source.close()
```

---

## 5. 動態知識管理

### 5.1 構建企業知識庫

```python
class EnterpriseKnowledgeBase:
    """企業級知識庫系統"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self._setup_structure()
    
    def _setup_structure(self):
        """建立知識庫結構"""
        categories = [
            ("policies", "公司政策"),
            ("procedures", "操作流程"),
            ("tutorials", "教程指南"),
            ("faqs", "常見問題"),
            ("templates", "模板庫")
        ]
        
        for name, desc in categories:
            self.app.createNamespace(name, description=desc)
    
    def add_article(self, title: str, content: str, 
                    category: str, tags: list = None):
        """添加知識文章"""
        # 生成唯一 ID
        article_id = f"article_{hash(title) & 0xFFFFFF:06x}"
        
        self.app.saveKnowledge(
            name=article_id,
            content=content,
            namespace=category,
            tags=tags or [],
            metadata={
                "title": title,
                "views": 0,
                "helpful_votes": 0
            }
        )
        return article_id
    
    def search_articles(self, query: str, top_k: int = 10):
        """搜索知識文章"""
        results = self.app.search(
            query=query,
            mode="monte_carlo",
            content_type="knowledge",
            monte_carlo_samples=100,
            result_limit=top_k
        )
        
        articles = []
        for r in results:
            knowledge = self.app.getKnowledge(r.name, r.namespace)
            if knowledge:
                articles.append({
                    "id": r.node_id,
                    "title": knowledge.get("metadata", {}).get("title", r.name),
                    "content": knowledge["content"][:200] + "...",
                    "score": r.score,
                    "tags": knowledge.get("tags", [])
                })
        
        return articles
    
    def get_article(self, article_id: str, category: str):
        """獲取完整文章"""
        knowledge = self.app.getKnowledge(article_id, category)
        if knowledge:
            # 增加閱讀計數
            knowledge["metadata"]["views"] = \
                knowledge.get("metadata", {}).get("views", 0) + 1
            self.app.saveKnowledge(
                article_id, 
                knowledge["content"],
                category,
                knowledge["tags"],
                knowledge["metadata"]
            )
        return knowledge

# 使用示例
with DynamicApp("./knowledge_base") as app:
    kb = EnterpriseKnowledgeBase(app)
    
    # 添加文章
    kb.add_article(
        title="新員工入職指南",
        content="""
# 新員工入職指南

歡迎加入我們的團隊！

## 第一週任務

1. 完成 HR 入職手續
2. 設置開發環境
3. 閱讀團隊規範文檔

## 常用資源

- 內部 Wiki: https://wiki.company.com
- 代碼倉庫: https://git.company.com
        """,
        category="tutorials",
        tags=["新員工", "入職", "指南"]
    )
    
    # 搜索文章
    results = kb.search_articles("新員工 入職")
    for article in results:
        print(f"[{article['score']:.2f}] {article['title']}")
```

### 5.2 個人筆記系統

```python
class PersonalNotes:
    """個人筆記系統"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("notes", "我的筆記")
        self.app.createNamespace("daily", "每日筆記", parent="notes")
        self.app.createNamespace("projects", "項目筆記", parent="notes")
    
    def quick_note(self, content: str, tags: list = None):
        """快速記錄"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.app.saveKnowledge(
            name=f"quick_{timestamp}",
            content=content,
            namespace="notes",
            tags=tags or ["quick"]
        )
    
    def daily_log(self, content: str):
        """每日日誌"""
        from datetime import datetime
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        # 檢查今天的日誌是否存在
        existing = self.app.getKnowledge(f"log_{today}", "daily")
        
        if existing:
            # 追加內容
            new_content = existing["content"] + f"\n\n---\n\n{content}"
            self.app.updateKnowledge(f"log_{today}", new_content, "daily")
        else:
            # 創建新日誌
            self.app.saveKnowledge(
                name=f"log_{today}",
                content=f"# {today} 工作日誌\n\n{content}",
                namespace="daily",
                tags=["daily", today]
            )
    
    def search_notes(self, query: str):
        """搜索筆記"""
        return self.app.search(
            query,
            mode="fuzzy",
            content_type="knowledge",
            result_limit=20
        )

# 使用示例
with DynamicApp("./my_notes") as app:
    notes = PersonalNotes(app)
    
    # 快速筆記
    notes.quick_note("今天學習了 Python 裝飾器", tags=["python", "learning"])
    
    # 每日日誌
    notes.daily_log("完成了 API 設計文檔")
    notes.daily_log("修復了登錄頁面的 bug")
    
    # 搜索
    results = notes.search_notes("Python")
    for r in results:
        print(f"📝 {r.name}: {r.preview[:50]}...")
```

---

## 6. 動態工具開發

### 6.1 插件系統架構

```python
class PluginSystem:
    """動態插件系統"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("plugins", "插件系統")
        self.loaded_plugins = {}
    
    def register_plugin(self, name: str, code: str, metadata: dict = None):
        """註冊插件"""
        # 驗證插件結構
        required_functions = ["initialize", "process", "cleanup"]
        
        for func in required_functions:
            if f"def {func}" not in code:
                raise ValueError(f"插件缺少必要函數: {func}")
        
        self.app.saveCode(
            name=name,
            code=code,
            namespace="plugins",
            metadata={
                "type": "plugin",
                "enabled": True,
                **(metadata or {})
            }
        )
    
    def load_plugin(self, name: str):
        """加載插件"""
        module = self.app.importCode(name, "plugins")
        
        if module:
            # 調用初始化
            if hasattr(module, "initialize"):
                module.initialize()
            
            self.loaded_plugins[name] = module
            return module
        
        return None
    
    def run_plugin(self, name: str, data: any):
        """運行插件處理"""
        if name not in self.loaded_plugins:
            self.load_plugin(name)
        
        plugin = self.loaded_plugins.get(name)
        if plugin and hasattr(plugin, "process"):
            return plugin.process(data)
        
        return None
    
    def unload_plugin(self, name: str):
        """卸載插件"""
        if name in self.loaded_plugins:
            plugin = self.loaded_plugins[name]
            if hasattr(plugin, "cleanup"):
                plugin.cleanup()
            del self.loaded_plugins[name]

# 使用示例
with DynamicApp("./plugin_app") as app:
    plugins = PluginSystem(app)
    
    # 註冊數據處理插件
    plugins.register_plugin("json_formatter", """
import json

def initialize():
    print("JSON Formatter 插件已加載")

def process(data):
    '''格式化 JSON 數據'''
    if isinstance(data, str):
        data = json.loads(data)
    return json.dumps(data, indent=2, ensure_ascii=False)

def cleanup():
    print("JSON Formatter 插件已卸載")
""", metadata={"version": "1.0", "author": "dev"})
    
    # 註冊文本處理插件
    plugins.register_plugin("text_stats", """
import re

def initialize():
    pass

def process(text):
    '''計算文本統計'''
    words = len(re.findall(r'\\w+', text))
    chars = len(text)
    lines = text.count('\\n') + 1
    
    return {
        "words": words,
        "characters": chars,
        "lines": lines,
        "avg_word_length": chars / words if words > 0 else 0
    }

def cleanup():
    pass
""")
    
    # 使用插件
    formatted = plugins.run_plugin("json_formatter", {"name": "test", "value": 42})
    print(formatted)
    
    stats = plugins.run_plugin("text_stats", "Hello World!\nThis is a test.")
    print(stats)  # {'words': 5, 'characters': 28, 'lines': 2, ...}
```

### 6.2 動態工作流引擎

```python
class WorkflowEngine:
    """動態工作流引擎"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("workflows", "工作流定義")
        self.app.createNamespace("tasks", "任務定義")
    
    def define_task(self, name: str, code: str, 
                    inputs: list = None, outputs: list = None):
        """定義任務"""
        self.app.saveCode(
            name=name,
            code=code,
            namespace="tasks",
            metadata={
                "inputs": inputs or [],
                "outputs": outputs or [],
                "type": "task"
            }
        )
    
    def define_workflow(self, name: str, steps: list):
        """定義工作流"""
        self.app.saveData(
            name=name,
            data={
                "name": name,
                "steps": steps,
                "type": "workflow"
            },
            namespace="workflows"
        )
    
    def execute(self, workflow_name: str, initial_context: dict = None):
        """執行工作流"""
        workflow = self.app.getData(workflow_name, "workflows")
        if not workflow:
            raise ValueError(f"工作流不存在: {workflow_name}")
        
        context = initial_context or {}
        execution_log = []
        
        for i, step in enumerate(workflow["steps"]):
            task_name = step["task"]
            step_params = step.get("params", {})
            
            # 構建任務輸入
            task_context = {**context, **step_params}
            
            # 執行任務
            result = self.app.run(
                task_name,
                namespace="tasks",
                entry_point="execute",
                kwargs={"context": task_context}
            )
            
            log_entry = {
                "step": i + 1,
                "task": task_name,
                "success": result.success,
                "time_ms": result.execution_time_ms
            }
            
            if result.success:
                # 合併輸出到上下文
                if isinstance(result.result, dict):
                    context.update(result.result)
                log_entry["output"] = result.result
            else:
                log_entry["error"] = result.error
                
                # 錯誤處理策略
                if step.get("on_error") == "stop":
                    execution_log.append(log_entry)
                    break
                elif step.get("on_error") == "skip":
                    pass  # 繼續下一步
            
            execution_log.append(log_entry)
        
        return {
            "success": all(e["success"] for e in execution_log),
            "context": context,
            "log": execution_log
        }

# 使用示例
with DynamicApp("./workflow_app") as app:
    engine = WorkflowEngine(app)
    
    # 定義任務：數據驗證
    engine.define_task("validate", """
def execute(context):
    data = context.get("data", [])
    
    if not isinstance(data, list):
        return {"error": "Data must be a list", "valid": False}
    
    if len(data) == 0:
        return {"error": "Data is empty", "valid": False}
    
    return {"valid": True, "count": len(data)}
""", inputs=["data"], outputs=["valid", "count"])
    
    # 定義任務：數據轉換
    engine.define_task("transform", """
def execute(context):
    if not context.get("valid"):
        return {"transformed": []}
    
    data = context.get("data", [])
    multiplier = context.get("multiplier", 2)
    
    transformed = [x * multiplier for x in data]
    return {"transformed": transformed}
""", inputs=["valid", "data", "multiplier"], outputs=["transformed"])
    
    # 定義任務：聚合
    engine.define_task("aggregate", """
def execute(context):
    data = context.get("transformed", [])
    
    if not data:
        return {"result": None}
    
    return {
        "result": {
            "sum": sum(data),
            "avg": sum(data) / len(data),
            "min": min(data),
            "max": max(data)
        }
    }
""", inputs=["transformed"], outputs=["result"])
    
    # 定義工作流
    engine.define_workflow("data_pipeline", [
        {"task": "validate", "on_error": "stop"},
        {"task": "transform", "params": {"multiplier": 3}},
        {"task": "aggregate"}
    ])
    
    # 執行工作流
    result = engine.execute("data_pipeline", {
        "data": [1, 2, 3, 4, 5]
    })
    
    print(f"成功: {result['success']}")
    print(f"結果: {result['context']['result']}")
    # {'sum': 45, 'avg': 9.0, 'min': 3, 'max': 15}
```

### 6.3 熱更新服務框架

```python
class HotReloadService:
    """支持熱更新的服務框架"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("services", "服務模組")
        self._handlers = {}
    
    def register_handler(self, route: str, code: str):
        """註冊處理器（可熱更新）"""
        handler_name = f"handler_{route.replace('/', '_')}"
        
        self.app.saveCode(
            handler_name,
            code,
            namespace="services",
            metadata={"route": route, "type": "handler"}
        )
        
        self._handlers[route] = handler_name
    
    def update_handler(self, route: str, new_code: str):
        """熱更新處理器"""
        if route in self._handlers:
            handler_name = self._handlers[route]
            self.app.updateCode(handler_name, new_code, "services")
            print(f"處理器已更新: {route}")
    
    def handle_request(self, route: str, request_data: dict):
        """處理請求"""
        if route not in self._handlers:
            return {"error": f"Route not found: {route}", "status": 404}
        
        handler_name = self._handlers[route]
        
        result = self.app.run(
            handler_name,
            namespace="services",
            entry_point="handle",
            kwargs={"request": request_data}
        )
        
        if result.success:
            return {"data": result.result, "status": 200}
        else:
            return {"error": result.error, "status": 500}
    
    def list_routes(self):
        """列出所有路由"""
        return list(self._handlers.keys())

# 使用示例
with DynamicApp("./service_app") as app:
    service = HotReloadService(app)
    
    # 註冊 API 處理器
    service.register_handler("/api/hello", """
def handle(request):
    name = request.get("name", "World")
    return {"message": f"Hello, {name}!"}
""")
    
    service.register_handler("/api/calculate", """
def handle(request):
    a = request.get("a", 0)
    b = request.get("b", 0)
    op = request.get("op", "add")
    
    if op == "add":
        return {"result": a + b}
    elif op == "subtract":
        return {"result": a - b}
    elif op == "multiply":
        return {"result": a * b}
    else:
        return {"error": f"Unknown operation: {op}"}
""")
    
    # 處理請求
    response = service.handle_request("/api/hello", {"name": "Alice"})
    print(response)  # {'data': {'message': 'Hello, Alice!'}, 'status': 200}
    
    response = service.handle_request("/api/calculate", {"a": 10, "b": 5, "op": "multiply"})
    print(response)  # {'data': {'result': 50}, 'status': 200}
    
    # 熱更新處理器
    service.update_handler("/api/hello", """
def handle(request):
    name = request.get("name", "World")
    greeting = request.get("greeting", "Hello")
    return {"message": f"{greeting}, {name}!", "version": "2.0"}
""")
    
    # 新處理器立即生效
    response = service.handle_request("/api/hello", {"name": "Bob", "greeting": "Hi"})
    print(response)  # {'data': {'message': 'Hi, Bob!', 'version': '2.0'}, 'status': 200}
```

---

## 7. 蒙特卡羅搜索引擎

### 7.1 搜索模式對比

| 模式 | 說明 | 適用場景 | 特點 |
|------|------|----------|------|
| `exact` | 精確匹配 | 已知確切關鍵詞 | 速度快，結果確定 |
| `fuzzy` | 模糊匹配 | 拼寫不確定 | 容錯性好 |
| `regex` | 正則表達式 | 複雜模式匹配 | 靈活強大 |
| `semantic` | 語義搜索 | 概念相似 | 理解語義 |
| `monte_carlo` | 蒙特卡羅 | 探索性搜索 | 結果有驚喜 |

### 7.2 蒙特卡羅搜索原理

```python
"""
蒙特卡羅搜索算法流程：

1. 候選收集：收集所有可能的匹配候選
2. 初步評分：計算每個候選的基礎相似度分數
   - Token 重疊度 (40%)
   - 向量相似度 (60%)
3. 概率採樣：根據分數進行加權隨機採樣
   - 高分候選更容易被選中
   - 探索因子確保低分候選也有機會
4. 精細評估：對採樣結果進行更細緻的評估
   - 位置加權（開頭出現加分）
   - 長度懲罰（過長/過短降分）
5. 隨機擾動：添加微小隨機性，保持驚喜
6. 排序返回：按最終分數排序

優勢：
- 每次搜索可能返回略有不同的結果
- 能發現傳統搜索遺漏的相關內容
- 平衡精確性和探索性
"""

# 直接使用蒙特卡羅搜索
results = app.search(
    query="機器學習 神經網絡",
    mode="monte_carlo",
    monte_carlo_samples=100,      # 採樣數量
    similarity_threshold=0.3,      # 相似度閾值
    result_limit=20                # 結果數量
)

for r in results:
    print(f"[{r.score:.3f}] {r.name}: {r.preview[:50]}...")
```

### 7.3 進階搜索技巧

```python
# 組合搜索策略
def smart_search(app, query: str, top_k: int = 10):
    """智能搜索：結合多種模式"""
    
    all_results = {}
    
    # 1. 先進行精確匹配
    exact_results = app.search(query, mode="exact", result_limit=top_k)
    for r in exact_results:
        all_results[r.node_id] = {"result": r, "exact": True}
    
    # 2. 模糊匹配補充
    fuzzy_results = app.search(query, mode="fuzzy", 
                               similarity_threshold=0.4, result_limit=top_k)
    for r in fuzzy_results:
        if r.node_id not in all_results:
            all_results[r.node_id] = {"result": r, "exact": False}
    
    # 3. 蒙特卡羅探索
    mc_results = app.search(query, mode="monte_carlo",
                            monte_carlo_samples=50, result_limit=top_k)
    for r in mc_results:
        if r.node_id not in all_results:
            all_results[r.node_id] = {"result": r, "exact": False, "exploration": True}
    
    # 排序：精確匹配優先，然後按分數
    sorted_results = sorted(
        all_results.values(),
        key=lambda x: (x.get("exact", False), x["result"].score),
        reverse=True
    )
    
    return [item["result"] for item in sorted_results[:top_k]]

# 使用
results = smart_search(app, "數據處理 函數")
```

### 7.4 搜索引擎自定義

```python
from ASMdynamicGradio import MonteCarloSearchEngine

# 自定義搜索引擎
class CustomSearchEngine(MonteCarloSearchEngine):
    """自定義搜索引擎"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_weights = {
            "code": 1.2,      # 代碼權重
            "knowledge": 1.0,  # 知識權重
            "data": 0.8       # 數據權重
        }
    
    def weighted_search(self, query: str, candidates: list, **kwargs):
        """加權搜索"""
        # 調整候選權重
        weighted_candidates = []
        for cid, content, meta in candidates:
            content_type = meta.get("type", "data")
            weight = self.custom_weights.get(content_type, 1.0)
            
            # 在內容前添加權重標記
            weighted_content = f"{'[HIGH]' if weight > 1 else ''} {content}"
            weighted_candidates.append((cid, weighted_content, meta))
        
        return self.monte_carlo_search(query, weighted_candidates, **kwargs)

# 使用自定義引擎
engine = CustomSearchEngine(dim=1024)
```

---

## 8. 演化可視化系統

### 8.1 基本演化操作

```python
with DynamicApp("./evolution_demo") as app:
    # 初始化演化狀態
    app.initEvolution(mode="random")      # 隨機
    # app.initEvolution(mode="center")    # 中心點
    # app.initEvolution(mode="gradient")  # 漸變
    # app.initEvolution(mode="noise")     # 噪聲
    
    # 獲取當前狀態
    state = app.getEvolutionState()
    print(f"狀態形狀: {state.shape}")      # (128, 128)
    print(f"活躍比例: {(state > 0.5).mean():.2%}")
    
    # 執行演化
    frames = app.evolve(
        steps=100,           # 演化步數
        rule="diffusion",    # 演化規則
        record_interval=5    # 每 5 步記錄一幀
    )
    
    print(f"記錄了 {len(frames)} 幀")
    
    # 檢查演化指標
    for frame in frames[-3:]:
        print(f"幀 {frame.frame_id}: "
              f"mean={frame.metrics['mean']:.4f}, "
              f"entropy={frame.metrics['entropy']:.4f}")
```

### 8.2 演化規則說明

| 規則 | 說明 | 視覺效果 |
|------|------|----------|
| `diffusion` | 擴散規則 | 平滑過渡，像墨水擴散 |
| `conway` | 康威生命遊戲 | 細胞生死演化 |
| `wave` | 波動規則 | 波紋擴散效果 |
| `growth` | 生長規則 | 強者更強，弱者衰退 |
| `erosion` | 侵蝕規則 | 逐漸消退 |

### 8.3 自定義演化規則

```python
# 通過系統底層註冊自定義規則
def custom_rule(state, param1=0.1, param2=0.5):
    """自定義演化規則"""
    import numpy as np
    
    # 計算鄰居平均值
    neighbors = (
        np.roll(state, 1, axis=0) +
        np.roll(state, -1, axis=0) +
        np.roll(state, 1, axis=1) +
        np.roll(state, -1, axis=1)
    ) / 4
    
    # 應用自定義邏輯
    new_state = state * (1 - param1) + neighbors * param1
    new_state = np.where(new_state > param2, new_state * 1.1, new_state * 0.9)
    
    return np.clip(new_state, 0, 1).astype(np.float32)

# 註冊規則
app._system.evolution_engine.register_rule("custom", custom_rule)

# 使用自定義規則
frames = app.evolve(steps=100, rule="custom", record_interval=5)
```

### 8.4 保存可視化輸出

```python
# 執行演化並渲染
app.initEvolution(mode="noise")
frames = app.evolve(steps=200, rule="diffusion", record_interval=4)

# 渲染幀
app._system.renderEvolution(frames, effect="plasma")

# 保存各種格式
outputs = app.saveVisualization(
    prefix="evolution_demo",
    effect="glow"
)

print("生成的文件:")
for fmt, path in outputs.items():
    if path:
        print(f"  {fmt}: {path}")
# json: ./dynamic_app_data/outputs/evolution_demo.json
# png: ./dynamic_app_data/outputs/evolution_demo.png
# gif: ./dynamic_app_data/outputs/evolution_demo.gif
# mp4: ./dynamic_app_data/outputs/evolution_demo.mp4
```

---

## 9. 實際應用場景

### 9.1 機器學習實驗管理

```python
class MLExperimentManager:
    """機器學習實驗管理器"""
    
    def __init__(self, app: DynamicApp, project_name: str):
        self.app = app
        self.project = project_name
        
        # 創建項目結構
        self.app.createNamespace(project_name)
        for sub in ["models", "data", "experiments", "metrics"]:
            self.app.createNamespace(f"{project_name}_{sub}", parent=project_name)
    
    def save_model_code(self, name: str, code: str, hyperparams: dict = None):
        """保存模型代碼"""
        self.app.saveCode(
            name=name,
            code=code,
            namespace=f"{self.project}_models",
            metadata={"hyperparams": hyperparams or {}}
        )
    
    def save_dataset(self, name: str, X, y, split: str = "train"):
        """保存數據集"""
        import numpy as np
        
        self.app.saveData(
            f"{name}_X_{split}",
            X,
            namespace=f"{self.project}_data",
            data_type="numpy",
            compression=True
        )
        self.app.saveData(
            f"{name}_y_{split}",
            y,
            namespace=f"{self.project}_data",
            data_type="numpy",
            compression=True
        )
    
    def load_dataset(self, name: str, split: str = "train"):
        """加載數據集"""
        X = self.app.getData(f"{name}_X_{split}", f"{self.project}_data")
        y = self.app.getData(f"{name}_y_{split}", f"{self.project}_data")
        return X, y
    
    def run_experiment(self, exp_name: str, model_name: str, 
                       dataset_name: str, config: dict = None):
        """運行實驗"""
        from datetime import datetime
        
        # 加載數據
        X_train, y_train = self.load_dataset(dataset_name, "train")
        X_test, y_test = self.load_dataset(dataset_name, "test")
        
        # 構建執行環境
        experiment_code = f"""
import numpy as np

# 加載模型代碼
{self.app.getCode(model_name, f"{self.project}_models")}

def main(X_train, y_train, X_test, y_test, config):
    # 訓練模型
    model = train(X_train, y_train, config)
    
    # 評估模型
    predictions = predict(model, X_test)
    accuracy = np.mean(predictions == y_test)
    
    return {{
        "accuracy": float(accuracy),
        "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else predictions
    }}
"""
        
        # 臨時保存實驗代碼
        self.app.saveCode(f"exp_{exp_name}", experiment_code, 
                          f"{self.project}_experiments")
        
        # 執行實驗
        result = self.app.run(
            f"exp_{exp_name}",
            namespace=f"{self.project}_experiments",
            kwargs={
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "config": config or {}
            }
        )
        
        # 保存結果
        experiment_record = {
            "name": exp_name,
            "model": model_name,
            "dataset": dataset_name,
            "config": config,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "metrics": result.result if result.success else None,
            "error": result.error if not result.success else None,
            "execution_time_ms": result.execution_time_ms
        }
        
        self.app.saveData(
            f"exp_record_{exp_name}",
            experiment_record,
            namespace=f"{self.project}_metrics"
        )
        
        return experiment_record
    
    def get_best_experiment(self, metric: str = "accuracy"):
        """獲取最佳實驗"""
        experiments = []
        
        for node in self.app.listNodes(namespace=f"{self.project}_metrics"):
            if node["name"].startswith("exp_record_"):
                record = self.app.getData(node["name"], f"{self.project}_metrics")
                if record and record.get("success") and record.get("metrics"):
                    experiments.append(record)
        
        if not experiments:
            return None
        
        return max(experiments, key=lambda x: x["metrics"].get(metric, 0))

# 使用示例
with DynamicApp("./ml_experiments") as app:
    manager = MLExperimentManager(app, "image_classification")
    
    # 保存模型代碼
    manager.save_model_code("simple_classifier", """
import numpy as np

def train(X, y, config):
    '''簡單的線性分類器'''
    lr = config.get("learning_rate", 0.01)
    epochs = config.get("epochs", 100)
    
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros(n_classes)
    
    for _ in range(epochs):
        scores = X @ W + b
        exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        dscores = probs.copy()
        dscores[range(len(y)), y] -= 1
        dscores /= len(y)
        
        W -= lr * (X.T @ dscores)
        b -= lr * dscores.sum(axis=0)
    
    return {"W": W, "b": b}

def predict(model, X):
    scores = X @ model["W"] + model["b"]
    return np.argmax(scores, axis=1)
""", hyperparams={"learning_rate": 0.01, "epochs": 100})
    
    # 保存模擬數據集
    import numpy as np
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 3, 100)
    X_test = np.random.randn(20, 10)
    y_test = np.random.randint(0, 3, 20)
    
    manager.save_dataset("synthetic", X_train, y_train, "train")
    manager.save_dataset("synthetic", X_test, y_test, "test")
    
    # 運行實驗
    result = manager.run_experiment(
        "exp_001",
        model_name="simple_classifier",
        dataset_name="synthetic",
        config={"learning_rate": 0.1, "epochs": 200}
    )
    
    print(f"實驗結果: {result['metrics']}")
```

### 9.2 API 網關模擬

```python
class APIGateway:
    """API 網關模擬器"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("api", "API 配置")
        self.app.createNamespace("handlers", "請求處理器")
        self.app.createNamespace("middleware", "中間件")
        self.routes = {}
    
    def register_route(self, method: str, path: str, handler_code: str):
        """註冊路由"""
        route_key = f"{method}:{path}"
        handler_name = f"handler_{method}_{path.replace('/', '_')}"
        
        self.app.saveCode(handler_name, handler_code, "handlers")
        self.routes[route_key] = handler_name
        
        # 保存路由配置
        self.app.saveData("routes", self.routes, "api")
    
    def add_middleware(self, name: str, code: str, priority: int = 0):
        """添加中間件"""
        self.app.saveCode(name, code, "middleware", 
                          metadata={"priority": priority})
    
    def request(self, method: str, path: str, 
                headers: dict = None, body: dict = None):
        """處理請求"""
        import time
        
        request_data = {
            "method": method,
            "path": path,
            "headers": headers or {},
            "body": body or {},
            "timestamp": time.time()
        }
        
        # 執行中間件（前置）
        middleware_nodes = self.app.listNodes(
            content_type="code", 
            namespace="middleware"
        )
        
        for mw in sorted(middleware_nodes, 
                        key=lambda x: x.get("metadata", {}).get("priority", 0)):
            result = self.app.run(
                mw["name"],
                namespace="middleware",
                entry_point="before_request",
                kwargs={"request": request_data}
            )
            
            if result.success and isinstance(result.result, dict):
                request_data.update(result.result)
        
        # 查找路由
        route_key = f"{method}:{path}"
        
        if route_key not in self.routes:
            return {
                "status": 404,
                "body": {"error": f"Route not found: {path}"}
            }
        
        handler_name = self.routes[route_key]
        
        # 執行處理器
        result = self.app.run(
            handler_name,
            namespace="handlers",
            entry_point="handle",
            kwargs={"request": request_data}
        )
        
        if result.success:
            response = {
                "status": 200,
                "body": result.result,
                "execution_time_ms": result.execution_time_ms
            }
        else:
            response = {
                "status": 500,
                "body": {"error": result.error}
            }
        
        return response

# 使用示例
with DynamicApp("./api_gateway") as app:
    gateway = APIGateway(app)
    
    # 添加認證中間件
    gateway.add_middleware("auth", """
def before_request(request):
    auth_header = request.get("headers", {}).get("Authorization", "")
    
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # 簡化的 token 驗證
        request["authenticated"] = len(token) > 10
        request["user_id"] = "user_123" if request["authenticated"] else None
    else:
        request["authenticated"] = False
    
    return request
""", priority=1)
    
    # 註冊 API 路由
    gateway.register_route("GET", "/users", """
def handle(request):
    if not request.get("authenticated"):
        return {"error": "Unauthorized"}
    
    return {
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
    }
""")
    
    gateway.register_route("POST", "/users", """
def handle(request):
    if not request.get("authenticated"):
        return {"error": "Unauthorized"}
    
    body = request.get("body", {})
    name = body.get("name", "Unknown")
    
    return {
        "created": True,
        "user": {"id": 3, "name": name}
    }
""")
    
    # 測試請求
    # 無認證
    response = gateway.request("GET", "/users")
    print(response)  # {'status': 200, 'body': {'error': 'Unauthorized'}, ...}
    
    # 有認證
    response = gateway.request(
        "GET", "/users",
        headers={"Authorization": "Bearer valid_token_12345"}
    )
    print(response)  # {'status': 200, 'body': {'users': [...]}, ...}
    
    # POST 請求
    response = gateway.request(
        "POST", "/users",
        headers={"Authorization": "Bearer valid_token_12345"},
        body={"name": "Charlie"}
    )
    print(response)  # {'status': 200, 'body': {'created': True, 'user': {...}}, ...}
```

### 9.3 配置管理中心

```python
class ConfigCenter:
    """配置管理中心"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("configs", "配置中心")
        
        # 創建環境
        for env in ["dev", "staging", "prod"]:
            self.app.createNamespace(f"configs_{env}", parent="configs")
        
        self._watchers = {}
    
    def set_config(self, key: str, value: any, env: str = "dev"):
        """設置配置"""
        from datetime import datetime
        
        config_data = {
            "value": value,
            "updated_at": datetime.now().isoformat(),
            "version": self._get_version(key, env) + 1
        }
        
        self.app.saveData(key, config_data, f"configs_{env}")
        
        # 通知監聽器
        if key in self._watchers:
            for callback in self._watchers[key]:
                callback(key, value, env)
    
    def get_config(self, key: str, env: str = "dev", default: any = None):
        """獲取配置"""
        data = self.app.getData(key, f"configs_{env}")
        
        if data:
            return data["value"]
        
        return default
    
    def _get_version(self, key: str, env: str) -> int:
        """獲取配置版本"""
        data = self.app.getData(key, f"configs_{env}")
        return data.get("version", 0) if data else 0
    
    def watch(self, key: str, callback: callable):
        """監聽配置變更"""
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
    
    def get_all_configs(self, env: str = "dev") -> dict:
        """獲取所有配置"""
        configs = {}
        
        for node in self.app.listNodes(namespace=f"configs_{env}"):
            data = self.app.getData(node["name"], f"configs_{env}")
            if data:
                configs[node["name"]] = data["value"]
        
        return configs
    
    def compare_envs(self, env1: str, env2: str) -> dict:
        """比較兩個環境的配置"""
        configs1 = self.get_all_configs(env1)
        configs2 = self.get_all_configs(env2)
        
        all_keys = set(configs1.keys()) | set(configs2.keys())
        
        diff = {}
        for key in all_keys:
            v1 = configs1.get(key)
            v2 = configs2.get(key)
            
            if v1 != v2:
                diff[key] = {"env1": v1, "env2": v2}
        
        return diff
    
    def copy_to_env(self, from_env: str, to_env: str, keys: list = None):
        """複製配置到另一個環境"""
        configs = self.get_all_configs(from_env)
        
        if keys:
            configs = {k: v for k, v in configs.items() if k in keys}
        
        for key, value in configs.items():
            self.set_config(key, value, to_env)
        
        return len(configs)

# 使用示例
with DynamicApp("./config_center") as app:
    config = ConfigCenter(app)
    
    # 設置開發環境配置
    config.set_config("database.host", "localhost", "dev")
    config.set_config("database.port", 5432, "dev")
    config.set_config("cache.enabled", True, "dev")
    config.set_config("log.level", "DEBUG", "dev")
    
    # 設置生產環境配置
    config.set_config("database.host", "db.production.com", "prod")
    config.set_config("database.port", 5432, "prod")
    config.set_config("cache.enabled", True, "prod")
    config.set_config("log.level", "ERROR", "prod")
    
    # 獲取配置
    host = config.get_config("database.host", "dev")
    print(f"Dev Database Host: {host}")  # localhost
    
    host = config.get_config("database.host", "prod")
    print(f"Prod Database Host: {host}")  # db.production.com
    
    # 比較環境
    diff = config.compare_envs("dev", "prod")
    print("環境差異:")
    for key, values in diff.items():
        print(f"  {key}: dev={values['env1']} prod={values['env2']}")
    
    # 監聽變更
    def on_config_change(key, value, env):
        print(f"配置變更: {key} = {value} ({env})")
    
    config.watch("database.host", on_config_change)
    config.set_config("database.host", "new-host.dev", "dev")
    # 輸出: 配置變更: database.host = new-host.dev (dev)
```

---

## 10. 系統優越性分析

### 10.1 與傳統方案對比

#### 10.1.1 代碼管理

| 特性 | 傳統文件系統 | Git 版本控制 | ASMdynamicGradio |
|------|-------------|--------------|------------------|
| **動態加載** | ❌ 需重啟 | ❌ 需重啟 | ✅ 即時生效 |
| **熱更新** | ❌ | ❌ | ✅ 運行時更新 |
| **統一存儲** | ❌ 分散 | ⚠️ 需倉庫 | ✅ 單一數據庫 |
| **向量搜索** | ❌ | ❌ | ✅ 語義搜索 |
| **執行追蹤** | ❌ | ❌ | ✅ 內建記錄 |

```python
# 傳統方式：修改代碼需要重啟
# 1. 編輯文件
# 2. 保存
# 3. 重啟應用
# 4. 測試

# ASMdynamicGradio：即時生效
app.updateCode("my_handler", new_code)
result = app.run("my_handler")  # 立即使用新代碼
```

#### 10.1.2 數據管理

| 特性 | 文件 + 數據庫 | ORM 框架 | ASMdynamicGradio |
|------|-------------|----------|------------------|
| **多格式支持** | ⚠️ 需適配 | ⚠️ 限 SQL | ✅ JSON/NumPy/二進制 |
| **自動壓縮** | ❌ | ❌ | ✅ 可選壓縮 |
| **向量索引** | ❌ | ❌ | ✅ HDC 向量 |
| **統一 API** | ❌ | ⚠️ | ✅ 一致接口 |

```python
# 傳統方式：不同類型需要不同處理
import json
import numpy as np
import pickle

# 保存 JSON
with open("config.json", "w") as f:
    json.dump(config, f)

# 保存 NumPy
np.save("data.npy", array)

# 保存任意對象
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# ASMdynamicGradio：統一接口
app.saveData("config", config)      # 自動識別為 JSON
app.saveData("data", array)          # 自動識別為 NumPy
app.saveData("model", model)         # 自動序列化
```

#### 10.1.3 搜索能力

| 特性 | 文件名搜索 | 全文搜索引擎 | ASMdynamicGradio |
|------|-----------|-------------|------------------|
| **模糊匹配** | ⚠️ 有限 | ✅ | ✅ |
| **語義搜索** | ❌ | ⚠️ 需配置 | ✅ 內建 |
| **蒙特卡羅** | ❌ | ❌ | ✅ 獨特優勢 |
| **即時索引** | ❌ | ⚠️ 需重建 | ✅ 自動 |

### 10.2 開發效率提升

#### 10.2.1 快速原型開發

```python
# 場景：快速實現一個數據處理管道

with DynamicApp("./prototype") as app:
    # 5 分鐘內完成原型
    
    # 第 1 步：定義數據清洗邏輯
    app.saveCode("cleaner", """
def main(data):
    # 移除空值
    data = [x for x in data if x is not None]
    # 去重
    data = list(set(data))
    return sorted(data)
""")
    
    # 第 2 步：定義轉換邏輯
    app.saveCode("transformer", """
def main(data):
    return [x * 2 + 1 for x in data]
""")
    
    # 第 3 步：定義聚合邏輯
    app.saveCode("aggregator", """
import statistics

def main(data):
    return {
        "count": len(data),
        "sum": sum(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data)
    }
""")
    
    # 第 4 步：組合執行
    raw_data = [1, 2, 2, None, 3, 4, None, 5]
    
    cleaned = app.run("cleaner", kwargs={"data": raw_data}).result
    transformed = app.run("transformer", kwargs={"data": cleaned}).result
    result = app.run("aggregator", kwargs={"data": transformed}).result
    
    print(result)
    # {'count': 5, 'sum': 35, 'mean': 7.0, 'median': 7}
    
    # 需要修改？直接更新，無需重啟
    app.updateCode("transformer", """
def main(data):
    return [x ** 2 for x in data]  # 改為平方
""")
    
    # 重新執行
    transformed = app.run("transformer", kwargs={"data": cleaned}).result
    result = app.run("aggregator", kwargs={"data": transformed}).result
    print(result)
    # {'count': 5, 'sum': 55, 'mean': 11.0, 'median': 9}
```

#### 10.2.2 A/B 測試

```python
# 場景：同時測試多個算法版本

with DynamicApp("./ab_test") as app:
    # 版本 A
    app.saveCode("algorithm_v1", """
def main(x):
    return x * 2
""")
    
    # 版本 B
    app.saveCode("algorithm_v2", """
def main(x):
    return x ** 2
""")
    
    # 版本 C
    app.saveCode("algorithm_v3", """
import math
def main(x):
    return math.log(x + 1) * 10
""")
    
    # 對比測試
    test_data = [1, 5, 10, 50, 100]
    
    for version in ["algorithm_v1", "algorithm_v2", "algorithm_v3"]:
        results = []
        for x in test_data:
            result = app.run(version, kwargs={"x": x})
            results.append(result.result)
        
        print(f"{version}: {results}")
    
    # 動態選擇最佳版本
    def select_algorithm(condition):
        if condition == "linear":
            return "algorithm_v1"
        elif condition == "quadratic":
            return "algorithm_v2"
        else:
            return "algorithm_v3"
    
    # 運行時切換
    algo = select_algorithm("quadratic")
    result = app.run(algo, kwargs={"x": 10})
    print(f"使用 {algo}: {result.result}")
```

#### 10.2.3 調試與問題排查

```python
with DynamicApp("./debug_demo") as app:
    # 保存可能有問題的代碼
    app.saveCode("buggy_code", """
def main(data):
    total = 0
    for item in data:
        total += item["value"]  # 可能 KeyError
    return total
""")
    
    # 測試正常情況
    result = app.run("buggy_code", kwargs={
        "data": [{"value": 1}, {"value": 2}]
    })
    print(f"正常: {result.result}")  # 3
    
    # 測試異常情況
    result = app.run("buggy_code", kwargs={
        "data": [{"value": 1}, {"amount": 2}]  # 缺少 value
    })
    
    if not result.success:
        print(f"錯誤: {result.error}")
        print(f"詳情: {result.stderr}")
    
    # 查看調試日誌
    logs = app.get_debug_log(limit=10)
    for log in logs:
        print(f"[{log['level']}] {log['message']}")
    
    # 修復代碼
    app.updateCode("buggy_code", """
def main(data):
    total = 0
    for item in data:
        total += item.get("value", 0)  # 使用 get 避免 KeyError
    return total
""")
    
    # 驗證修復
    result = app.run("buggy_code", kwargs={
        "data": [{"value": 1}, {"amount": 2}]
    })
    print(f"修復後: {result.result}")  # 1
```

### 10.3 核心優勢總結

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASMdynamicGradio 核心優勢                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  🚀 動態執行                                                         │
│     • 運行時加載、更新、執行代碼                                      │
│     • 無需重啟即可生效                                                │
│     • 支持熱更新和熱修復                                              │
│                                                                      │
│  📦 統一存儲                                                         │
│     • 代碼、數據、知識統一管理                                        │
│     • 多格式自動處理（JSON/NumPy/二進制）                             │
│     • 內建壓縮和向量索引                                              │
│                                                                      │
│  🔍 智能搜索                                                         │
│     • 五種搜索模式（精確/模糊/正則/語義/蒙特卡羅）                     │
│     • HDC 向量編碼實現語義理解                                        │
│     • 蒙特卡羅搜索帶來探索性結果                                      │
│                                                                      │
│  🌳 命名空間隔離                                                     │
│     • 層級化的項目組織                                                │
│     • 多環境/多租戶支持                                               │
│     • 靈活的導入導出                                                  │
│                                                                      │
│  🌐 Web 介面                                                         │
│     • 現代化 Gradio 界面                                              │
│     • 語法高亮代碼編輯                                                │
│     • 實時執行和調試                                                  │
│                                                                      │
│  🌀 演化可視化                                                       │
│     • 細胞自動機引擎                                                  │
│     • 多種演化規則                                                    │
│     • PNG/GIF/MP4 輸出                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 11. 最佳實踐與設計模式

### 11.1 項目結構規範

```python
def setup_project_structure(app: DynamicApp, project_name: str):
    """建立標準項目結構"""
    
    # 頂層項目命名空間
    app.createNamespace(project_name, f"Project: {project_name}")
    
    # 標準子命名空間
    structure = {
        "core": "核心業務邏輯",
        "utils": "工具函數",
        "models": "數據模型",
        "handlers": "請求處理器",
        "tasks": "後台任務",
        "tests": "測試代碼",
        "configs": "配置數據",
        "docs": "文檔知識"
    }
    
    for name, desc in structure.items():
        app.createNamespace(
            f"{project_name}_{name}",
            description=desc,
            parent=project_name
        )
    
    # 創建項目說明
    app.saveKnowledge(
        "README",
        f"""
# {project_name}

## 項目結構

- `core/`: 核心業務邏輯
- `utils/`: 工具函數
- `models/`: 數據模型
- `handlers/`: 請求處理器
- `tasks/`: 後台任務
- `tests/`: 測試代碼
- `configs/`: 配置數據
- `docs/`: 文檔知識

## 快速開始

```python
from ASMdynamicGradio import DynamicApp

with DynamicApp("./project_data") as app:
    # 導入核心模組
    core = app.importCode("main", "{project_name}_core")
    
    # 執行主函數
    result = core.run()
```
        """,
        namespace=f"{project_name}_docs",
        tags=["readme", "documentation"]
    )
    
    return structure

# 使用
with DynamicApp("./my_project") as app:
    structure = setup_project_structure(app, "my_app")
    print("項目結構已創建:", list(structure.keys()))
```

### 11.2 錯誤處理模式

```python
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

@dataclass
class OperationResult:
    """操作結果封裝"""
    success: bool
    data: Optional[any] = None
    error: Optional[str] = None
    details: Optional[dict] = None

@contextmanager
def safe_operation(app: DynamicApp, operation_name: str):
    """安全操作上下文"""
    import time
    
    start_time = time.time()
    result = OperationResult(success=True)
    
    try:
        yield result
    except Exception as e:
        result.success = False
        result.error = str(e)
        result.details = {"traceback": traceback.format_exc()}
        
        # 記錄錯誤
        app._log_debug("ERROR", f"{operation_name} 失敗: {e}", exc_info=True)
    finally:
        duration = time.time() - start_time
        
        # 記錄操作日誌
        app.saveData(
            f"op_log_{int(time.time() * 1000)}",
            {
                "operation": operation_name,
                "success": result.success,
                "duration_ms": duration * 1000,
                "error": result.error,
                "timestamp": datetime.now().isoformat()
            },
            namespace="system",
            metadata={"type": "operation_log"}
        )

# 使用
with DynamicApp("./app") as app:
    with safe_operation(app, "save_critical_data") as result:
        app.saveData("critical", {"important": "data"})
        result.data = "保存成功"
    
    if result.success:
        print(result.data)
    else:
        print(f"操作失敗: {result.error}")
```

### 11.3 測試模式

```python
class DynamicTestRunner:
    """動態測試運行器"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self.app.createNamespace("tests", "測試套件")
        self.results = []
    
    def add_test(self, name: str, test_code: str):
        """添加測試"""
        wrapped_code = f"""
def test():
    try:
        # 用戶測試代碼
{chr(10).join('        ' + line for line in test_code.split(chr(10)))}
        return {{"passed": True}}
    except AssertionError as e:
        return {{"passed": False, "error": str(e)}}
    except Exception as e:
        return {{"passed": False, "error": f"Unexpected error: {{e}}"}}
"""
        self.app.saveCode(name, wrapped_code, "tests")
    
    def run_all(self) -> dict:
        """運行所有測試"""
        self.results = []
        
        tests = self.app.listNodes(content_type="code", namespace="tests")
        
        for test in tests:
            result = self.app.run(test["name"], "tests", entry_point="test")
            
            test_result = {
                "name": test["name"],
                "passed": False,
                "error": None,
                "time_ms": result.execution_time_ms
            }
            
            if result.success and isinstance(result.result, dict):
                test_result["passed"] = result.result.get("passed", False)
                test_result["error"] = result.result.get("error")
            else:
                test_result["error"] = result.error
            
            self.results.append(test_result)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        
        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "results": self.results
        }
    
    def report(self) -> str:
        """生成測試報告"""
        report = "# 測試報告\n\n"
        
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        
        report += f"**結果**: {passed}/{total} 通過\n\n"
        report += "## 詳細結果\n\n"
        
        for r in self.results:
            icon = "✅" if r["passed"] else "❌"
            report += f"- {icon} `{r['name']}` ({r['time_ms']:.2f}ms)"
            if r["error"]:
                report += f"\n  - 錯誤: {r['error']}"
            report += "\n"
        
        return report

# 使用
with DynamicApp("./test_app") as app:
    runner = DynamicTestRunner(app)
    
    # 添加測試
    runner.add_test("test_addition", """
result = 1 + 1
assert result == 2, f"Expected 2, got {result}"
""")
    
    runner.add_test("test_string", """
s = "hello"
assert len(s) == 5
assert s.upper() == "HELLO"
""")
    
    runner.add_test("test_failing", """
assert 1 == 2, "This should fail"
""")
    
    # 運行測試
    summary = runner.run_all()
    print(f"通過: {summary['passed']}/{summary['total']}")
    
    # 生成報告
    report = runner.report()
    print(report)
```

### 11.4 性能優化

```python
# 1. 批量操作
def batch_save_codes(app: DynamicApp, codes: dict, namespace: str):
    """批量保存代碼"""
    nodes = []
    for name, code in codes.items():
        node = app.saveCode(name, code, namespace)
        nodes.append(node)
    return nodes

# 2. 延遲加載
class LazyModule:
    """延遲加載的模組"""
    
    def __init__(self, app: DynamicApp, name: str, namespace: str):
        self._app = app
        self._name = name
        self._namespace = namespace
        self._module = None
    
    def __getattr__(self, attr):
        if self._module is None:
            self._module = self._app.importCode(self._name, self._namespace)
        return getattr(self._module, attr)

# 3. 結果緩存
from functools import lru_cache

class CachedRunner:
    """帶緩存的執行器"""
    
    def __init__(self, app: DynamicApp):
        self.app = app
        self._cache = {}
    
    def run_cached(self, name: str, namespace: str, **kwargs):
        """緩存執行結果"""
        # 生成緩存鍵
        cache_key = f"{namespace}.{name}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self.app.run(name, namespace, kwargs=kwargs)
        
        if result.success:
            self._cache[cache_key] = result
        
        return result
    
    def invalidate(self, pattern: str = None):
        """清除緩存"""
        if pattern is None:
            self._cache.clear()
        else:
            self._cache = {
                k: v for k, v in self._cache.items()
                if pattern not in k
            }

# 4. 壓縮大數據
def save_large_data(app: DynamicApp, name: str, data, namespace: str):
    """自動壓縮大數據"""
    import sys
    
    size = sys.getsizeof(data)
    compression = size > 10 * 1024  # 超過 10KB 啟用壓縮
    
    return app.saveData(
        name, data, namespace,
        compression=compression
    )
```

---

## 快速參考卡

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASMdynamicGradio 快速參考                         │
├─────────────────────────────────────────────────────────────────────┤
│ CLI 命令                                                             │
│   python ASMdynamicGradio.py               啟動 Web 介面            │
│   python ASMdynamicGradio.py --test        運行測試                 │
│   python ASMdynamicGradio.py --port 8080   自定義端口               │
│   python ASMdynamicGradio.py --share       公共分享                 │
├─────────────────────────────────────────────────────────────────────┤
│ 代碼管理                                                             │
│   app.saveCode(name, code, ns)             保存代碼                 │
│   app.getCode(name, ns)                    獲取代碼                 │
│   app.updateCode(name, code, ns)           更新代碼                 │
│   app.deleteCode(name, ns)                 刪除代碼                 │
│   app.importCode(name, ns)                 導入為模組               │
│   app.run(name, ns, entry_point, kwargs)   執行代碼                 │
├─────────────────────────────────────────────────────────────────────┤
│ 數據管理                                                             │
│   app.saveData(name, data, ns)             保存數據                 │
│   app.getData(name, ns)                    獲取數據                 │
│   app.deleteData(name, ns)                 刪除數據                 │
├─────────────────────────────────────────────────────────────────────┤
│ 知識管理                                                             │
│   app.saveKnowledge(name, content, ns)     保存知識                 │
│   app.getKnowledge(name, ns)               獲取知識                 │
│   app.updateKnowledge(name, content, ns)   更新知識                 │
├─────────────────────────────────────────────────────────────────────┤
│ 文件管理                                                             │
│   app.addFile(name, data, ns)              添加文件                 │
│   app.getFile(name, ns)                    獲取文件                 │
│   app.getFileInfo(name, ns)                獲取文件信息             │
│   app.deleteFile(name, ns)                 刪除文件                 │
├─────────────────────────────────────────────────────────────────────┤
│ 搜索功能                                                             │
│   app.search(query, mode="fuzzy")          模糊搜索                 │
│   app.search(query, mode="exact")          精確搜索                 │
│   app.search(query, mode="regex")          正則搜索                 │
│   app.search(query, mode="semantic")       語義搜索                 │
│   app.search(query, mode="monte_carlo")    蒙特卡羅搜索             │
├─────────────────────────────────────────────────────────────────────┤
│ 命名空間                                                             │
│   app.createNamespace(name, desc, parent)  創建命名空間             │
│   app.listNamespaces()                     列出命名空間             │
│   app.getNamespace(name)                   獲取命名空間信息         │
├─────────────────────────────────────────────────────────────────────┤
│ 節點管理                                                             │
│   app.listNodes(content_type, ns)          列出節點                 │
│   app.getNode(name, ns)                    獲取節點詳情             │
├─────────────────────────────────────────────────────────────────────┤
│ 導入導出                                                             │
│   app.fromFolder(path, ns)                 從文件夾導入             │
│   app.toFolder(path, ns)                   導出到文件夾             │
│   app.fromFileDict(storage, ns)            從 FileDict 導入         │
│   app.toFileDict(storage, ns)              導出到 FileDict          │
├─────────────────────────────────────────────────────────────────────┤
│ 演化可視化                                                           │
│   app.initEvolution(mode)                  初始化演化               │
│   app.evolve(steps, rule, interval)        執行演化                 │
│   app.getEvolutionState()                  獲取演化狀態             │
│   app.saveVisualization(prefix, effect)    保存可視化               │
├─────────────────────────────────────────────────────────────────────┤
│ 系統管理                                                             │
│   app.getStats()                           獲取統計                 │
│   app.getSystemInfo()                      獲取系統信息             │
│   app.get_debug_log(limit)                 獲取調試日誌             │
│   app.clear_debug_log()                    清空調試日誌             │
│   app.close()                              關閉應用                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 結語

ASMdynamicGradio 提供了一個完整的動態知識與工具體系解決方案，通過統一的 API 實現了：

1. **代碼的動態管理與執行** - 無需重啟即可更新邏輯
2. **多格式數據的統一存儲** - 一致的接口處理各種數據類型
3. **智能的搜索與發現** - 蒙特卡羅搜索帶來驚喜性結果
4. **靈活的命名空間隔離** - 支持複雜的項目組織結構
5. **直觀的 Web 介面** - 降低使用門檻

這套系統特別適合：
- 快速原型開發
- 機器學習實驗管理
- 動態配置管理
- 知識庫構建
- 插件式架構

希望這份指南能幫助您充分利用 ASMdynamicGradio 的強大功能！