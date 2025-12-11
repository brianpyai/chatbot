
## 1. 系統概述

ASMdynamicGradio 是一個動態知識與工具體系，提供以下核心能力：

| 功能模組 | 說明 |
|---------|------|
| **動態代碼管理** | 保存、載入、更新、執行 Python 代碼 |
| **命名空間隔離** | 多項目、多環境的內容隔離管理 |
| **資料組管理** | 多類型內容的統一組織與校準 |
| **Embedding 校準** | 跨類型內容的語義關聯映射 |
| **蒙特卡羅搜索** | 動態採樣的智能搜索引擎 |
| **演化可視化** | 元胞自動機驅動的動態可視化 |

---

## 2. 安裝與初始化

### 2.1 基本初始化

```python
from ASMdynamicGradio import DynamicApp

# 基本初始化
app = DynamicApp(
    storage_dir="./my_project_data",   # 存儲目錄
    namespace="main",                   # 默認命名空間
    auto_load=True                      # 自動載入已有數據
)

# 使用上下文管理器（推薦）
with DynamicApp("./data", namespace="dev") as app:
    # 在此執行操作
    app.saveCode("hello", "def main(): return 'Hello!'")
    result = app.run("hello")
    print(result.result)  # 輸出: Hello!
# 自動清理資源
```

### 2.2 檢查系統狀態

```python
# 獲取系統信息
info = app.getSystemInfo()
print(f"存儲目錄: {info['storage_dir']}")
print(f"命名空間列表: {info['namespaces']}")
print(f"統計數據: {info['stats']}")

# 獲取詳細統計
stats = app.getStats()
print(f"代碼節點數: {stats['code_nodes']}")
print(f"數據節點數: {stats['data_nodes']}")
print(f"資料組數: {stats['data_groups']}")
```

---

## 3. 代碼管理與動態執行

### 3.1 保存與獲取代碼

```python
# 保存代碼
code = """
import math

def calculate_circle_area(radius):
    '''計算圓面積'''
    return math.pi * radius ** 2

def main(radius=5):
    '''主入口函數'''
    area = calculate_circle_area(radius)
    return {
        "radius": radius,
        "area": round(area, 4),
        "circumference": round(2 * math.pi * radius, 4)
    }
"""

node = app.saveCode(
    name="circle_calculator",
    code=code,
    namespace="math_tools",
    metadata={"author": "developer", "version": "1.0"}
)
print(f"已保存: {node.node_id}")

# 獲取代碼
loaded_code = app.getCode("circle_calculator", namespace="math_tools")
print(loaded_code)

# 更新代碼
updated_code = code.replace("1.0", "1.1")
app.updateCode("circle_calculator", updated_code, namespace="math_tools")
```

### 3.2 動態執行代碼

```python
# 基本執行
result = app.run("circle_calculator", namespace="math_tools")

if result.success:
    print(f"執行成功！")
    print(f"結果: {result.result}")
    print(f"執行時間: {result.execution_time_ms:.2f} ms")
else:
    print(f"執行失敗: {result.error}")
    print(f"詳細錯誤:\n{result.stderr}")

# 帶參數執行
result = app.run(
    name="circle_calculator",
    namespace="math_tools",
    entry_point="main",
    kwargs={"radius": 10}
)
print(result.result)  # {'radius': 10, 'area': 314.1593, 'circumference': 62.8319}

# 執行不同入口函數
result = app.run(
    name="circle_calculator",
    namespace="math_tools",
    entry_point="calculate_circle_area",
    args=(7,)
)
print(f"半徑7的圓面積: {result.result}")
```

### 3.3 動態導入模組

```python
# 導入為模組使用
circle_module = app.importCode("circle_calculator", namespace="math_tools")

# 直接調用模組函數
area = circle_module.calculate_circle_area(15)
print(f"面積: {area}")

# 在其他代碼中使用導入的模組
integration_code = """
def main():
    # 使用全局注入的模組
    results = []
    for r in [1, 2, 3, 4, 5]:
        results.append(circle.calculate_circle_area(r))
    return results
"""

app.saveCode("batch_calculate", integration_code, namespace="math_tools")
result = app.run(
    "batch_calculate",
    namespace="math_tools",
    globals_dict={"circle": circle_module}  # 注入模組
)
print(result.result)  # [3.14..., 12.56..., 28.27..., 50.26..., 78.54...]
```

### 3.4 代碼管理操作

```python
# 列出所有代碼節點
nodes = app.listNodes(content_type="code", namespace="math_tools")
for node in nodes:
    print(f"{node['namespace']}.{node['name']} - 類型: {node['type']}")

# 獲取節點詳情
detail = app.getNode("circle_calculator", namespace="math_tools")
print(f"創建時間: {detail['created']}")
print(f"修改時間: {detail['modified']}")
print(f"元數據: {detail['metadata']}")

# 刪除代碼
app.deleteCode("circle_calculator", namespace="math_tools")
```

---

## 4. 命名空間管理

### 4.1 創建與組織命名空間

```python
# 創建獨立命名空間
app.createNamespace(
    name="production",
    description="生產環境代碼",
    parent=None
)

# 創建層級命名空間
app.createNamespace("development", description="開發環境")
app.createNamespace("feature_auth", description="認證模組", parent="development")
app.createNamespace("feature_api", description="API模組", parent="development")

# 列出所有命名空間
namespaces = app.listNamespaces()
print(f"命名空間: {namespaces}")

# 獲取命名空間詳情
ns_info = app.getNamespace("development")
print(f"子命名空間: {ns_info.children}")
print(f"節點數量: {len(ns_info.nodes)}")
```

### 4.2 複製命名空間（能力分發）

```python
# 場景：將開發環境的代碼分發到測試環境
app.saveCode("auth_handler", "def verify(): return True", "development")
app.saveCode("api_router", "def route(): return '/api'", "development")
app.saveData("config", {"debug": True}, "development")

# 複製整個命名空間
copied = app.copyNamespace(
    source="development",
    target="testing",
    include_codes=True,
    include_data=True,
    include_groups=True
)

print(f"複製統計:")
print(f"  代碼: {copied['codes']} 個")
print(f"  數據: {copied['data']} 個")
print(f"  資料組: {copied['groups']} 個")

# 驗證複製結果
testing_code = app.getCode("auth_handler", namespace="testing")
assert testing_code is not None  # 成功複製
```

### 4.3 重命名與刪除命名空間

```python
# 重命名命名空間
app.renameNamespace("testing", "staging")

# 安全刪除（非空時會報錯）
try:
    app.deleteNamespace("staging", force=False)
except ValueError as e:
    print(f"無法刪除: {e}")

# 強制刪除（包括所有內容）
app.deleteNamespace("staging", force=True)
print("命名空間及所有內容已刪除")
```

---

## 5. 資料組管理與 Embedding 校準

### 5.1 創建資料組

資料組是將多種類型內容（文字、代碼、圖片、HTML等）統一管理的容器。

```python
# 創建包含多類型內容的資料組
items = [
    # (content, type, metadata)
    ("Python 是一種高級編程語言", "text", {"lang": "zh"}),
    ("def hello(): return 'Hello Python!'", "code", {"language": "python"}),
    ("# Python 入門指南\n\n這是一份 Python 學習資料...", "markdown", None),
    ("<div class='tip'>Python 小技巧</div>", "html", None),
    ('{"framework": "Django", "version": "4.2"}', "json", {"category": "web"})
]

group = app.createDataGroup(
    name="python_learning",
    items=items,
    namespace="knowledge_base",
    description="Python 學習資料集",
    tags=["python", "programming", "tutorial"],
    calibrate=True  # 啟用 Embedding 校準
)

print(f"資料組 ID: {group.group_id}")
print(f"項目數量: {len(group.items)}")
print(f"已校準: {group.calibrated}")
print(f"統一向量存在: {group.unified_vector is not None}")
```

### 5.2 向資料組添加內容

```python
# 添加單個項目
app.addToDataGroup(
    name="python_learning",
    content="Python 的 list comprehension 非常強大",
    item_type="text",
    metadata={"topic": "advanced"},
    namespace="knowledge_base",
    recalibrate=True  # 重新校準整個資料組
)

# 添加代碼範例
app.addToDataGroup(
    name="python_learning",
    content="squares = [x**2 for x in range(10)]",
    item_type="code",
    namespace="knowledge_base"
)

# 獲取更新後的資料組
group = app.getDataGroup("python_learning", namespace="knowledge_base")
print(f"更新後項目數: {len(group.items)}")
```

### 5.3 更新與刪除資料組

```python
# 更新資料組元信息
app.updateDataGroup(
    name="python_learning",
    namespace="knowledge_base",
    description="更新後的 Python 學習資料集",
    tags=["python", "programming", "tutorial", "advanced"]
)

# 完全替換內容
new_items = [
    ("新的內容1", "text", None),
    ("新的內容2", "markdown", None)
]
app.updateDataGroup(
    name="python_learning",
    items=new_items,
    namespace="knowledge_base",
    calibrate=True
)

# 刪除資料組
app.deleteDataGroup("python_learning", namespace="knowledge_base")
```

### 5.4 Embedding 校準詳解

Embedding 校準將不同類型的內容映射到同一語義空間，使相關內容具有更高的相似度。

```python
# 批量導入並校準
items = [
    ("user_auth", "def authenticate(user, pwd): ...", "code", {"module": "auth"}),
    ("auth_doc", "# 用戶認證\n\n本模組處理用戶認證流程...", "text", {"type": "doc"}),
    ("auth_config", {"secret_key": "xxx", "expire": 3600}, "data", None),
    ("login_page", "<form id='login'>...</form>", "html", {"ui": True})
]

# 方式1：作為獨立節點導入（帶校準）
nodes = app.batchImportWithCalibration(
    items=items,
    namespace="auth_module",
    as_data_group=False
)
print(f"導入 {len(nodes)} 個節點")

# 方式2：作為資料組導入（帶校準）
group = app.batchImportWithCalibration(
    items=items,
    namespace="auth_module",
    as_data_group=True,
    group_name="auth_bundle"
)
print(f"創建資料組: {group.name}，包含 {len(group.items)} 項")

# 校準整個命名空間
result = app.calibrateNamespace("auth_module")
print(f"校準了 {result['calibrated']} 個節點")
```

---

## 6. 知識管理

### 6.1 保存與獲取知識條目

```python
# 保存 Markdown 格式的知識
app.saveKnowledge(
    name="design_patterns_intro",
    content="""
# 設計模式簡介

設計模式是軟件開發中常見問題的解決方案模板。

## 創建型模式
- 單例模式
- 工廠模式
- 建造者模式

## 結構型模式
- 適配器模式
- 裝飾器模式
- 代理模式

## 行為型模式
- 觀察者模式
- 策略模式
- 命令模式
    """,
    namespace="knowledge_base",
    tags=["design-patterns", "software-engineering", "tutorial"],
    attachments=["singleton_example.py", "factory_example.py"],
    metadata={"difficulty": "intermediate", "author": "tech_team"}
)

# 獲取知識條目
knowledge = app.getKnowledge("design_patterns_intro", namespace="knowledge_base")
print(f"內容長度: {len(knowledge['content'])} 字符")
print(f"標籤: {knowledge['tags']}")
print(f"創建時間: {knowledge['created']}")
```

### 6.2 更新知識條目

```python
# 追加新內容
original = app.getKnowledge("design_patterns_intro", namespace="knowledge_base")
updated_content = original['content'] + """

## 新增：SOLID 原則

1. 單一職責原則
2. 開放封閉原則
3. 里氏替換原則
4. 接口隔離原則
5. 依賴反轉原則
"""

app.updateKnowledge(
    name="design_patterns_intro",
    content=updated_content,
    namespace="knowledge_base",
    tags=original['tags'] + ["solid-principles"]
)
```

---

## 7. 高效搜索與查詢

### 7.1 基本搜索

```python
# 模糊搜索（默認）
results = app.search(
    query="Python 函數",
    mode="fuzzy",
    content_type="all",
    result_limit=10
)

for r in results:
    print(f"[{r.score:.2%}] {r.namespace}.{r.name} ({r.content_type})")
    print(f"  預覽: {r.preview[:100]}...")
    print()
```

### 7.2 搜索模式詳解

```python
# 精確匹配
results = app.search(
    query="def main():",
    mode="exact",
    content_type="code"
)

# 正則表達式搜索
results = app.search(
    query=r"def \w+\(self",  # 搜索類方法
    mode="regex",
    content_type="code"
)

# 語義搜索
results = app.search(
    query="用戶登錄驗證",
    mode="semantic",
    similarity_threshold=0.5,
    content_type="all"
)

# 蒙特卡羅搜索（帶隨機探索）
results = app.search(
    query="數據處理",
    mode="monte_carlo",
    monte_carlo_samples=100,  # 採樣數
    similarity_threshold=0.3,
    result_limit=20
)

# 關聯搜索（包含關聯項目）
results = app.search(
    query="認證模組",
    mode="related",
    similarity_threshold=0.5
)

for r in results:
    print(f"{r.name} - 關聯項目: {r.related_items}")
```

### 7.3 跨類型關聯搜索

```python
# 跨多種內容類型搜索
results = app.searchRelated(
    query="用戶管理系統",
    content_types=["code", "knowledge", "data_group"],
    namespace=None,  # 搜索所有命名空間
    threshold=0.5,
    limit=20
)

for r in results:
    print(f"[{r.score:.2%}] {r.name} ({r.content_type})")
    if r.related_items:
        print(f"  關聯: {', '.join(r.related_items[:3])}")
```

### 7.4 高級搜索選項

```python
# 完整搜索選項
results = app.search(
    query="機器學習模型",
    mode="monte_carlo",
    content_type="all",
    namespace="ml_project",       # 限定命名空間
    similarity_threshold=0.4,     # 相似度閾值
    monte_carlo_samples=150,      # 蒙特卡羅採樣數
    fast_match_limit=1000,        # 快速匹配候選數
    result_limit=30,              # 結果數量限制
    include_content=True,         # 包含內容預覽
    include_data_groups=True      # 包含資料組
)

# 處理搜索結果
search_report = []
for r in results:
    search_report.append({
        "id": r.node_id,
        "name": r.name,
        "namespace": r.namespace,
        "type": r.content_type,
        "score": r.score,
        "preview": r.preview[:200],
        "related": r.related_items
    })

# 保存搜索結果
app.saveData(
    "search_result_ml",
    {"query": "機器學習模型", "results": search_report},
    namespace="search_history"
)
```

---

## 8. 批量導入與能力分發

### 8.1 從文件夾導入

```python
# 從項目文件夾導入
nodes = app.fromFolder(
    folder_path="./my_python_project",
    namespace="imported_project",
    recursive=True,               # 遞歸導入子目錄
    calibrate=True,               # 啟用 Embedding 校準
    file_patterns=["*.py", "*.json", "*.md"]  # 文件過濾
)

print(f"導入了 {len(nodes)} 個節點")

# 查看導入結果
for node in nodes[:5]:
    print(f"  - {node.name} ({node.node_type})")
```

### 8.2 導出到文件夾

```python
# 導出命名空間到文件夾
count = app.toFolder(
    folder_path="./export/production_backup",
    namespace="production",
    include_codes=True,
    include_data=True
)

print(f"導出了 {count} 個節點")
```

### 8.3 FileDict 互操作

```python
from ASMFileDict3 import FileDict

# 從外部 FileDict 導入
external_storage = FileDict("./external_data")
nodes = app.fromFileDict(
    source_storage=external_storage,
    namespace="external_import"
)

# 導出到外部 FileDict
target_storage = FileDict("./backup_storage")
count = app.toFileDict(
    target_storage=target_storage,
    namespace="production"
)
```

### 8.4 能力分發工作流

```python
def distribute_capabilities(app, source_ns, target_namespaces):
    """
    將源命名空間的能力分發到多個目標命名空間
    """
    results = {}
    
    for target in target_namespaces:
        # 創建目標命名空間
        app.createNamespace(target, description=f"分發自 {source_ns}")
        
        # 複製內容
        copied = app.copyNamespace(source_ns, target)
        results[target] = copied
        
        # 校準目標命名空間
        app.calibrateNamespace(target)
    
    return results

# 使用範例
distribution = distribute_capabilities(
    app,
    source_ns="core_lib",
    target_namespaces=["project_a", "project_b", "project_c"]
)

for ns, stats in distribution.items():
    print(f"{ns}: 代碼 {stats['codes']}, 數據 {stats['data']}, 資料組 {stats['groups']}")
```

---

## 9. 演化可視化

### 9.1 初始化與演化

```python
# 初始化演化狀態
app.initEvolution(mode="random")  # 可選: random, center, gradient, noise

# 獲取初始狀態
state = app.getEvolutionState()
print(f"狀態形狀: {state.shape}")  # (128, 128)
print(f"值範圍: [{state.min():.4f}, {state.max():.4f}]")

# 執行演化
frames = app.evolve(
    steps=100,
    rule="diffusion",        # 可選: diffusion, conway, wave, growth, erosion
    record_interval=10       # 每 10 步記錄一幀
)

print(f"記錄了 {len(frames)} 幀")

# 分析最終幀
final_frame = frames[-1]
print(f"第 {final_frame.step} 步:")
print(f"  平均值: {final_frame.metrics['mean']:.4f}")
print(f"  標準差: {final_frame.metrics['std']:.4f}")
print(f"  活躍率: {final_frame.metrics['active_ratio']:.2%}")
```

### 9.2 演化規則說明

```python
# 不同演化規則的效果
rules = {
    "diffusion": "擴散規則 - 平滑過渡效果",
    "conway": "康威生命遊戲 - 細胞自動機",
    "wave": "波動規則 - 週期性振盪",
    "growth": "生長規則 - 逐漸擴展",
    "erosion": "侵蝕規則 - 逐漸消退"
}

for rule, desc in rules.items():
    app.initEvolution(mode="random")
    frames = app.evolve(steps=50, rule=rule, record_interval=50)
    final = frames[-1]
    print(f"{rule}: {desc}")
    print(f"  最終狀態 - 平均: {final.metrics['mean']:.4f}, 活躍率: {final.metrics['active_ratio']:.2%}")
```

### 9.3 可視化輸出

```python
# 保存可視化結果
outputs = app.saveVisualization(
    prefix="evolution_demo",
    effect="glow"  # 可選: none, glow, wave, fire, plasma, fractal
)

print(f"輸出文件:")
for name, path in outputs.items():
    print(f"  {name}: {path}")
```

---

## 10. 經驗回收與統計分析

### 10.1 調試日誌管理

```python
# 獲取調試日誌
logs = app.get_debug_log(limit=50)

for log in logs[-10:]:
    print(f"[{log['timestamp']}] [{log['level']}] {log['message']}")
    if log.get('details'):
        print(f"  詳情: {log['details']}")

# 清空調試日誌
app.clear_debug_log()
```

### 10.2 執行經驗回收

```python
def collect_execution_experience(app, namespace):
    """
    收集執行經驗並生成報告
    """
    experience = {
        "namespace": namespace,
        "timestamp": datetime.now().isoformat(),
        "executions": [],
        "success_rate": 0,
        "avg_execution_time": 0
    }
    
    # 獲取所有代碼節點
    code_nodes = app.listNodes(content_type="code", namespace=namespace)
    
    total_time = 0
    success_count = 0
    
    for node in code_nodes:
        # 測試執行
        result = app.run(node["name"], namespace=namespace)
        
        experience["executions"].append({
            "name": node["name"],
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "error": result.error if not result.success else None
        })
        
        if result.success:
            success_count += 1
            total_time += result.execution_time_ms
    
    if code_nodes:
        experience["success_rate"] = success_count / len(code_nodes)
        if success_count > 0:
            experience["avg_execution_time"] = total_time / success_count
    
    return experience

# 使用範例
from datetime import datetime

experience = collect_execution_experience(app, "production")
print(f"成功率: {experience['success_rate']:.1%}")
print(f"平均執行時間: {experience['avg_execution_time']:.2f} ms")

# 保存經驗報告
app.saveData(
    f"experience_report_{datetime.now().strftime('%Y%m%d')}",
    experience,
    namespace="analytics"
)
```

### 10.3 知識圖譜分析

```python
def build_knowledge_graph(app, namespace):
    """
    構建命名空間的知識圖譜
    """
    graph = {
        "nodes": [],
        "edges": []
    }
    
    # 收集所有節點
    all_nodes = app.listNodes(namespace=namespace)
    
    for node in all_nodes:
        graph["nodes"].append({
            "id": node["id"],
            "name": node["name"],
            "type": node["type"],
            "metadata": node.get("metadata", {})
        })
    
    # 使用關聯搜索建立邊
    for node in all_nodes:
        results = app.searchRelated(
            query=node["name"],
            namespace=namespace,
            threshold=0.6,
            limit=5
        )
        
        for r in results:
            if r.node_id != node["id"]:
                graph["edges"].append({
                    "source": node["id"],
                    "target": r.node_id,
                    "weight": r.score
                })
    
    return graph

# 構建並保存知識圖譜
kg = build_knowledge_graph(app, "knowledge_base")
app.saveData("knowledge_graph", kg, namespace="analytics")

print(f"節點數: {len(kg['nodes'])}")
print(f"邊數: {len(kg['edges'])}")
```

---

## 11. 完整工作流程範例

### 11.1 項目初始化與開發流程

```python
#!/usr/bin/env python3
"""
完整項目工作流程範例
"""

from ASMdynamicGradio import DynamicApp
from datetime import datetime
import json

def setup_project(app, project_name):
    """設置新項目結構"""
    
    # 創建項目命名空間層級
    app.createNamespace(project_name, description=f"{project_name} 主空間")
    app.createNamespace(f"{project_name}_core", description="核心代碼", parent=project_name)
    app.createNamespace(f"{project_name}_tests", description="測試代碼", parent=project_name)
    app.createNamespace(f"{project_name}_docs", description="文檔", parent=project_name)
    app.createNamespace(f"{project_name}_config", description="配置", parent=project_name)
    
    # 初始化配置
    config = {
        "project_name": project_name,
        "version": "0.1.0",
        "created": datetime.now().isoformat(),
        "settings": {
            "debug": True,
            "log_level": "INFO"
        }
    }
    app.saveData("config", config, f"{project_name}_config")
    
    return f"項目 {project_name} 初始化完成"


def add_core_module(app, project_name, module_name, code, tests=None, docs=None):
    """添加核心模組"""
    
    # 保存代碼
    app.saveCode(
        module_name,
        code,
        f"{project_name}_core",
        metadata={"module": module_name, "created": datetime.now().isoformat()}
    )
    
    # 保存測試
    if tests:
        app.saveCode(
            f"test_{module_name}",
            tests,
            f"{project_name}_tests",
            metadata={"tests_for": module_name}
        )
    
    # 保存文檔
    if docs:
        app.saveKnowledge(
            f"{module_name}_doc",
            docs,
            f"{project_name}_docs",
            tags=[module_name, "documentation"]
        )
    
    return f"模組 {module_name} 已添加"


def run_all_tests(app, project_name):
    """執行所有測試"""
    
    test_nodes = app.listNodes(content_type="code", namespace=f"{project_name}_tests")
    results = []
    
    for node in test_nodes:
        result = app.run(node["name"], namespace=f"{project_name}_tests")
        results.append({
            "test": node["name"],
            "passed": result.success,
            "time_ms": result.execution_time_ms,
            "error": result.error if not result.success else None
        })
    
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    
    # 保存測試報告
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total if total > 0 else 0,
        "details": results
    }
    
    app.saveData(
        f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        report,
        f"{project_name}_tests"
    )
    
    return report


def deploy_to_production(app, project_name):
    """部署到生產環境"""
    
    # 複製核心代碼到生產命名空間
    app.createNamespace("production", description="生產環境")
    
    copied = app.copyNamespace(
        f"{project_name}_core",
        "production",
        include_codes=True,
        include_data=False,  # 不複製測試數據
        include_groups=False
    )
    
    # 複製生產配置
    dev_config = app.getData("config", f"{project_name}_config")
    prod_config = {**dev_config, "settings": {"debug": False, "log_level": "WARNING"}}
    app.saveData("config", prod_config, "production")
    
    # 校準生產環境
    app.calibrateNamespace("production")
    
    return f"已部署 {copied['codes']} 個模組到生產環境"


# 執行完整工作流程
with DynamicApp("./project_data") as app:
    
    # 1. 設置項目
    print(setup_project(app, "my_app"))
    
    # 2. 添加核心模組
    user_module = """
def create_user(name, email):
    return {"name": name, "email": email, "active": True}

def validate_email(email):
    return "@" in email and "." in email

def main():
    user = create_user("Alice", "alice@example.com")
    return user
"""
    
    user_tests = """
def main():
    # 測試用戶創建
    from types import SimpleNamespace
    user = {"name": "Test", "email": "test@test.com", "active": True}
    assert user["name"] == "Test"
    assert user["active"] == True
    return "All tests passed"
"""
    
    user_docs = """
# User Module

## Functions

### create_user(name, email)
創建新用戶對象。

### validate_email(email)
驗證電子郵件格式。
"""
    
    print(add_core_module(app, "my_app", "user", user_module, user_tests, user_docs))
    
    # 3. 執行測試
    report = run_all_tests(app, "my_app")
    print(f"測試結果: {report['passed']}/{report['total']} 通過")
    
    # 4. 部署
    if report["success_rate"] == 1.0:
        print(deploy_to_production(app, "my_app"))
    else:
        print("測試未全部通過，取消部署")
    
    # 5. 查看最終統計
    stats = app.getStats()
    print(f"\n最終統計:")
    print(f"  總節點數: {stats['total_nodes']}")
    print(f"  代碼節點: {stats['code_nodes']}")
    print(f"  數據節點: {stats['data_nodes']}")
    print(f"  命名空間: {stats['namespaces']}")
```

### 11.2 知識庫構建與查詢

```python
def build_knowledge_base(app):
    """構建完整的知識庫"""
    
    # 創建知識庫命名空間
    app.createNamespace("kb", description="知識庫")
    app.createNamespace("kb_tutorials", description="教程", parent="kb")
    app.createNamespace("kb_references", description="參考資料", parent="kb")
    app.createNamespace("kb_examples", description="代碼範例", parent="kb")
    
    # 添加教程
    tutorials = [
        ("python_basics", "# Python 基礎\n\n變量、數據類型、控制流...", ["python", "basics"]),
        ("python_functions", "# Python 函數\n\n函數定義、參數、返回值...", ["python", "functions"]),
        ("python_oop", "# Python OOP\n\n類、繼承、多態...", ["python", "oop"]),
    ]
    
    for name, content, tags in tutorials:
        app.saveKnowledge(name, content, "kb_tutorials", tags=tags)
    
    # 添加代碼範例資料組
    example_items = [
        ("def greet(name):\n    return f'Hello, {name}!'", "code", {"topic": "functions"}),
        ("class Person:\n    def __init__(self, name):\n        self.name = name", "code", {"topic": "oop"}),
        ("squares = [x**2 for x in range(10)]", "code", {"topic": "comprehension"}),
        ("with open('file.txt') as f:\n    content = f.read()", "code", {"topic": "file_io"}),
    ]
    
    app.createDataGroup(
        "python_code_examples",
        example_items,
        "kb_examples",
        description="Python 代碼範例集",
        tags=["python", "examples"],
        calibrate=True
    )
    
    # 校準整個知識庫
    for ns in ["kb_tutorials", "kb_references", "kb_examples"]:
        app.calibrateNamespace(ns)
    
    return "知識庫構建完成"


def query_knowledge_base(app, query):
    """查詢知識庫"""
    
    print(f"\n查詢: {query}")
    print("=" * 50)
    
    # 跨類型關聯搜索
    results = app.searchRelated(
        query=query,
        content_types=["knowledge", "code", "data_group"],
        threshold=0.4,
        limit=10
    )
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.content_type}] {r.name}")
        print(f"   相似度: {r.score:.2%}")
        print(f"   命名空間: {r.namespace}")
        if r.preview:
            print(f"   預覽: {r.preview[:100]}...")
        if r.related_items:
            print(f"   關聯: {', '.join(r.related_items)}")
    
    return results


# 使用範例
with DynamicApp("./knowledge_base_data") as app:
    
    # 構建知識庫
    print(build_knowledge_base(app))
    
    # 執行查詢
    query_knowledge_base(app, "Python 類和對象")
    query_knowledge_base(app, "如何定義函數")
    query_knowledge_base(app, "列表推導式")
```

---

## 12. 最佳實踐與性能優化

### 12.1 命名空間組織策略

```python
# 推薦的命名空間層級結構
"""
project/
├── core/           # 核心代碼
│   ├── auth/       # 認證模組
│   ├── api/        # API 模組
│   └── utils/      # 工具函數
├── tests/          # 測試代碼
├── docs/           # 文檔
├── config/         # 配置
│   ├── dev/        # 開發配置
│   ├── staging/    # 預發布配置
│   └── prod/       # 生產配置
└── data/           # 數據
    ├── models/     # 數據模型
    └── samples/    # 示例數據
"""

def setup_recommended_structure(app, project_name):
    """設置推薦的項目結構"""
    
    namespaces = [
        (project_name, "項目根", None),
        (f"{project_name}_core", "核心代碼", project_name),
        (f"{project_name}_core_auth", "認證模組", f"{project_name}_core"),
        (f"{project_name}_core_api", "API模組", f"{project_name}_core"),
        (f"{project_name}_core_utils", "工具函數", f"{project_name}_core"),
        (f"{project_name}_tests", "測試代碼", project_name),
        (f"{project_name}_docs", "文檔", project_name),
        (f"{project_name}_config", "配置", project_name),
        (f"{project_name}_data", "數據", project_name),
    ]
    
    for name, desc, parent in namespaces:
        app.createNamespace(name, description=desc, parent=parent)
    
    return f"已創建 {len(namespaces)} 個命名空間"
```

### 12.2 批量操作優化

```python
def batch_save_codes(app, codes, namespace, calibrate_after=True):
    """
    批量保存代碼的優化方法
    """
    nodes = []
    
    # 禁用單次校準，改為批量校準
    for name, code, metadata in codes:
        node = app.saveCode(name, code, namespace, metadata)
        nodes.append(node)
    
    # 一次性校準
    if calibrate_after:
        app.calibrateNamespace(namespace)
    
    return nodes


def batch_search(app, queries, namespace=None):
    """
    批量搜索的優化方法
    """
    results = {}
    
    for query in queries:
        results[query] = app.search(
            query=query,
            mode="semantic",  # 語義搜索效率較高
            namespace=namespace,
            fast_match_limit=500,  # 減少候選數
            result_limit=10
        )
    
    return results
```

### 12.3 記憶體管理

```python
def process_large_dataset(app, items, namespace, batch_size=100):
    """
    處理大量數據時的記憶體管理策略
    """
    import gc
    
    total_processed = 0
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # 處理批次
        for name, content, item_type, metadata in batch:
            if item_type == "code":
                app.saveCode(name, content, namespace, metadata)
            else:
                app.saveData(name, content, namespace, metadata=metadata)
        
        total_processed += len(batch)
        
        # 定期清理
        if total_processed % 500 == 0:
            gc.collect()
            print(f"已處理 {total_processed} 項，執行垃圾回收")
    
    return total_processed
```

### 12.4 搜索優化策略

```python
def optimized_search_strategy(app, query, expected_type=None):
    """
    根據查詢特徵選擇最優搜索策略
    """
    
    # 分析查詢特徵
    is_code_pattern = any(kw in query.lower() for kw in ["def ", "class ", "import ", "return "])
    is_short_query = len(query) < 20
    is_regex = any(c in query for c in r".*+?[]{}()|^$\\")
    
    if is_regex:
        mode = "regex"
        threshold = 0.0
    elif is_code_pattern:
        mode = "exact"
        threshold = 0.0
    elif is_short_query:
        mode = "fuzzy"
        threshold = 0.4
    else:
        mode = "semantic"
        threshold = 0.5
    
    return app.search(
        query=query,
        mode=mode,
        content_type=expected_type or "all",
        similarity_threshold=threshold,
        result_limit=20
    )
```

### 12.5 錯誤處理模式

```python
def safe_execute(app, name, namespace, **kwargs):
    """
    安全執行代碼的封裝
    """
    try:
        result = app.run(name, namespace, **kwargs)
        
        if result.success:
            return {"success": True, "result": result.result}
        else:
            return {
                "success": False,
                "error": result.error,
                "details": result.stderr
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "details": traceback.format_exc()
        }


def safe_batch_execute(app, tasks, namespace):
    """
    安全批量執行
    """
    results = []
    
    for task in tasks:
        name = task.get("name")
        kwargs = task.get("kwargs", {})
        
        result = safe_execute(app, name, namespace, kwargs=kwargs)
        result["task"] = task
        results.append(result)
    
    return {
        "total": len(results),
        "success": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "details": results
    }
```

---

## 總結

ASMdynamicGradio 動態知識與工具體系提供了一套完整的解決方案，涵蓋：

| 能力領域 | 核心功能 | 關鍵方法 |
|---------|---------|---------|
| **代碼管理** | 保存、載入、執行、導入 | `saveCode`, `getCode`, `run`, `importCode` |
| **命名空間** | 隔離、複製、重命名、刪除 | `createNamespace`, `copyNamespace`, `renameNamespace` |
| **資料組** | 多類型內容統一管理 | `createDataGroup`, `addToDataGroup`, `updateDataGroup` |
| **Embedding 校準** | 跨類型語義關聯 | `batchImportWithCalibration`, `calibrateNamespace` |
| **智能搜索** | 多模式、關聯搜索 | `search`, `searchRelated` |
| **能力分發** | 項目複製、導入導出 | `copyNamespace`, `fromFolder`, `toFolder` |
| **經驗回收** | 日誌、統計、分析 | `get_debug_log`, `getStats`, `getSystemInfo` |
| **演化可視化** | 動態視覺化輸出 | `initEvolution`, `evolve`, `saveVisualization` |

通過合理組合這些功能，可以構建高效的動態知識管理和代碼開發工作流程。