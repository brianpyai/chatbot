# AIchat.py 完整使用說明

## 目錄

1. [模組概述](#模組概述)
2. [安裝與配置](#安裝與配置)
3. [資料結構](#資料結構)
4. [ChatClient 類別](#chatclient-類別)
5. [核心聊天函數](#核心聊天函數)
6. [AIProcessor 類別](#aiprocessor-類別)
7. [ChatTerminal 互動終端](#chatterminal-互動終端)
8. [命令列介面](#命令列介面)
9. [完整範例](#完整範例)

---

## 模組概述

`AIchat.py` 是一個專業的 AI 聊天模組，提供以下功能：

- **互動式聊天終端**：美觀的 UI 和完整的命令系統
- **流式聊天**：支援多種模型（Ollama 本地模型 + API 模型）
- **AI 輔助工具**：程式碼修改、自動補全、翻譯、摘要等
- **完整測試套件**：全面的功能驗證

---

## 安裝與配置

### 依賴套件

```python
from openai import OpenAI
from myText import Text  # 自訂文字格式化模組
```

### API 金鑰配置

```python
# 方法 1：從私有模組導入
from myPrivate import api_key as API_KEY

# 方法 2：直接設定（不建議在程式碼中硬編碼）
API_KEY = "your-api-key-here"
```

### 模型配置

```python
# Ollama 本地模型
OLLAMA_BOTS = ["qwen3:0.6b", "qwen3:1.7b"]

# API 模型
DEFAULT_FIXED_BOTS = [
    "GPT-5.2", "GPT-5.1-Codex-Mini", "Gemini-2.5-Flash-Lite",
    "GPT-5-mini", "GPT-5-nano", "Gemini-3-Flash",
    "Gemini-3-Pro", "Web-Search"
]

# 預設模型
DEFAULT_BOT = "GPT-5-nano"
```

---

## 資料結構

### TaskType 列舉

定義 AI 處理任務的類型。

```python
from enum import Enum

class TaskType(Enum):
    CODE_MODIFY = "code_modify"  # 程式碼修改
    COMPLETE = "complete"        # 自動補全
    SUGGEST = "suggest"          # 輸入建議
    SELECT = "select"            # 選項選擇
    REWRITE = "rewrite"          # 內容重寫
    SUMMARIZE = "summarize"      # 內容摘要
    TRANSLATE = "translate"      # 翻譯
```

**使用範例：**

```python
# 使用列舉
task = TaskType.TRANSLATE

# 從字串轉換
task = TaskType("translate")

# 取得值
print(task.value)  # 輸出: "translate"
```

---

### Message 資料類別

表示單一聊天訊息。

```python
@dataclass
class Message:
    role: str           # "user" 或 "assistant"
    content: str        # 訊息內容
    timestamp: float    # 時間戳記（自動生成）
```

**使用範例：**

```python
# 建立訊息
msg = Message(role="user", content="你好！")

# 序列化為字典
data = msg.to_dict()
# {'role': 'user', 'content': '你好！', 'timestamp': 1703123456.789}

# 從字典還原
msg2 = Message.from_dict(data)
```

---

### ConversationContext 資料類別

管理對話上下文，支援多輪對話。

```python
@dataclass
class ConversationContext:
    messages: List[Message]      # 訊息列表
    system: Optional[str]        # 系統提示詞
    model: str                   # 使用的模型
```

**使用範例：**

```python
# 建立對話上下文
ctx = ConversationContext(
    system="你是一個有幫助的助手",
    model="GPT-5-nano"
)

# 新增訊息
ctx.add_message("user", "Python 是什麼？")
ctx.add_message("assistant", "Python 是一種程式語言...")

# 取得 API 格式的訊息列表
api_messages = ctx.get_messages_for_api()
# [
#     {"role": "system", "content": "你是一個有幫助的助手"},
#     {"role": "user", "content": "Python 是什麼？"},
#     {"role": "assistant", "content": "Python 是一種程式語言..."}
# ]

# 清除對話歷史
ctx.clear()

# 序列化與還原
data = ctx.to_dict()
ctx2 = ConversationContext.from_dict(data)
```

---

## ChatClient 類別

核心聊天客戶端，支援 Ollama 和 API 兩種後端。

### 初始化

```python
client = ChatClient(model="GPT-5-nano", api_key="your-key")
```

**參數說明：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `model` | str | DEFAULT_BOT | 使用的模型名稱 |
| `api_key` | str | API_KEY | API 金鑰 |

### 方法詳解

#### set_model(model)

切換使用的模型。

```python
client = ChatClient(model="GPT-5-nano")
client.set_model("GPT-5-mini")  # 切換到不同模型
```

#### stream_chat(message, system, role, context)

流式聊天，逐塊返回回應。

```python
# 基本用法
for chunk in client.stream_chat("你好！"):
    print(chunk, end="", flush=True)

# 使用系統提示詞
for chunk in client.stream_chat(
    message="寫一首詩",
    system="你是一位詩人，用優美的語言回答"
):
    print(chunk, end="", flush=True)

# 使用對話上下文（多輪對話）
ctx = ConversationContext(system="你是程式專家")
ctx.add_message("user", "什麼是變數？")
ctx.add_message("assistant", "變數是用來儲存資料的容器...")

for chunk in client.stream_chat("那函數呢？", context=ctx):
    print(chunk, end="", flush=True)
```

**參數說明：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `message` | str | (必填) | 使用者訊息 |
| `system` | str | None | 系統提示詞 |
| `role` | str | "user" | 訊息角色 |
| `context` | ConversationContext | None | 對話上下文 |

#### chat(message, system, context)

非流式聊天，返回完整回應。

```python
# 基本用法
response = client.chat("什麼是人工智慧？")
print(response)

# 使用系統提示詞
response = client.chat(
    message="解釋量子計算",
    system="用簡單的語言解釋，適合初學者"
)
```

#### is_available()

檢查 API 是否可用。

```python
client = ChatClient(model="GPT-5-nano")
if client.is_available():
    print("API 連線正常")
else:
    print("API 無法連線")
```

---

## 核心聊天函數

### chatStreamOllama()

便捷的流式聊天函數。

```python
def chatStreamOllama(
    message: str,
    model: str = DEFAULT_BOT,
    role: str = "user",
    stream: bool = True,
    system: Optional[str] = None
) -> Generator[str, None, None]
```

**使用範例：**

```python
# 基本流式輸出
for chunk in chatStreamOllama("解釋機器學習"):
    print(chunk, end="", flush=True)
print()  # 換行

# 指定模型和系統提示詞
for chunk in chatStreamOllama(
    message="寫一個 Python 函數",
    model="GPT-5.1-Codex-Mini",
    system="你是資深 Python 開發者"
):
    print(chunk, end="", flush=True)

# 收集完整回應
chunks = list(chatStreamOllama("你好"))
full_response = ''.join(chunks)
```

---

### askBot()

批量問答生成器，支援多個問題。

```python
def askBot(
    questions: Union[str, List[str]],
    user: str = "You",
    Bot: str = "Assistant",
    model: str = DEFAULT_BOT,
    system: Optional[str] = None,
    verbose: bool = False
) -> Generator[Tuple[str, str], None, None]
```

**使用範例：**

```python
# 單一問題
for question, answer in askBot("Python 的優點是什麼？"):
    print(f"問：{question}")
    print(f"答：{answer}")

# 多個問題
questions = [
    "什麼是變數？",
    "什麼是函數？",
    "什麼是類別？"
]

for q, a in askBot(questions, model="GPT-5-nano"):
    print(f"\n問：{q}")
    print(f"答：{a}")

# 帶系統提示詞
for q, a in askBot(
    questions=["解釋 REST API"],
    system="用繁體中文回答，保持簡潔",
    verbose=True  # 即時顯示輸出
):
    pass  # verbose=True 時會自動列印

# 使用自訂名稱和詳細輸出
for q, a in askBot(
    questions=["天氣如何？"],
    user="小明",
    Bot="AI助手",
    verbose=True
):
    # verbose=True 會顯示：
    # 小明: 天氣如何？
    # AI助手: [回應內容]
    pass
```

---

## AIProcessor 類別

AI 輔助內容處理工具，提供多種文字處理功能。

### 初始化

```python
processor = AIProcessor(model="GPT-5-nano")
```

### 通用處理方法

```python
def process(
    context: str,      # 完整上下文
    part: str,         # 要處理的部分
    task: TaskType,    # 任務類型
    **kwargs           # 任務特定參數
) -> Union[str, List[str], Dict[str, Any]]
```

---

### 任務類型詳解

#### 1. CODE_MODIFY - 程式碼修改

根據指令修改程式碼。

```python
processor = AIProcessor(model="GPT-5.1-Codex-Mini")

# 原始程式碼
code = """
def add(a, b):
    return a + b
"""

# 修改程式碼
result = processor.process(
    context="Python 數學函數",
    part=code,
    task=TaskType.CODE_MODIFY,
    instruction="添加類型提示和文件字串"
)

print(result)
# 輸出類似：
# def add(a: int, b: int) -> int:
#     """Add two numbers and return the result."""
#     return a + b
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `instruction` | str | "Improve the code" | 修改指令 |

---

#### 2. COMPLETE - 自動補全

自動補全未完成的內容。

```python
processor = AIProcessor()

result = processor.process(
    context="撰寫商業電子郵件",
    part="親愛的客戶您好，感謝您的來信。關於您詢問的產品",
    task=TaskType.COMPLETE,
    max_length=100
)

print(result)
# 輸出類似：
# ，我們很高興為您介紹我們的最新款式，該產品具有以下特點...
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `max_length` | int | 100 | 最大補全長度（字元數） |

---

#### 3. SUGGEST - 輸入建議

根據上下文生成輸入建議。

```python
processor = AIProcessor()

suggestions = processor.process(
    context="程式語言相關搜尋",
    part="Py",
    task=TaskType.SUGGEST,
    count=5
)

print(suggestions)
# 輸出類似：
# ['Python', 'PyTorch', 'Pygame', 'PyQt', 'Pydantic']
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `count` | int | 5 | 建議數量 |

---

#### 4. SELECT - 選項選擇

從多個選項中選擇最佳答案。

```python
processor = AIProcessor()

result = processor.process(
    context="設計警告標誌",
    part="最適合用於危險警告的顏色是？",
    task=TaskType.SELECT,
    options=["藍色", "紅色", "綠色", "黃色"]
)

print(result)
# 輸出：
# {'selected': 1, 'option': '紅色'}
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `options` | List[str] | None | 選項列表 |

**回傳值：**
```python
{
    "selected": int,    # 選擇的索引（從 0 開始）
    "option": str       # 選擇的選項內容
}
```

---

#### 5. REWRITE - 內容重寫

用指定風格重寫內容。

```python
processor = AIProcessor()

# 將口語化文字改為正式風格
result = processor.process(
    context="商業溝通",
    part="嘿，這東西超讚的啦，你一定要買！",
    task=TaskType.REWRITE,
    style="formal"  # 正式風格
)

print(result)
# 輸出類似：
# 您好，這是一款優質的產品，我們誠摯推薦您考慮購買。

# 其他風格範例
styles = ["casual", "professional", "academic", "humorous", "concise"]
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `style` | str | "professional" | 重寫風格 |

---

#### 6. SUMMARIZE - 內容摘要

將長文摘要為簡短內容。

```python
processor = AIProcessor()

long_text = """
Python 是一種高階程式語言，由 Guido van Rossum 於 1991 年創建。
Python 以其簡潔的語法和強大的功能而聞名。它支援多種程式設計範式，
包括程序式、物件導向和函數式程式設計。Python 廣泛應用於網頁開發、
資料科學、人工智慧、自動化腳本等領域。由於其易學易用的特性，
Python 成為初學者學習程式設計的首選語言之一。
"""

result = processor.process(
    context="",
    part=long_text,
    task=TaskType.SUMMARIZE,
    max_words=30
)

print(result)
# 輸出類似：
# Python 是一種易學的高階程式語言，支援多種程式設計範式，廣泛用於網頁開發和資料科學。
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `max_words` | int | 50 | 最大字數 |

---

#### 7. TRANSLATE - 翻譯

將內容翻譯成目標語言。

```python
processor = AIProcessor()

# 英文翻譯成中文
result = processor.process(
    context="",
    part="Hello, how are you today?",
    task=TaskType.TRANSLATE,
    target_lang="繁體中文"
)
print(result)
# 輸出：你好，你今天好嗎？

# 中文翻譯成日文
result = processor.process(
    context="",
    part="謝謝你的幫助",
    task=TaskType.TRANSLATE,
    target_lang="Japanese"
)
print(result)
# 輸出：ご協力ありがとうございます

# 支援的語言範例
languages = [
    "English", "繁體中文", "简体中文", "Japanese",
    "Korean", "French", "German", "Spanish"
]
```

**參數：**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `target_lang` | str | "English" | 目標語言 |

---

### aiProcess() 便捷函數

簡化的處理函數，無需建立 AIProcessor 實例。

```python
def aiProcess(
    context: str,
    part: str,
    task: Union[str, TaskType],
    model: str = DEFAULT_BOT,
    **kwargs
) -> Any
```

**使用範例：**

```python
# 使用字串指定任務類型
result = aiProcess(
    context="",
    part="Good morning!",
    task="translate",  # 字串形式
    target_lang="繁體中文"
)

# 使用列舉指定任務類型
result = aiProcess(
    context="Python 程式",
    part="def hello",
    task=TaskType.SUGGEST,
    count=3
)

# 指定模型
result = aiProcess(
    context="",
    part="print('hello')",
    task="code_modify",
    model="GPT-5.1-Codex-Mini",
    instruction="添加錯誤處理"
)
```

---

## ChatTerminal 互動終端

完整功能的互動式聊天終端。

### 初始化與啟動

```python
from AIchat import ChatTerminal, chatTerminal

# 方法 1：使用類別
terminal = ChatTerminal(
    user="小明",           # 使用者名稱
    Bot="AI助手",          # 機器人名稱
    model="GPT-5-nano",   # 使用的模型
    system="你是一個友善的助手",  # 系統提示詞
    stream=True           # 啟用流式輸出
)
terminal.run()

# 方法 2：使用便捷函數
chatTerminal(
    user="小明",
    Bot="AI助手",
    model="GPT-5-nano",
    system="你是一個友善的助手"
)
```

### 終端命令完整列表

啟動終端後，可使用以下命令：

#### 導航命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/?` | 顯示快速幫助 | `/?` |
| `/help` | 顯示詳細幫助 | `/help` |
| `/exit` | 退出終端 | `/exit` |
| `/quit` | 退出終端（別名） | `/quit` |
| `/q` | 退出終端（簡寫） | `/q` |

#### 模型命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/model` | 顯示當前模型 | `/model` |
| `/model <n>` | 切換到編號 n 的模型 | `/model 3` |
| `/model <name>` | 切換到指定名稱的模型 | `/model GPT-5-mini` |
| `/models` | 列出所有可用模型 | `/models` |

#### 對話命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/system` | 顯示當前系統提示詞 | `/system` |
| `/system <prompt>` | 設定系統提示詞 | `/system 你是專業程式設計師` |
| `/system clear` | 清除系統提示詞 | `/system clear` |
| `/reset` | 重置對話歷史 | `/reset` |
| `/history` | 顯示最近 10 條訊息 | `/history` |
| `/history <n>` | 顯示最近 n 條訊息 | `/history 20` |
| `/retry` | 重新發送上一條訊息 | `/retry` |

#### 顯示命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/clear` | 清除終端畫面 | `/clear` |
| `/stream` | 切換流式輸出開關 | `/stream` |
| `/markdown` | 切換 Markdown 渲染開關 | `/markdown` |

#### 檔案命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/save` | 儲存對話（自動命名） | `/save` |
| `/save <file>` | 儲存對話到指定檔案 | `/save my_chat.json` |
| `/load <file>` | 載入對話 | `/load my_chat.json` |
| `/export` | 匯出為文字檔（自動命名） | `/export` |
| `/export <file>` | 匯出到指定檔案 | `/export chat.txt` |

#### 資訊命令

| 命令 | 說明 | 範例 |
|------|------|------|
| `/stats` | 顯示會話統計資訊 | `/stats` |

### 終端使用範例

```
╔═══════════════════════════════════════════════════════════╗
║      █████╗ ██╗     ██████╗██╗  ██╗ █████╗ ████████╗      ║
║     ██╔══██╗██║    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝      ║
║     ███████║██║    ██║     ███████║███████║   ██║         ║
║     ██╔══██║██║    ██║     ██╔══██║██╔══██║   ██║         ║
║     ██║  ██║██║    ╚██████╗██║  ██║██║  ██║   ██║         ║
║     ╚═╝  ╚═╝╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝         ║
╚═══════════════════════════════════════════════════════════╝

  Model: GPT-5-nano  |  Type /? for help  |  /exit to quit
  ─────────────────────────────────────────────────────────

You:
什麼是 Python？