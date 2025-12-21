# AIchat.py 完整使用說明文檔 v3.2

## 目錄

1. [模組概述](#1-模組概述)
2. [安裝與配置](#2-安裝與配置)
3. [枚舉類型 (Enums)](#3-枚舉類型-enums)
4. [資料類別 (Data Classes)](#4-資料類別-data-classes)
5. [AIDownloader 下載器類別](#5-aidownloader-下載器類別)
6. [ChatClient 聊天客戶端類別](#6-chatclient-聊天客戶端類別)
7. [核心聊天函數](#7-核心聊天函數)
8. [AIProcessor 內容處理器](#8-aiprocessor-內容處理器)
9. [ChatTerminal 互動終端](#9-chatterminal-互動終端)
10. [測試函數](#10-測試函數)
11. [命令列使用](#11-命令列使用)

---

## 1. 模組概述

`AIchat.py` 是一個功能完整的 AI 聊天模組，支援多種 AI 模型、串流回應、圖片生成、網路搜尋、函數調用，以及自動下載附件等功能。

### 主要特色

- 支援多種 AI 模型（本地 Ollama 與 API 模型）
- 串流式回應輸出
- 圖片生成與視覺理解
- 網路搜尋功能
- 函數調用 (Function Calling)
- 自動下載 AI 生成的附件
- 多輪對話上下文管理
- 互動式終端介面
- 完整的 Markdown 渲染支援

---

## 2. 安裝與配置

### 依賴套件

```python
from openai import OpenAI      # 必要
from myText import Text        # 自定義文字處理模組
import httpx                   # 選用，用於非同步下載
```

### 配置常數

```python
# API 金鑰設定（從 myPrivate 模組匯入或手動設定）
API_KEY = "your-api-key"

# 可用的本地 Ollama 模型
OLLAMA_BOTS = ["qwen3:0.6b", "qwen3:1.7b"]

# API 模型列表
DEFAULT_FIXED_BOTS = [
    "GPT-5.2", "GPT-5.1-Codex-Mini", "Gemini-2.5-Flash-Lite",
    "GPT-5-mini", "GPT-5-nano", "Gemini-3-Flash",
    "Gemini-3-Pro", "Web-Search",
    "Qwen-Image", "Nano-Banana",
    "GPT-Image-1-Mini", "FLUX-schnell", "Sonic-3.0"
]

# 特定功能支援的模型
IMAGE_GENERATION_MODELS = ["GPT-Image-1", "GPT-Image-1-Mini", "Qwen-Image"]
WEB_SEARCH_MODELS = ["Web-Search", "Gemini-3-Pro", "Gemini-3-Flash", "GPT-5.2"]
VISION_MODELS = ["GPT-5-nano", "GPT-5-mini", "Gemini-3-Pro", "Gemini-3-Flash", "GPT-5.2"]

# 預設值
DEFAULT_BOT = "GPT-5-nano"           # 預設聊天模型
DEFAULT_BOT_IMG = "GPT-Image-1-Mini" # 預設圖片生成模型
DEFAULT_DOWNLOAD_DIR = "./AIdownload" # 預設下載目錄
```

---

## 3. 枚舉類型 (Enums)

### 3.1 TaskType - 任務類型

用於 `AIProcessor` 的任務分類。

```python
from enum import Enum

class TaskType(Enum):
    CODE_MODIFY = "code_modify"  # 程式碼修改
    COMPLETE = "complete"        # 自動完成
    SUGGEST = "suggest"          # 建議生成
    SELECT = "select"            # 選項選擇
    REWRITE = "rewrite"          # 內容改寫
    SUMMARIZE = "summarize"      # 內容摘要
    TRANSLATE = "translate"      # 翻譯

# 使用範例
task = TaskType.TRANSLATE
print(task.value)  # 輸出: "translate"
```

### 3.2 ImageAspect - 圖片長寬比

```python
class ImageAspect(Enum):
    SQUARE = "1:1"      # 正方形
    LANDSCAPE = "3:2"   # 橫向
    PORTRAIT = "2:3"    # 縱向
    WIDE = "16:9"       # 寬螢幕
    AUTO = "auto"       # 自動

# 使用範例
aspect = ImageAspect.LANDSCAPE
print(aspect.value)  # 輸出: "3:2"
```

### 3.3 ImageQuality - 圖片品質

```python
class ImageQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# 使用範例
quality = ImageQuality.HIGH
```

---

## 4. 資料類別 (Data Classes)

### 4.1 Attachment - 通用附件

表示 API 回應中的附件。

```python
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class Attachment:
    url: str                                    # 附件 URL
    content_type: str = "application/octet-stream"  # MIME 類型
    filename: Optional[str] = None              # 檔案名稱
    data: Optional[bytes] = None                # 二進位資料

# 建立附件
attachment = Attachment(
    url="https://example.com/image.png",
    content_type="image/png",
    filename="image.png"
)

# 取得副檔名
print(attachment.extension)  # 輸出: ".png"
```

### 4.2 StreamResponse - 串流回應結果

封裝串流完成後的完整結果。

```python
@dataclass
class StreamResponse:
    content: str = ""                           # 回應文字內容
    attachments: List[Attachment] = field(default_factory=list)   # 附件列表
    downloaded_files: List[Path] = field(default_factory=list)    # 已下載檔案
    extracted_urls: List[str] = field(default_factory=list)       # 擷取的 URL

# 使用範例
response = StreamResponse(
    content="這是回應內容",
    attachments=[attachment],
    downloaded_files=[Path("./AIdownload/image_20231215_123456.png")]
)

# 檢查是否有附件
if response.has_attachments():
    print(f"共有 {len(response.attachments)} 個附件")

# 檢查是否有下載
if response.has_downloads():
    print(f"已下載 {len(response.downloaded_files)} 個檔案")
```

### 4.3 StreamingChatResponse - 串流聊天回應包裝器

包裝串流聊天回應，支援迭代並收集附件。

```python
# 建立方式（通常由 ChatClient 內部建立）
stream = client.stream_chat_with_attachments("Hello")

# 迭代使用
for chunk, attachments in stream:
    print(chunk, end="", flush=True)

# 取得最終結果
result = stream.result  # 類型: StreamResponse
print(f"完整內容: {result.content}")
print(f"下載檔案: {result.downloaded_files}")

# 或直接消費整個串流
full_content = stream.stream_to_string()

# 使用 consume() 方法
result = stream.consume()  # 消費並返回 StreamResponse
```

### 4.4 ImageAttachment - 圖片附件

用於發送圖片到支援視覺的模型。

```python
@dataclass
class ImageAttachment:
    source: str                          # URL 或檔案路徑
    is_base64: bool = False              # 是否為 base64 編碼
    base64_data: Optional[str] = None    # base64 資料
    mime_type: str = "image/jpeg"        # MIME 類型

# 從本地檔案建立
img_from_file = ImageAttachment.from_file("./photo.jpg")

# 從 URL 建立
img_from_url = ImageAttachment.from_url("https://example.com/image.png")

# 從 base64 建立
img_from_base64 = ImageAttachment.from_base64(
    base64_data="iVBORw0KGgo...",
    mime_type="image/png"
)

# 轉換為 API 格式
api_format = img_from_file.to_api_format()
# 輸出: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
```

### 4.5 FileAttachment - 檔案附件

用於發送檔案到 API。

```python
@dataclass
class FileAttachment:
    file_path: str                       # 檔案路徑
    filename: Optional[str] = None       # 檔案名稱
    base64_data: Optional[str] = None    # base64 資料
    mime_type: str = "application/octet-stream"

# 建立檔案附件
file_attachment = FileAttachment(file_path="./document.pdf")

# 自動偵測 MIME 類型並讀取檔案
print(file_attachment.mime_type)  # 輸出: "application/pdf"

# 轉換為 API 格式
api_format = file_attachment.to_api_format()
```

### 4.6 FunctionDefinition - 函數定義

用於函數調用功能。

```python
@dataclass
class FunctionDefinition:
    name: str                            # 函數名稱
    description: str                     # 函數描述
    parameters: Dict[str, Any]           # 參數定義 (JSON Schema)
    handler: Optional[Callable] = None   # 處理函數

# 定義函數
def get_weather(location: str, unit: str = "celsius") -> str:
    return f"{location} 的天氣是晴天，溫度 25°{unit[0].upper()}"

weather_func = FunctionDefinition(
    name="get_weather",
    description="取得指定地點的天氣資訊",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市名稱"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    },
    handler=get_weather
)

# 轉換為 API 格式
api_format = weather_func.to_api_format()
```

### 4.7 FunctionCall - 函數調用

表示模型返回的函數調用。

```python
@dataclass
class FunctionCall:
    id: str                              # 調用 ID
    name: str                            # 函數名稱
    arguments: Dict[str, Any]            # 參數

# 從 API 回應解析
func_call = FunctionCall.from_tool_call(tool_call_from_api)

# 手動建立
func_call = FunctionCall(
    id="call_123",
    name="get_weather",
    arguments={"location": "台北", "unit": "celsius"}
)
```

### 4.8 Message - 聊天訊息

表示對話中的單一訊息。

```python
@dataclass
class Message:
    role: str                            # 角色: "user", "assistant", "system", "tool"
    content: Union[str, List[Dict]]      # 內容
    timestamp: float                     # 時間戳記
    images: List[ImageAttachment]        # 圖片附件
    files: List[FileAttachment]          # 檔案附件
    tool_calls: List[FunctionCall]       # 函數調用
    tool_call_id: Optional[str]          # 工具回應 ID
    attachments: List[Attachment]        # 回應附件

# 建立訊息
message = Message(
    role="user",
    content="請描述這張圖片",
    images=[ImageAttachment.from_file("./photo.jpg")]
)

# 轉換為字典
msg_dict = message.to_dict()

# 從字典建立
message = Message.from_dict(msg_dict)

# 轉換為 API 格式
api_format = message.to_api_format()
```

### 4.9 ConversationContext - 對話上下文

管理多輪對話的上下文。

```python
@dataclass
class ConversationContext:
    messages: List[Message]              # 訊息列表
    system: Optional[str]                # 系統提示
    model: str                           # 使用的模型
    tools: List[FunctionDefinition]      # 已註冊的工具

# 建立對話上下文
context = ConversationContext(
    system="你是一個專業的助手",
    model="GPT-5-nano"
)

# 新增訊息
context.add_message(
    role="user",
    content="你好！",
    images=[ImageAttachment.from_url("https://example.com/img.png")]
)

# 新增助手回應
context.add_message(
    role="assistant",
    content="你好！有什麼可以幫助你的嗎？"
)

# 新增工具結果
context.add_tool_result(
    tool_call_id="call_123",
    result={"temperature": 25, "condition": "sunny"}
)

# 註冊工具
context.register_tool(weather_func)

# 取得 API 格式的訊息列表
api_messages = context.get_messages_for_api()

# 取得 API 格式的工具列表
api_tools = context.get_tools_for_api()

# 清除對話
context.clear()

# 序列化與反序列化
data = context.to_dict()
context = ConversationContext.from_dict(data)
```

---

## 5. AIDownloader 下載器類別

處理 AI 生成內容的自動下載，支援同步與非同步操作。

### 5.1 初始化

```python
from pathlib import Path

# 建立下載器
downloader = AIDownloader(download_dir="./my_downloads")

# 或使用全域下載器
downloader = get_downloader()  # 使用預設目錄
downloader = get_downloader("./custom_dir")  # 使用自訂目錄
```

### 5.2 download_url - 同步下載 URL

```python
def download_url(
    self,
    url: str,                    # 要下載的 URL
    prefix: str = "download",    # 檔案名稱前綴
    extension: str = None        # 副檔名（自動偵測）
) -> Optional[Path]:             # 返回下載的檔案路徑

# 使用範例
filepath = downloader.download_url(
    "https://example.com/image.png",
    prefix="my_image"
)
if filepath:
    print(f"已下載: {filepath}")
else:
    print("下載失敗")

# 下載不含副檔名的 CDN URL
filepath = downloader.download_url(
    "https://pfst.cf2.poecdn.net/base/image/abc123",
    prefix="cdn_image"
)
# 會自動偵測檔案類型並設定副檔名
```

### 5.3 download_url_async - 非同步下載 URL

```python
async def download_url_async(
    self,
    url: str,
    prefix: str = "download",
    extension: str = None
) -> Optional[Path]:

# 使用範例
import asyncio

async def download_images():
    urls = [
        "https://example.com/img1.png",
        "https://example.com/img2.png",
        "https://example.com/img3.png"
    ]
    
    tasks = [downloader.download_url_async(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    for filepath in results:
        if filepath:
            print(f"已下載: {filepath}")

asyncio.run(download_images())
```

### 5.4 download_attachments_sync - 同步下載多個附件

```python
def download_attachments_sync(
    self,
    attachments: List[Attachment],
    prefix: str = "attachment"
) -> List[Path]:

# 使用範例
attachments = [
    Attachment(url="https://example.com/file1.png", content_type="image/png"),
    Attachment(url="https://example.com/file2.jpg", content_type="image/jpeg")
]

downloaded_files = downloader.download_attachments_sync(attachments)
for filepath in downloaded_files:
    print(f"已下載: {filepath}")
```

### 5.5 download_attachments_async - 非同步下載多個附件

```python
async def download_attachments_async(
    self,
    attachments: List[Attachment],
    prefix: str = "attachment"
) -> List[Path]:

# 使用範例
async def batch_download():
    attachments = [
        Attachment(url=url, content_type="image/png")
        for url in ["https://example.com/1.png", "https://example.com/2.png"]
    ]
    
    files = await downloader.download_attachments_async(attachments)
    return files

downloaded = asyncio.run(batch_download())
```

### 5.6 save_base64 - 儲存 Base64 資料

```python
def save_base64(
    self,
    base64_data: str,            # base64 編碼的資料
    prefix: str = "image",       # 檔案名稱前綴
    extension: str = ".png"      # 副檔名
) -> Optional[Path]:

# 使用範例
import base64

# 儲存 base64 圖片
image_b64 = base64.b64encode(image_bytes).decode()
filepath = downloader.save_base64(
    base64_data=image_b64,
    prefix="generated_image",
    extension=".png"
)

# 支援 data URL 格式
data_url = "data:image/png;base64,iVBORw0KGgo..."
filepath = downloader.save_base64(data_url, prefix="data_url_image")
```

### 5.7 save_text - 儲存文字內容

```python
def save_text(
    self,
    content: str,                # 文字內容
    prefix: str = "text",        # 檔案名稱前綴
    extension: str = ".txt"      # 副檔名
) -> Optional[Path]:

# 使用範例
filepath = downloader.save_text(
    content="# AI 生成的報告\n\n這是內容...",
    prefix="report",
    extension=".md"
)
print(f"已儲存: {filepath}")
```

### 5.8 extract_and_download_urls - 從文字擷取並下載 URL

支援多種 URL 格式，包括 Markdown 圖片語法和 CDN URL。

```python
def extract_and_download_urls(
    self,
    text: str,                   # 要分析的文字
    auto_download: bool = True   # 是否自動下載
) -> List[Tuple[str, Optional[Path]]]:

# 使用範例
response_text = """
這是生成的圖片：
![cat.png](https://pfst.cf2.poecdn.net/base/image/abc123?w=1024)

還有這個：
https://example.com/photo.jpg
"""

# 擷取並下載所有 URL
url_files = downloader.extract_and_download_urls(response_text, auto_download=True)
for url, filepath in url_files:
    if filepath:
        print(f"URL: {url[:50]}...")
        print(f"下載至: {filepath}")

# 僅擷取 URL（不下載）
url_files = downloader.extract_and_download_urls(response_text, auto_download=False)
urls = [url for url, _ in url_files]
print(f"找到 {len(urls)} 個 URL")
```

支援的 URL 格式：

- Markdown 圖片: `![alt](https://example.com/image.png)`
- 標準檔案 URL: `https://example.com/file.jpg`
- CDN URL: `https://pfst.cf2.poecdn.net/base/image/...`
- 支援的 CDN: poecdn.net, cloudfront.net, amazonaws.com, imgur.com, discordapp.com

### 5.9 extract_and_download_urls_async - 非同步版本

```python
async def extract_and_download_urls_async(
    self,
    text: str,
    auto_download: bool = True
) -> List[Tuple[str, Optional[Path]]]:

# 使用範例
async def process_response():
    response = "![image](https://example.com/img.png)"
    results = await downloader.extract_and_download_urls_async(response)
    return results

url_files = asyncio.run(process_response())
```

### 5.10 list_downloads - 列出所有下載檔案

```python
def list_downloads(self) -> List[Path]:

# 使用範例
files = downloader.list_downloads()
for f in files:
    print(f"- {f.name} ({f.stat().st_size / 1024:.1f} KB)")
```

### 5.11 get_download_stats - 取得下載統計

```python
def get_download_stats(self) -> Dict[str, Any]:

# 使用範例
stats = downloader.get_download_stats()
print(f"目錄: {stats['directory']}")
print(f"檔案數: {stats['file_count']}")
print(f"總大小: {stats['total_size_mb']} MB")

# 輸出範例:
# {
#     "directory": "./AIdownload",
#     "file_count": 15,
#     "total_size_bytes": 5242880,
#     "total_size_mb": 5.0
# }
```

---

## 6. ChatClient 聊天客戶端類別

主要的 AI 聊天客戶端，支援多種功能。

### 6.1 初始化

```python
client = ChatClient(
    model="GPT-5-nano",           # 模型名稱
    api_key=None,                 # API 金鑰（預設使用全域設定）
    download_dir="./AIdownload",  # 下載目錄
    auto_download_attachments=True # 自動下載附件
)

# 檢查客戶端屬性
print(client.model)
print(client.downloader.download_dir)
```

### 6.2 set_model - 切換模型

```python
def set_model(self, model: str):

# 使用範例
client = ChatClient(model="GPT-5-nano")
response = client.chat("你好")

# 切換到其他模型
client.set_model("GPT-5-mini")
response = client.chat("你好")  # 使用新模型

# 切換到 Ollama 模型
client.set_model("qwen3:0.6b")
```

### 6.3 register_function / execute_function - 函數註冊與執行

```python
def register_function(self, name: str, handler: Callable):
def execute_function(self, func_call: FunctionCall) -> Any:

# 使用範例
def calculate_sum(numbers: list) -> int:
    """計算數字總和"""
    return sum(numbers)

def get_current_time() -> str:
    """取得當前時間"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 註冊函數
client.register_function("calculate_sum", calculate_sum)
client.register_function("get_current_time", get_current_time)

# 手動執行函數
func_call = FunctionCall(
    id="call_1",
    name="calculate_sum",
    arguments={"numbers": [1, 2, 3, 4, 5]}
)
result = client.execute_function(func_call)
print(result)  # 輸出: 15
```

### 6.4 stream_chat - 串流聊天

```python
def stream_chat(
    self,
    message: str,                              # 使用者訊息
    system: Optional[str] = None,              # 系統提示
    role: str = "user",                        # 訊息角色
    context: Optional[ConversationContext] = None,  # 對話上下文
    images: List[ImageAttachment] = None,      # 圖片附件
    files: List[FileAttachment] = None,        # 檔案附件
    web_search: bool = False,                  # 啟用網路搜尋
    tools: List[FunctionDefinition] = None,    # 工具定義
    extra_body: Dict[str, Any] = None          # 額外 API 參數
) -> Generator[str, None, None]:

# 基本使用
for chunk in client.stream_chat("解釋量子力學"):
    print(chunk, end="", flush=True)
print()

# 使用系統提示
for chunk in client.stream_chat(
    "寫一首詩",
    system="你是一位專業的詩人，用優美的語言回答"
):
    print(chunk, end="", flush=True)

# 使用對話上下文
context = ConversationContext(
    system="你是一個 Python 專家",
    model="GPT-5-nano"
)
context.add_message("user", "什麼是裝飾器？")
context.add_message("assistant", "裝飾器是一種修改函數行為的方式...")

for chunk in client.stream_chat("給我一個例子", context=context):
    print(chunk, end="", flush=True)

# 帶圖片的聊天
images = [
    ImageAttachment.from_file("./photo.jpg"),
    ImageAttachment.from_url("https://example.com/image.png")
]
for chunk in client.stream_chat("描述這些圖片", images=images):
    print(chunk, end="", flush=True)

# 啟用網路搜尋
for chunk in client.stream_chat(
    "今天的新聞",
    web_search=True
):
    print(chunk, end="", flush=True)
```

### 6.5 stream_chat_with_attachments - 串流聊天並收集附件

這是最完整的串流方法，會自動處理附件和 URL 下載。

```python
def stream_chat_with_attachments(
    self,
    message: str,
    system: Optional[str] = None,
    role: str = "user",
    context: Optional[ConversationContext] = None,
    images: List[ImageAttachment] = None,
    files: List[FileAttachment] = None,
    web_search: bool = False,
    tools: List[FunctionDefinition] = None,
    extra_body: Dict[str, Any] = None,
    auto_download: bool = None  # 覆蓋預設的自動下載設定
) -> StreamingChatResponse:

# 基本使用
stream = client.stream_chat_with_attachments("生成一張貓的圖片")

# 迭代並輸出
for chunk, attachments in stream:
    print(chunk, end="", flush=True)

# 取得完整結果
result = stream.result
print(f"\n\n完整內容: {result.content}")
print(f"附件數: {len(result.attachments)}")
print(f"擷取的 URL: {result.extracted_urls}")
print(f"下載的檔案: {result.downloaded_files}")

# 使用 consume() 直接取得結果
stream = client.stream_chat_with_attachments("Hello")
result = stream.consume()  # 消費整個串流並返回 StreamResponse

# 只取得文字內容
stream = client.stream_chat_with_attachments("Hello")
text = stream.stream_to_string()  # 消費串流並返回字串
```

### 6.6 stream_chat_with_attachments_async - 非同步版本

```python
async def stream_chat_with_attachments_async(
    self,
    message: str,
    system: Optional[str] = None,
    role: str = "user",
    context: Optional[ConversationContext] = None,
    images: List[ImageAttachment] = None,
    files: List[FileAttachment] = None,
    web_search: bool = False,
    auto_download: bool = None
) -> StreamResponse:

# 使用範例
import asyncio

async def chat_async():
    client = ChatClient(model="GPT-5-nano")
    
    result = await client.stream_chat_with_attachments_async(
        "生成一張風景圖",
        auto_download=True
    )
    
    print(f"內容: {result.content}")
    print(f"下載: {result.downloaded_files}")
    return result

result = asyncio.run(chat_async())
```

### 6.7 chat - 非串流聊天

```python
def chat(
    self,
    message: str,
    system: Optional[str] = None,
    context: Optional[ConversationContext] = None,
    images: List[ImageAttachment] = None,
    files: List[FileAttachment] = None,
    web_search: bool = False,
    tools: List[FunctionDefinition] = None,
    auto_execute_tools: bool = False,       # 自動執行工具調用
    extra_body: Dict[str, Any] = None,
    collect_attachments: bool = False       # 收集附件資訊
) -> Union[str, Dict[str, Any], StreamResponse]:

# 基本使用
response = client.chat("你好！")
print(response)

# 使用系統提示
response = client.chat(
    "寫一個排序演算法",
    system="你是一個 Python 專家，用簡潔的程式碼回答"
)

# 收集附件
response = client.chat(
    "這張圖片裡有什麼？",
    images=[ImageAttachment.from_file("./photo.jpg")],
    collect_attachments=True
)
if isinstance(response, StreamResponse):
    print(response.content)
    print(f"附件: {response.attachments}")

# 使用工具
tools = [weather_func]  # 前面定義的 FunctionDefinition
response = client.chat(
    "台北現在的天氣如何？",
    tools=tools,
    auto_execute_tools=True
)
if isinstance(response, dict) and response.get("type") == "tool_calls":
    print(f"工具調用結果: {response['results']}")
```

### 6.8 chat_with_image - 圖片聊天便捷方法

```python
def chat_with_image(
    self,
    message: str,
    image_paths: List[str] = None,   # 本地圖片路徑
    image_urls: List[str] = None,    # 網路圖片 URL
    system: Optional[str] = None
) -> str:

# 使用範例
# 使用本地圖片
response = client.chat_with_image(
    "這張圖片裡有什麼動物？",
    image_paths=["./cat.jpg", "./dog.png"]
)

# 使用網路圖片
response = client.chat_with_image(
    "描述這些圖片的共同點",
    image_urls=[
        "https://example.com/img1.jpg",
        "https://example.com/img2.jpg"
    ]
)

# 混合使用
response = client.chat_with_image(
    "比較這些圖片",
    image_paths=["./local.jpg"],
    image_urls=["https://example.com/remote.jpg"],
    system="你是一個專業的圖片分析師"
)
```

### 6.9 generate_image - 圖片生成

```python
def generate_image(
    self,
    prompt: str,                              # 圖片描述
    model: str = None,                        # 模型（預設使用 DEFAULT_BOT_IMG）
    aspect: ImageAspect = ImageAspect.SQUARE, # 長寬比
    quality: ImageQuality = ImageQuality.MEDIUM,  # 品質
    auto_download: bool = True                # 自動下載
) -> Dict[str, Any]:

# 返回值結構:
# {
#     "content": "回應文字",
#     "urls": ["https://..."],
#     "downloaded_files": ["/path/to/file.png"],
#     "attachments": [Attachment(...)],
#     "model": "GPT-Image-1-Mini",
#     "prompt": "原始提示"
# }

# 基本使用
result = client.generate_image("一隻在沙灘上曬太陽的橘貓")
print(f"生成完成！")
print(f"URL: {result['urls']}")
print(f"已下載: {result['downloaded_files']}")

# 指定參數
result = client.generate_image(
    prompt="未來城市的天際線，賽博朋克風格",
    model="GPT-Image-1-Mini",
    aspect=ImageAspect.WIDE,    # 16:9
    quality=ImageQuality.HIGH,
    auto_download=True
)

# 不自動下載
result = client.generate_image(
    "一朵玫瑰花",
    auto_download=False
)
print(f"URL: {result['urls']}")  # 手動下載這些 URL
```

### 6.10 web_search_chat - 網路搜尋聊天

```python
def web_search_chat(
    self,
    query: str,
    system: Optional[str] = None,
    model: str = None  # 預設使用 "Web-Search"
) -> str:

# 使用範例
result = client.web_search_chat("2024年諾貝爾物理學獎得主是誰？")
print(result)

# 使用系統提示
result = client.web_search_chat(
    "台灣今天的天氣",
    system="請用簡潔的方式回答，包含溫度和天氣狀況"
)

# 使用特定模型
result = client.web_search_chat(
    "最新的 AI 技術新聞",
    model="Gemini-3-Pro"  # 必須在 WEB_SEARCH_MODELS 中
)
```

### 6.11 is_available - 檢查 API 可用性

```python
def is_available(self) -> bool:

# 使用範例
if client.is_available():
    print("API 連線正常")
    response = client.chat("Hello")
else:
    print("API 連線失敗")
```

---

## 7. 核心聊天函數

這些是模組級的便捷函數，無需建立 ChatClient 實例。

### 7.1 chatStreamOllama - 基本串流聊天

```python
def chatStreamOllama(
    message: str,
    model: str = DEFAULT_BOT,
    role: str = "user",
    stream: bool = True,
    system: Optional[str] = None,
    images: List[str] = None,      # 圖片路徑或 URL 列表
    web_search: bool = False
) -> Generator[str, None, None]:

# 基本使用
for chunk in chatStreamOllama("解釋機器學習"):
    print(chunk, end="", flush=True)
print()

# 使用特定模型和系統提示
for chunk in chatStreamOllama(
    "寫一個快速排序",
    model="GPT-5-mini",
    system="你是 Python 專家，只回答程式碼"
):
    print(chunk, end="", flush=True)

# 帶圖片
for chunk in chatStreamOllama(
    "這是什麼？",
    images=["./photo.jpg", "https://example.com/img.png"]
):
    print(chunk, end="", flush=True)

# 使用網路搜尋
for chunk in chatStreamOllama(
    "今天的股市新聞",
    model="Web-Search",
    web_search=True
):
    print(chunk, end="", flush=True)
```

### 7.2 chatStreamWithAttachments - 串流聊天並收集附件

```python
def chatStreamWithAttachments(
    message: str,
    model: str = DEFAULT_BOT,
    system: Optional[str] = None,
    images: List[str] = None,
    web_search: bool = False,
    auto_download: bool = True,
    download_dir: str = DEFAULT_DOWNLOAD_DIR
) -> StreamingChatResponse:

# 基本使用
stream = chatStreamWithAttachments("生成一張日落圖片")

for chunk, attachments in stream:
    print(chunk, end="", flush=True)

result = stream.result
print(f"\n下載的檔案: {result.downloaded_files}")

# 完整範例
stream = chatStreamWithAttachments(
    message="創作一張海邊風景圖",
    model="GPT-Image-1-Mini",
    system=None,
    images=None,
    web_search=False,
    auto_download=True,
    download_dir="./my_images"
)

# 處理串流
all_text = []
for chunk, atts in stream:
    all_text.append(chunk)
    print(chunk, end="", flush=True)

# 取得結果
result = stream.result
print(f"\n\n=== 結果 ===")
print(f"內容長度: {len(result.content)}")
print(f"附件數量: {len(result.attachments)}")
print(f"擷取的 URL: {result.extracted_urls}")
print(f"下載的檔案:")
for f in result.downloaded_files:
    print(f"  - {f}")
```

### 7.3 chatStreamWithAttachmentsAsync - 非同步版本

```python
async def chatStreamWithAttachmentsAsync(
    message: str,
    model: str = DEFAULT_BOT,
    system: Optional[str] = None,
    images: List[str] = None,
    web_search: bool = False,
    auto_download: bool = True,
    download_dir: str = DEFAULT_DOWNLOAD_DIR
) -> StreamResponse:

# 使用範例
import asyncio

async def main():
    result = await chatStreamWithAttachmentsAsync(
        "生成一張貓咪圖片",
        model="GPT-Image-1-Mini",
        auto_download=True
    )
    
    print(f"內容: {result.content}")
    print(f"下載: {result.downloaded_files}")
    return result

result = asyncio.run(main())

# 批量處理
async def batch_generate():
    prompts = ["一隻貓", "一隻狗", "一隻兔子"]
    tasks = [
        chatStreamWithAttachmentsAsync(f"畫{p}")
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(batch_generate())
```

### 7.4 askBot - 批量問答

```python
def askBot(
    questions: Union[str, List[str]],
    user: str = "You",
    Bot: str = "Assistant",
    model: str = DEFAULT_BOT,
    system: Optional[str] = None,
    verbose: bool = False,           # 是否輸出進度
    web_search: bool = False,
    collect_attachments: bool = False
) -> Generator[Tuple[str, str, Optional[List[Attachment]]], None, None]:

# 單一問題
for q, a, atts in askBot("什麼是 Python？"):
    print(f"問: {q}")
    print(f"答: {a}")

# 多個問題
questions = [
    "什麼是變數？",
    "什麼是函數？",
    "什麼是類別？"
]

for q, a, atts in askBot(questions, verbose=True):
    print(f"\n問題: {q}")
    print(f"回答: {a}")

# 收集附件
for q, a, atts in askBot(
    ["生成一張圖片"],
    model="GPT-Image-1-Mini",
    collect_attachments=True,
    verbose=True
):
    print(f"問: {q}")
    print(f"答: {a}")
    if atts:
        print(f"附件: {len(atts)} 個")

# 使用網路搜尋
for q, a, _ in askBot(
    ["今天的新聞頭條"],
    model="Web-Search",
    web_search=True,
    verbose=True
):
    print(a)
```

### 7.5 generateImage - 圖片生成便捷函數

```python
def generateImage(
    prompt: str,
    model: str = DEFAULT_BOT_IMG,
    aspect: str = "1:1",        # "1:1", "3:2", "2:3", "16:9", "auto"
    quality: str = "medium",    # "low", "medium", "high"
    auto_download: bool = True,
    download_dir: str = DEFAULT_DOWNLOAD_DIR
) -> Dict[str, Any]:

# 基本使用
result = generateImage("一隻可愛的柴犬")
print(f"圖片 URL: {result['urls']}")
print(f"下載位置: {result['downloaded_files']}")

# 指定參數
result = generateImage(
    prompt="壯觀的銀河系，攝影風格",
    model="GPT-Image-1-Mini",
    aspect="16:9",
    quality="high",
    auto_download=True,
    download_dir="./space_images"
)

# 不自動下載
result = generateImage(
    "簡單的幾何圖形",
    auto_download=False
)
# 手動處理 result['urls']
```

### 7.6 webSearch - 網路搜尋便捷函數

```python
def webSearch(
    query: str,
    model: str = "Web-Search",
    system: Optional[str] = None
) -> str:

# 基本使用
result = webSearch("最新的科技新聞")
print(result)

# 使用系統提示
result = webSearch(
    "台灣的天氣預報",
    system="請提供今天和明天的天氣預報，包含溫度和降雨機率"
)

# 使用不同模型
result = webSearch(
    "人工智慧的最新發展",
    model="Gemini-3-Pro"
)
```

### 7.7 chatWithImage - 圖片聊天便捷函數

```python
def chatWithImage(
    message: str,
    images: List[str],       # 圖片路徑或 URL 列表
    model: str = DEFAULT_BOT,
    system: Optional[str] = None
) -> str:

# 使用本地圖片
result = chatWithImage(
    "這張圖片裡有什麼？",
    images=["./photo1.jpg", "./photo2.png"]
)
print(result)

# 使用 URL
result = chatWithImage(
    "比較這兩張圖片",
    images=[
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg"
    ],
    model="GPT-5-nano"
)

# 混合使用
result = chatWithImage(
    "這些圖片的共同主題是什麼？",
    images=[
        "./local_image.jpg",
        "https://example.com/remote.png"
    ],
    system="你是專業的藝術評論家"
)
```

### 7.8 chatWithTools - 工具調用便捷函數

```python
def chatWithTools(
    message: str,
    tools: List[Dict[str, Any]],      # OpenAI 格式的工具定義
    model: str = DEFAULT_BOT,
    system: Optional[str] = None,
    handlers: Dict[str, Callable] = None,  # 函數處理器
    auto_execute: bool = True
) -> Union[str, Dict[str, Any]]:

# 定義工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "取得指定地點的天氣資訊",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名稱，例如：台北、東京"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "溫度單位"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "執行數學計算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "數學表達式，例如：2+3*4"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 定義處理器
def get_weather(location: str, unit: str = "celsius") -> str:
    # 實際應該呼叫天氣 API
    return f"{location} 的天氣是晴天，溫度 25°C"

def calculate(expression: str) -> str:
    try:
        result = eval(expression)  # 注意：實際使用時應該用更安全的方式
        return str(result)
    except:
        return "計算錯誤"

handlers = {
    "get_weather": get_weather,
    "calculate": calculate
}

# 使用（自動執行）
result = chatWithTools(
    "台北現在的天氣如何？然後幫我計算 15 * 8 + 32",
    tools=tools,
    handlers=handlers,
    auto_execute=True
)

if isinstance(result, dict):
    print(f"工具調用結果: {result['results']}")
else:
    print(f"回應: {result}")

# 不自動執行（手動處理）
result = chatWithTools(
    "東京的天氣如何？",
    tools=tools,
    auto_execute=False
)

if isinstance(result, dict) and result.get("type") == "tool_calls":
    for tc in result["tool_calls"]:
        print(f"需要調用: {tc.name}")
        print(f"參數: {tc.arguments}")
        # 手動執行函數
        if tc.name in handlers:
            output = handlers[tc.name](**tc.arguments)
            print(f"結果: {output}")
```

---

## 8. AIProcessor 內容處理器

提供各種 AI 輔助的文字處理功能。

### 8.1 初始化與設定

```python
processor = AIProcessor(model="GPT-5-nano")

# 切換模型
processor.set_model("GPT-5-mini")
```

### 8.2 process - 通用處理方法

```python
def process(
    self,
    context: str,          # 完整上下文
    part: str,             # 要處理的部分
    task: TaskType,        # 任務類型
    **kwargs               # 任務特定參數
) -> Union[str, List[str], Dict[str, Any]]:

# 使用範例
result = processor.process(
    context="這是一個 Python 程式...",
    part="def hello():\n    print('world')",
    task=TaskType.CODE_MODIFY,
    instruction="加入型別提示"
)
```

### 8.3 CODE_MODIFY - 程式碼修改

```python
# 直接使用 process
result = processor.process(
    context="這是一個處理使用者資料的模組",
    part="""
def process_user(name, age):
    if age > 18:
        return f"{name} is adult"
    return f"{name} is minor"
""",
    task=TaskType.CODE_MODIFY,
    instruction="加入完整的型別提示和文件字串"
)
print(result)

# 輸出可能是：
# def process_user(name: str, age: int) -> str:
#     """
#     處理使用者資料並返回描述字串。
#     
#     Args:
#         name: 使用者名稱
#         age: 使用者年齡
#     
#     Returns:
#         描述使用者年齡狀態的字串
#     """
#     if age > 18:
#         return f"{name} is adult"
#     return f"{name} is minor"
```

### 8.4 COMPLETE - 自動完成

```python
result = processor.process(
    context="這是一篇關於機器學習的文章",
    part="深度學習是機器學習的一個子領域，它使用",
    task=TaskType.COMPLETE,
    max_length=100
)
print(result)
# 輸出: "多層神經網路來學習資料的表示和特徵..."
```

### 8.5 SUGGEST - 生成建議

```python
# 程式碼建議
suggestions = processor.process(
    context="Python 資料處理模組",
    part="def process_",
    task=TaskType.SUGGEST,
    count=5
)
print(suggestions)
# 輸出: ['process_data', 'process_file', 'process_json', 'process_csv', 'process_input']

# 文字建議
suggestions = processor.process(
    context="撰寫電子郵件",
    part="感謝您的",
    task=TaskType.SUGGEST,
    count=3
)
print(suggestions)
# 輸出: ['感謝您的來信', '感謝您的耐心等待', '感謝您的支持']
```

### 8.6 SELECT - 選項選擇

```python
result = processor.process(
    context="使用者需要一個處理大量資料的解決方案",
    part="哪種資料結構最適合快速查詢？",
    task=TaskType.SELECT,
    options=[
        "鏈結串列 (Linked List)",
        "雜湊表 (Hash Table)",
        "陣列 (Array)",
        "二元搜尋樹 (BST)"
    ]
)
print(result)
# 輸出: {"selected": 1, "option": "雜湊表 (Hash Table)"}
```

### 8.7 REWRITE - 內容改寫

```python
# 專業風格
result = processor.process(
    context="商業報告",
    part="這個東西很好用，大家都喜歡。",
    task=TaskType.REWRITE,
    style="professional"
)
print(result)
# 輸出: "此產品具有優異的使用體驗，廣受用戶好評。"

# 簡化風格
result = processor.process(
    context="給小學生的說明",
    part="光合作用是植物利用光能將二氧化碳和水轉化為葡萄糖和氧氣的生化過程。",
    task=TaskType.REWRITE,
    style="simple and easy to understand for children"
)
print(result)

# 幽默風格
result = processor.process(
    context="社群媒體貼文",
    part="今天的會議很長。",
    task=TaskType.REWRITE,
    style="humorous and casual"
)
print(result)
```

### 8.8 SUMMARIZE - 內容摘要

```python
result = processor.process(
    context="",
    part="""
    人工智慧（AI）是計算機科學的一個分支，致力於創建能夠執行通常需要人類智慧的任務的系統。
    這些任務包括視覺感知、語音識別、決策制定和語言翻譯等。AI 系統可以分為兩大類：
    窄 AI（專注於特定任務）和通用 AI（具有人類水平的認知能力）。
    目前大多數 AI 應用都屬於窄 AI 範疇，如語音助手、推薦系統和自動駕駛汽車。
    """,
    task=TaskType.SUMMARIZE,
    max_words=30
)
print(result)
# 輸出: "人工智慧是創建執行人類智慧任務系統的領域，分為窄AI和通用AI兩類。"
```

### 8.9 TRANSLATE - 翻譯

```python
# 翻譯成英文
result = processor.process(
    context="",
    part="今天天氣真好，我們去公園散步吧！",
    task=TaskType.TRANSLATE,
    target_lang="English"
)
print(result)
# 輸出: "The weather is so nice today, let's go for a walk in the park!"

# 翻譯成日文
result = processor.process(
    context="",
    part="謝謝你的幫助",
    task=TaskType.TRANSLATE,
    target_lang="Japanese"
)
print(result)
# 輸出: "ご協力ありがとうございます"

# 翻譯成繁體中文
result = processor.process(
    context="",
    part="Hello, how are you?",
    task=TaskType.TRANSLATE,
    target_lang="Traditional Chinese"
)
print(result)
# 輸出: "你好，你好嗎？"
```

### 8.10 aiProcess - 便捷函數

```python
def aiProcess(
    context: str,
    part: str,
    task: Union[str, TaskType],  # 可以用字串或枚舉
    model: str = DEFAULT_BOT,
    **kwargs
) -> Any:

# 使用字串指定任務
result = aiProcess(
    context="",
    part="Hello, world!",
    task="translate",  # 等同於 TaskType.TRANSLATE
    target_lang="Chinese"
)
print(result)

# 使用枚舉
suggestions = aiProcess(
    context="Python 程式設計",
    part="import ",
    task=TaskType.SUGGEST,
    model="GPT-5-mini",
    count=5
)
print(suggestions)

# 程式碼修改
modified_code = aiProcess(
    context="Web 應用程式",
    part="def get_user(id):\n    return db.query(id)",
    task="code_modify",
    instruction="加入錯誤處理和日誌記錄"
)
print(modified_code)
```

---

## 9. ChatTerminal 互動終端

功能完整的互動式聊天終端介面。

### 9.1 初始化與啟動

```python
# 使用類別
terminal = ChatTerminal(
    user="Alice",                    # 使用者名稱
    Bot="Claude",                    # 機器人名稱
    model="GPT-5-nano",              # 使用的模型
    system="你是一個友善的助手",     # 系統提示
    stream=True,                     # 啟用串流
    download_dir="./AIdownload",     # 下載目錄
    auto_download=True               # 自動下載附件
)
terminal.run()

# 使用便捷函數
chatTerminal(
    user="使用者",
    Bot="助手",
    model="GPT-5-nano"
)
```

### 9.2 終端命令列表

| 命令 | 說明 | 範例 |
|------|------|------|
| `/? ` | 快速說明 | `/? ` |
| `/help` | 完整說明 | `/help` |
| `/exit`, `/quit`, `/q` | 退出 | `/exit` |
| `/model [name\|number]` | 切換模型 | `/model GPT-5-mini` 或 `/model 4` |
| `/models` | 列出所有模型 | `/models` |
| `/system [prompt]` | 設定系統提示 | `/system 你是專家` |
| `/clear` | 清除螢幕 | `/clear` |
| `/reset` | 重設對話 | `/reset` |
| `/history [n]` | 顯示歷史 | `/history 20` |
| `/save [filename]` | 儲存對話 | `/save chat.json` |
| `/load <filename>` | 載入對話 | `/load chat.json` |
| `/export [filename]` | 匯出純文字 | `/export chat.txt` |
| `/stream` | 切換串流模式 | `/stream` |
| `/markdown` | 切換 Markdown 渲染 | `/markdown` |
| `/retry` | 重試上一訊息 | `/retry` |
| `/stats` | 顯示統計 | `/stats` |
| `/image <prompt>` | 生成圖片 | `/image 一隻貓` |
| `/attach <path\|url>` | 附加圖片 | `/attach ./photo.jpg` |
| `/search <query>` | 網路搜尋 | `/search 今天新聞` |
| `/downloads` | 下載資訊 | `/downloads` |
| `/autodownload` | 切換自動下載 | `/autodownload` |
| `/tools` | 顯示工具 | `/tools` |

### 9.3 使用範例

```
╔═══════════════════════════════════════════════════════╗
║      █████╗ ██╗     ██████╗██╗  ██╗ █████╗ ████████╗  ║
║     ██╔══██╗██║    ██╔════╝██║  ██║██╔══██╗╚══██╔══╝  ║
║     ███████║██║    ██║     ███████║███████║   ██║     ║
║     ██╔══██║██║    ██║     ██╔══██║██╔══██║   ██║     ║
║     ██║  ██║██║    ╚██████╗██║  ██║██║  ██║   ██║     ║
║     ╚═╝  ╚═╝╚═╝     ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝     ║
║                        v3.2                           ║
╚═══════════════════════════════════════════════════════╝

  Model: GPT-5-nano  |  Type /? for help  |  /exit to quit
