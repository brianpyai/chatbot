# ASMhttpServerJIT 完整使用說明

## 目錄

1. [模組概述](#1-模組概述)
2. [安裝與依賴](#2-安裝與依賴)
3. [快速開始](#3-快速開始)
4. [核心類別詳解](#4-核心類別詳解)
   - [ASMHTTPServer](#41-asmhttpserver)
   - [HttpRequest](#42-httprequest)
   - [HttpResponse](#43-httpresponse)
   - [HttpRouter](#44-httprouter)
   - [HTTPError](#45-httperror)
5. [路由系統](#5-路由系統)
6. [靜態檔案服務](#6-靜態檔案服務)
7. [請求處理](#7-請求處理)
8. [回應處理](#8-回應處理)
9. [HTTPS/TLS 配置](#9-httpstls-配置)
10. [快取系統](#10-快取系統)
11. [效能基準測試](#11-效能基準測試)
12. [環境變數配置](#12-環境變數配置)
13. [完整應用範例](#13-完整應用範例)
14. [API 參考](#14-api-參考)

---

## 1. 模組概述

**ASMhttpServerJIT** 是一個高效能的 HTTP/1.1 伺服器實作，專為 Python 應用程式設計。

### 主要特性

| 特性 | 說明 |
|------|------|
| **HTTP/1.1 完整支援** | 支援 Content-Length、Keep-Alive、Pipelining |
| **執行緒池架構** | 固定 worker pool，避免每連線建立執行緒的開銷 |
| **靜態檔案服務** | 內建 ETag、Last-Modified、304 Not Modified、Cache-Control |
| **LRU 快取** | 全域 body cache + 單次讀取（single-flight）機制 |
| **嚴格 URL 解碼** | 無效 percent-encoding 回傳 400 Bad Request |
| **HTTPS 支援** | 可配置 TLS 憑證 |
| **效能基準測試** | 內建大規模效能測試工具 |

### 版本資訊

```python
import ASMhttpServerJIT as http

print(http.__version__)  # 2.1.1-HTTPJIT-BASE4.1
print(http.get_module_info())
```

---

## 2. 安裝與依賴

### 必要依賴

- Python 3.8+
- 標準函式庫（無需額外安裝）

### 可選依賴（增強效能）

```python
# 可選依賴狀態檢查
from ASMhttpServerJIT import get_module_info

info = get_module_info()
print(info["backends"])
# {
#     "ASMComputingJIT": {"available": True/False, ...},
#     "ASMioJIT": {"available": True/False, ...},
#     "ASMcryptoJIT": {"available": True/False, ...},
#     "numpy": {"available": True/False, ...}
# }
```

| 依賴 | 用途 |
|------|------|
| `ASMComputingJIT` | 預編譯表加速 |
| `ASMioJIT` | JIT 快速計數器 |
| `ASMcryptoJIT` | 安全比較函式、加密功能 |
| `numpy` | 百分位數計算加速 |

---

## 3. 快速開始

### 最簡單的伺服器

```python
from ASMhttpServerJIT import ASMHTTPServer, HttpRequest

# 建立伺服器
server = ASMHTTPServer(
    host="0.0.0.0",
    port=8080
)

# 新增路由
@server.route("/hello", methods="GET")
def hello(req: HttpRequest) -> str:
    return "Hello, World!"

@server.route("/api/data", methods=["GET", "POST"])
def api_data(req: HttpRequest) -> dict:
    if req.method == "GET":
        return {"message": "Data retrieved", "method": req.method}
    else:
        data = req.json()
        return {"message": "Data received", "data": data}

# 啟動伺服器（阻塞式）
server.serve_forever()
```

### 非阻塞式啟動

```python
from ASMhttpServerJIT import ASMHTTPServer
import time

server = ASMHTTPServer(host="0.0.0.0", port=8080)

@server.route("/ping")
def ping(req):
    return {"pong": True}

# 非阻塞啟動
server.start()
print(f"Server running on port {server.port}")

try:
    while True:
        # 主程式可以做其他事情
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()
    print("Server stopped")
```

---

## 4. 核心類別詳解

### 4.1 ASMHTTPServer

主要的伺服器類別，負責接受連線、解析請求、分派路由。

#### 建構參數

```python
server = ASMHTTPServer(
    # 網路配置
    host="0.0.0.0",           # 綁定位址
    port=8080,                 # 綁定埠口（0 表示自動分配）
    backlog=512,               # listen backlog 大小
    tcp_nodelay=True,          # 啟用 TCP_NODELAY
    
    # 靜態檔案配置
    static_root="./static",    # 靜態檔案根目錄
    docs_prefix="/docs/",      # 靜態檔案 URL 前綴
    enable_static=True,        # 是否啟用靜態檔案服務
    
    # HTTPS 配置
    enable_https=False,        # 是否啟用 HTTPS
    tls_certfile="./cert.pem", # TLS 憑證檔案路徑
    tls_keyfile="./key.pem",   # TLS 私鑰檔案路徑
    ssl_context=None,          # 自訂 SSLContext（優先於 certfile/keyfile）
    
    # 請求限制
    max_request_body=8*1024*1024,  # 最大請求 body 大小 (8MB)
    max_header_bytes=64*1024,      # 最大 header 大小 (64KB)
    max_target_bytes=8*1024,       # 最大 URL 長度 (8KB)
    
    # 連線配置
    keep_alive_timeout=5.0,    # Keep-Alive 逾時秒數
    
    # Worker 配置
    worker_count=None,         # Worker 數量（None=自動：min(256, max(16, CPU*8))）
    conn_queue_size=4096,      # 連線佇列大小
    
    # 其他
    server_header="ASMHTTPServerJIT/2.1",  # Server header 值
    enable_builtin_routes=True,            # 啟用內建路由
)
```

#### 主要方法

```python
# 啟動伺服器（非阻塞）
server.start()

# 停止伺服器
server.stop()

# 啟動伺服器（阻塞，直到 Ctrl+C）
server.serve_forever()

# 新增路由
server.add_route("/path", methods=["GET", "POST"], handler=my_handler, exact=True)

# 裝飾器方式新增路由
@server.route("/path", methods="GET", exact=True)
def my_handler(req: HttpRequest):
    return "OK"

# 掛載靜態檔案目錄
server.mount_static("/assets/", "./public")
```

#### 內建路由

當 `enable_builtin_routes=True` 時，自動註冊以下路由：

| 路由 | 方法 | 說明 |
|------|------|------|
| `/health` | GET | 健康檢查，回傳 `{"ok": true, "time": ...}` |
| `/jit/info` | GET | 模組資訊 |
| `/asm/info` | GET | 模組資訊（別名） |
| `/asm/cache` | GET | 快取統計資訊 |
| `/routes` | GET | 所有已註冊路由列表 |

---

### 4.2 HttpRequest

HTTP 請求物件，包含解析後的請求資訊。

#### 屬性

```python
@dataclass
class HttpRequest:
    method: str                        # HTTP 方法（GET, POST, ...）
    target: str                        # 原始 request-target
    path: str                          # 解碼後的 URL 路徑
    query_string: str                  # 查詢字串（不含 ?）
    http_version: str                  # HTTP 版本（HTTP/1.1, HTTP/1.0）
    headers: Dict[str, str]            # 標準化後的 headers
    body: bytes                        # 請求 body
    client_addr: Optional[Tuple[str, int]]  # 客戶端位址 (IP, port)
    server: Optional[ASMHTTPServer]    # 伺服器參考
```

#### 方法

```python
def header(self, name: str, default: Optional[str] = None) -> Optional[str]:
    """取得 header 值（不區分大小寫）"""
    pass

def json(self) -> Any:
    """將 body 解析為 JSON"""
    pass

def text(self, encoding: str = "utf-8", errors: str = "strict") -> str:
    """將 body 解碼為字串"""
    pass

def form(self, encoding: str = "utf-8") -> Dict[str, List[str]]:
    """將 body 解析為 form 表單（application/x-www-form-urlencoded）"""
    pass
```

#### 使用範例

```python
@server.route("/api/user", methods=["GET", "POST", "PUT"])
def user_handler(req: HttpRequest):
    # 取得 header
    content_type = req.header("Content-Type")
    auth = req.header("Authorization", default="")
    
    # 取得客戶端資訊
    if req.client_addr:
        client_ip, client_port = req.client_addr
        print(f"Request from {client_ip}:{client_port}")
    
    # 根據方法處理
    if req.method == "GET":
        # 解析 query string
        from urllib.parse import parse_qs
        params = parse_qs(req.query_string)
        user_id = params.get("id", [None])[0]
        return {"user_id": user_id}
    
    elif req.method == "POST":
        # 解析 JSON body
        try:
            data = req.json()
            return {"received": data}
        except ValueError as e:
            from ASMhttpServerJIT import HTTPError
            raise HTTPError(400, f"Invalid JSON: {e}")
    
    elif req.method == "PUT":
        # 解析表單
        if "application/x-www-form-urlencoded" in (content_type or ""):
            form_data = req.form()
            return {"form": form_data}
        else:
            # 純文字 body
            text = req.text()
            return {"text": text}
```

---

### 4.3 HttpResponse

HTTP 回應物件，用於自訂回應內容。

#### 建構參數

```python
response = HttpResponse(
    status=200,                # HTTP 狀態碼
    headers={},                # 回應 headers
    body=b"",                  # 回應 body
    http_version="HTTP/1.1"   # HTTP 版本
)
```

#### body 支援的類型

| 類型 | 說明 |
|------|------|
| `bytes` / `bytearray` | 直接發送二進位資料 |
| `str` | 自動編碼為 UTF-8 |
| `Iterable[bytes/str]` | 串流發送（自動 chunked encoding） |
| `_SendfileBody(path)` | 使用 sendfile 發送檔案 |
| `None` | 無 body |

#### 方法

```python
def set_header(self, name: str, value: str) -> None:
    """設定 header（自動正規化名稱）"""
    pass

def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
    """取得 header 值"""
    pass

def send(self, transport, *, method="GET", server_header=None, keep_alive=True) -> None:
    """發送回應到 transport（內部使用）"""
    pass
```

#### 使用範例

```python
from ASMhttpServerJIT import HttpResponse, HttpRequest

@server.route("/custom-response")
def custom_response(req: HttpRequest) -> HttpResponse:
    response = HttpResponse(status=201)
    response.set_header("Content-Type", "application/json")
    response.set_header("X-Custom-Header", "my-value")
    response.body = b'{"created": true}'
    return response

@server.route("/stream")
def stream_response(req: HttpRequest) -> HttpResponse:
    """串流回應（chunked transfer encoding）"""
    def generate():
        for i in range(10):
            yield f"data: {i}\n".encode("utf-8")
    
    response = HttpResponse(status=200, body=generate())
    response.set_header("Content-Type", "text/event-stream")
    return response

@server.route("/download")
def download_file(req: HttpRequest) -> HttpResponse:
    """檔案下載"""
    from pathlib import Path
    response = HttpResponse(status=200)
    response.set_header("Content-Type", "application/octet-stream")
    response.set_header("Content-Disposition", 'attachment; filename="data.bin"')
    response.body = Path("./data.bin").read_bytes()
    return response
```

---

### 4.4 HttpRouter

路由器，負責路由匹配與分派。

#### 方法

```python
def add_route(
    self, 
    path: str,                          # URL 路徑（必須以 / 開頭）
    methods: Union[str, List[str]],     # HTTP 方法
    handler: Callable[[HttpRequest], Any],  # 處理函式
    *, 
    exact: bool = True                  # True=精確匹配, False=前綴匹配
) -> None:
    """新增路由"""
    pass

def resolve(self, path: str, method: str) -> HttpRoute:
    """解析路由（內部使用）
    
    Raises:
        HTTPError(404): 找不到路由
        HTTPError(405): 找到路由但方法不允許
    """
    pass

def list_routes(self) -> List[HttpRoute]:
    """列出所有已註冊路由"""
    pass
```

#### 路由匹配規則

```python
# 精確匹配（exact=True，預設）
router.add_route("/users", "GET", list_users, exact=True)
# 僅匹配 /users，不匹配 /users/123

# 前綴匹配（exact=False）
router.add_route("/api/", ["GET", "POST"], api_handler, exact=False)
# 匹配 /api/, /api/users, /api/users/123 等

# 方法萬用字元
router.add_route("/any", "*", any_method_handler)
# 匹配任何 HTTP 方法
```

#### 路由優先順序

1. 精確匹配優先於前綴匹配
2. 前綴匹配按路徑長度降序（最長前綴優先）

```python
router.add_route("/docs/", "GET", docs_handler, exact=False)
router.add_route("/docs/api/", "GET", api_docs_handler, exact=False)

# GET /docs/api/users -> api_docs_handler（更長的前綴優先）
# GET /docs/guide     -> docs_handler
```

---

### 4.5 HTTPError

HTTP 錯誤例外，用於在 handler 中回傳錯誤回應。

#### 建構

```python
from ASMhttpServerJIT import HTTPError

raise HTTPError(
    status=404,                    # HTTP 狀態碼
    message="Resource not found", # 錯誤訊息
    headers={"X-Error": "true"}   # 額外 headers
)
```

#### 使用範例

```python
from ASMhttpServerJIT import HTTPError, HttpRequest

@server.route("/api/item/<id>")
def get_item(req: HttpRequest):
    item_id = req.path.split("/")[-1]
    
    # 驗證
    if not item_id.isdigit():
        raise HTTPError(400, "Invalid item ID")
    
    # 授權檢查
    auth = req.header("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPError(401, "Unauthorized", headers={
            "WWW-Authenticate": 'Bearer realm="api"'
        })
    
    # 查找資源
    item = database.get(int(item_id))
    if item is None:
        raise HTTPError(404, f"Item {item_id} not found")
    
    return item
```

#### 錯誤回應格式

HTTPError 會被轉換為 JSON 回應：

```json
{
    "status": 404,
    "error": "Not Found",
    "message": "Item 123 not found"
}
```

---

## 5. 路由系統

### 基本路由註冊

```python
from ASMhttpServerJIT import ASMHTTPServer, HttpRequest

server = ASMHTTPServer(port=8080)

# 方法 1: 裝飾器
@server.route("/hello", methods="GET")
def hello(req: HttpRequest):
    return "Hello!"

# 方法 2: 直接呼叫
def goodbye(req: HttpRequest):
    return "Goodbye!"

server.add_route("/goodbye", methods=["GET", "POST"], handler=goodbye)

# 方法 3: Lambda
server.add_route("/ping", "GET", lambda req: {"pong": True})
```

### 多方法處理

```python
@server.route("/resource", methods=["GET", "POST", "PUT", "DELETE"])
def resource_handler(req: HttpRequest):
    if req.method == "GET":
        return {"action": "list"}
    elif req.method == "POST":
        data = req.json()
        return {"action": "create", "data": data}
    elif req.method == "PUT":
        data = req.json()
        return {"action": "update", "data": data}
    elif req.method == "DELETE":
        return {"action": "delete"}
```

### 前綴路由（API 版本控制）

```python
# API v1
@server.route("/api/v1/", methods=["GET", "POST"], exact=False)
def api_v1(req: HttpRequest):
    sub_path = req.path[len("/api/v1/"):]
    return {"version": 1, "path": sub_path}

# API v2
@server.route("/api/v2/", methods=["GET", "POST"], exact=False)
def api_v2(req: HttpRequest):
    sub_path = req.path[len("/api/v2/"):]
    return {"version": 2, "path": sub_path}
```

### 路由參數解析

伺服器本身不內建路由參數解析，但可以自行實作：

```python
import re

@server.route("/users/", methods="GET", exact=False)
def users_handler(req: HttpRequest):
    # 解析路徑 /users/{id}
    match = re.match(r"^/users/(\d+)/?$", req.path)
    if match:
        user_id = int(match.group(1))
        return {"user_id": user_id}
    
    # 解析路徑 /users/{id}/posts
    match = re.match(r"^/users/(\d+)/posts/?$", req.path)
    if match:
        user_id = int(match.group(1))
        return {"user_id": user_id, "resource": "posts"}
    
    # 列表
    if req.path == "/users/" or req.path == "/users":
        return {"users": []}
    
    raise HTTPError(404, "Not Found")
```

---

## 6. 靜態檔案服務

### 自動掛載

```python
server = ASMHTTPServer(
    port=8080,
    static_root="./static",    # 靜態檔案目錄
    docs_prefix="/docs/",      # URL 前綴
    enable_static=True
)

# 目錄結構:
# ./static/
#   ├── index.html      -> /docs/
#   ├── style.css       -> /docs/style.css
#   └── js/
#       └── app.js      -> /docs/js/app.js
```

### 手動掛載多個目錄

```python
server = ASMHTTPServer(port=8080, enable_static=False)

# 掛載多個靜態目錄
server.mount_static("/assets/", "./public/assets")
server.mount_static("/uploads/", "./data/uploads")
server.mount_static("/docs/", "./documentation")
```

### 快取控制

```python
import os

# 透過環境變數設定 Cache-Control max-age（預設 600 秒）
os.environ["ASMHTTP_STATIC_MAX_AGE"] = "3600"  # 1 小時

server = ASMHTTPServer(port=8080)
```

### ETag 與 304 Not Modified

伺服器自動處理：

1. **ETag 生成**：基於檔案修改時間和大小
2. **If-None-Match**：客戶端傳送 ETag，伺服器比對後回傳 304
3. **If-Modified-Since**：客戶端傳送時間戳，伺服器比對後回傳 304

```python
# 客戶端請求
# GET /docs/style.css HTTP/1.1
# If-None-Match: W/"18a1b2c3d4e5-1234"

# 伺服器回應（檔案未變更）
# HTTP/1.1 304 Not Modified
# ETag: W/"18a1b2c3d4e5-1234"
# Last-Modified: Thu, 01 Jan 2025 00:00:00 GMT
```

### HEAD 請求最佳化

對於 HEAD 請求，伺服器不會讀取檔案內容，僅回傳 metadata：

```python
# HEAD /docs/large-file.zip HTTP/1.1
# 
# HTTP/1.1 200 OK
# Content-Type: application/zip
# Content-Length: 1073741824
# ETag: W/"..."
# Last-Modified: ...
# (無 body)
```

---

## 7. 請求處理

### JSON 請求

```python
@server.route("/api/data", methods="POST")
def handle_json(req: HttpRequest):
    # 驗證 Content-Type
    content_type = req.header("Content-Type") or ""
    if "application/json" not in content_type:
        raise HTTPError(415, "Unsupported Media Type")
    
    # 解析 JSON
    try:
        data = req.json()
    except ValueError as e:
        raise HTTPError(400, f"Invalid JSON: {e}")
    
    # 驗證必要欄位
    if "name" not in data:
        raise HTTPError(400, "Missing required field: name")
    
    return {"received": data, "status": "ok"}
```

### 表單請求

```python
@server.route("/api/form", methods="POST")
def handle_form(req: HttpRequest):
    content_type = req.header("Content-Type") or ""
    
    if "application/x-www-form-urlencoded" in content_type:
        form_data = req.form()
        # form_data = {"name": ["John"], "tags": ["a", "b"]}
        name = form_data.get("name", [""])[0]
        tags = form_data.get("tags", [])
        return {"name": name, "tags": tags}
    
    raise HTTPError(415, "Unsupported Media Type")
```

### 二進位資料

```python
@server.route("/api/upload", methods="POST")
def handle_upload(req: HttpRequest):
    # 取得原始 bytes
    data = req.body
    
    # 計算雜湊
    import hashlib
    digest = hashlib.sha256(data).hexdigest()
    
    # 儲存檔案
    from pathlib import Path
    Path(f"./uploads/{digest}.bin").write_bytes(data)
    
    return {
        "size": len(data),
        "sha256": digest
    }
```

### Query String 解析

```python
from urllib.parse import parse_qs

@server.route("/api/search", methods="GET")
def search(req: HttpRequest):
    # 解析 query string
    params = parse_qs(req.query_string, keep_blank_values=True)
    
    # 取得參數
    query = params.get("q", [""])[0]
    page = int(params.get("page", ["1"])[0])
    limit = int(params.get("limit", ["10"])[0])
    
    return {
        "query": query,
        "page": page,
        "limit": limit,
        "results": []
    }
```

---

## 8. 回應處理

### 回傳類型自動轉換

Handler 可以回傳多種類型，伺服器會自動轉換：

```python
# 1. dict/list -> JSON
@server.route("/json")
def return_json(req):
    return {"key": "value"}
# Content-Type: application/json

# 2. str -> 純文字
@server.route("/text")
def return_text(req):
    return "Hello World"
# Content-Type: text/plain; charset=utf-8

# 3. bytes -> 二進位
@server.route("/binary")
def return_binary(req):
    return b"\x00\x01\x02\x03"
# Content-Type: application/octet-stream

# 4. Path -> 檔案（含 ETag/Last-Modified）
@server.route("/file")
def return_file(req):
    from pathlib import Path
    return Path("./data/file.txt")
# Content-Type: 根據副檔名自動判斷

# 5. Iterable -> 串流（chunked）
@server.route("/stream")
def return_stream(req):
    def gen():
        for i in range(100):
            yield f"line {i}\n"
    return gen()
# Transfer-Encoding: chunked

# 6. HttpResponse -> 完全控制
@server.route("/custom")
def return_custom(req):
    return HttpResponse(
        status=201,
        headers={"X-Custom": "value"},
        body=b"created"
    )
```

### 重導向

```python
from ASMhttpServerJIT import HttpResponse

@server.route("/old-path")
def redirect(req):
    response = HttpResponse(status=302)
    response.set_header("Location", "/new-path")
    return response

@server.route("/permanent-redirect")
def permanent_redirect(req):
    response = HttpResponse(status=301)
    response.set_header("Location", "https://example.com/new-location")
    return response
```

### CORS 處理

```python
@server.route("/api/", methods=["GET", "POST", "OPTIONS"], exact=False)
def api_with_cors(req: HttpRequest):
    # 處理 preflight
    if req.method == "OPTIONS":
        response = HttpResponse(status=204)
        response.set_header("Access-Control-Allow-Origin", "*")
        response.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.set_header("Access-Control-Max-Age", "86400")
        return response
    
    # 實際請求
    result = {"data": "..."}
    response = HttpResponse(
        status=200,
        body=json.dumps(result).encode()
    )
    response.set_header("Content-Type", "application/json")
    response.set_header("Access-Control-Allow-Origin", "*")
    return response
```

---

## 9. HTTPS/TLS 配置

### 使用憑證檔案

```python
server = ASMHTTPServer(
    host="0.0.0.0",
    port=443,
    enable_https=True,
    tls_certfile="./ssl/cert.pem",
    tls_keyfile="./ssl/key.pem"
)

server.serve_forever()
```

### 使用自訂 SSLContext

```python
import ssl

# 建立自訂 SSLContext
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.minimum_version = ssl.TLSVersion.TLSv1_2
ctx.load_cert_chain(
    certfile="./ssl/cert.pem",
    keyfile="./ssl/key.pem",
    password="optional-password"
)

# 進階設定
ctx.set_ciphers("ECDHE+AESGCM:DHE+AESGCM")
ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3

server = ASMHTTPServer(
    host="0.0.0.0",
    port=443,
    enable_https=True,
    ssl_context=ctx
)
```

### 生成自簽憑證（開發用）

```bash
# 生成私鑰和自簽憑證
openssl req -x509 -newkey rsa:4096 \
    -keyout key.pem \
    -out cert.pem \
    -days 365 \
    -nodes \
    -subj "/CN=localhost"
```

---

## 10. 快取系統

### 快取配置

```python
import os

# 啟用/停用快取
os.environ["ASMHTTP_BODY_CACHE_DISABLED"] = "0"  # 0=啟用, 1=停用

# 設定快取大小
os.environ["ASMHTTP_BODY_CACHE_MB"] = "512"  # 512 MB
# 或
os.environ["ASMHTTP_BODY_CACHE_BYTES"] = "536870912"  # 512 * 1024 * 1024
```

### 快取統計

```python
from ASMhttpServerJIT import _HTTP_BODY_CACHE

if _HTTP_BODY_CACHE:
    stats = _HTTP_BODY_CACHE.stats()
    print(stats)
    # {
    #     "name": "asmhttp-body",
    #     "max_bytes": 268435456,
    #     "used_bytes": 12345678,
    #     "entries": 42,
    #     "hits": 10000,
    #     "misses": 500,
    #     "hit_rate": 0.952,
    #     "min_cached_bytes": 1024
    # }
```

### 快取行為

| 功能 | 說明 |
|------|------|
| **LRU 淘汰** | 當快取滿時，淘汰最久未使用的項目 |
| **單次讀取** | 多個請求同時讀取同一檔案時，僅執行一次磁碟讀取 |
| **自適應閾值** | 根據命中率自動調整 `min_cached_bytes` |
| **JSON/文字快取** | API 回應的 JSON 序列化結果也會被快取 |

---

## 11. 效能基準測試

### 執行內建基準測試

```python
from ASMhttpServerJIT import HTTPJITPerformanceBenchmark

# 執行所有基準測試（正常模式）
HTTPJITPerformanceBenchmark.run_all(heavy=False)

# 執行大規模基準測試
HTTPJITPerformanceBenchmark.run_all(heavy=True)
```

### 基準測試項目

| 測試 | 說明 |
|------|------|
| **Benchmark 1** | 靜態檔案吞吐量（connection-close） |
| **Benchmark 2A** | 動態 JSON API（connection-close） |
| **Benchmark 2B** | 動態 JSON API（keep-alive 單連線） |
| **Benchmark 3A** | 併發客戶端（connection-close） |
| **Benchmark 3B** | 併發客戶端（keep-alive） |

### 環境變數控制

```bash
# 是否執行基準測試
export ASMHTTPJIT_RUN_BENCH=1

# 是否執行大規模測試
export ASMHTTPJIT_BENCH_HEAVY=0

# 是否使用獨立子程序（避免 GIL 干擾）
export ASMHTTPJIT_BENCH_ISOLATE=1

# 基準測試 Worker 數量
export ASMHTTPJIT_BENCH_WORKERS=64

# 基準測試客戶端逾時
export ASMHTTPJIT_BENCH_CLIENT_TIMEOUT=30
```

### 輸出範例

```
======================================================================
HTTP Benchmark 2B: Dynamic JSON API throughput [/api/bench] [keep-alive single conn]
======================================================================
requests: 20000/20000, errors: 0, time: 2.34 s
QPS: 8,547 req/s
P50: 0.11 ms
P95: 0.18 ms
P99: 0.25 ms
```

---

## 12. 環境變數配置

### 完整環境變數列表

| 變數 | 預設值 | 說明 |
|------|--------|------|
| `ASMHTTPJIT_WORKERS` | `min(256, max(16, CPU*8))` | Worker 執行緒數量 |
| `ASMHTTP_BODY_CACHE_DISABLED` | `0` | 停用快取（1=停用） |
| `ASMHTTP_BODY_CACHE_MB` | `256` | 快取大小（MB） |
| `ASMHTTP_BODY_CACHE_BYTES` | - | 快取大小（bytes，優先於 MB） |
| `ASMHTTP_STATIC_MAX_AGE` | `600` | 靜態檔案 Cache-Control max-age |
| `ASMHTTPJIT_RUN_BENCH` | `1` | 執行基準測試 |
| `ASMHTTPJIT_BENCH_HEAVY` | `0` | 大規模基準測試 |
| `ASMHTTPJIT_BENCH_ISOLATE` | `1` | 使用獨立子程序 |
| `ASMHTTPJIT_BENCH_WORKERS` | `max(32, clients*2)` | 基準測試 Worker 數 |
| `ASMHTTPJIT_BENCH_CLIENT_TIMEOUT` | `30` | 基準測試客戶端逾時 |
| `ASMHTTPJIT_START_SERVER` | `1` | 測試後啟動伺服器 |

---

## 13. 完整應用範例

### RESTful API 伺服器

```python
#!/usr/bin/env python3
"""
完整的 RESTful API 範例
"""

from ASMhttpServerJIT import ASMHTTPServer, HttpRequest, HttpResponse, HTTPError
import json
import uuid
from datetime import datetime
from typing import Dict, Any

# 模擬資料庫
_database: Dict[str, Dict[str, Any]] = {}

# 建立伺服器
server = ASMHTTPServer(
    host="0.0.0.0",
    port=8080,
    enable_https=False,
    enable_static=False,
    worker_count=32,
    keep_alive_timeout=30.0
)

# ==================== 中間件模擬 ====================

def cors_headers(response: HttpResponse) -> HttpResponse:
    """新增 CORS headers"""
    response.set_header("Access-Control-Allow-Origin", "*")
    response.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    response.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
    return response

def require_auth(req: HttpRequest) -> str:
    """驗證 Authorization header，回傳 user_id"""
    auth = req.header("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPError(401, "Unauthorized", headers={
            "WWW-Authenticate": 'Bearer realm="api"'
        })
    # 簡化：token 就是 user_id
    return auth[7:]

# ==================== 路由 ====================

@server.route("/", methods="GET")
def index(req: HttpRequest) -> dict:
    return {
        "service": "MyAPI",
        "version": "1.0.0",
        "endpoints": ["/api/items", "/health"]
    }

@server.route("/api/items", methods=["GET", "POST", "OPTIONS"])
def items_collection(req: HttpRequest):
    # CORS preflight
    if req.method == "OPTIONS":
        return cors_headers(HttpResponse(status=204))
    
    if req.method == "GET":
        # 列出所有項目
        items = list(_database.values())
        response = HttpResponse(
            status=200,
            body=json.dumps({"items": items, "total": len(items)}).encode()
        )
        response.set_header("Content-Type", "application/json")
        return cors_headers(response)
    
    elif req.method == "POST":
        # 建立新項目
        user_id = require_auth(req)
        
        try:
            data = req.json()
        except ValueError:
            raise HTTPError(400, "Invalid JSON")
        
        if "name" not in data:
            raise HTTPError(400, "Missing required field: name")
        
        item_id = str(uuid.uuid4())
        item = {
            "id": item_id,
            "name": data["name"],
            "description": data.get("description", ""),
            "created_by": user_id,
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        _database[item_id] = item
        
        response = HttpResponse(
            status=201,
            body=json.dumps(item).encode()
        )
        response.set_header("Content-Type", "application/json")
        response.set_header("Location", f"/api/items/{item_id}")
        return cors_headers(response)

@server.route("/api/items/", methods=["GET", "PUT", "DELETE", "OPTIONS"], exact=False)
def items_resource(req: HttpRequest):
    # CORS preflight
    if req.method == "OPTIONS":
        return cors_headers(HttpResponse(status=204))
    
    # 解析 item_id
    path_parts = req.path.rstrip("/").split("/")
    if len(path_parts) < 4:
        raise HTTPError(404, "Not Found")
    
    item_id = path_parts[3]
    
    if req.method == "GET":
        # 取得單一項目
        item = _database.get(item_id)
        if not item:
            raise HTTPError(404, f"Item {item_id} not found")
        
        response = HttpResponse(
            status=200,
            body=json.dumps(item).encode()
        )
        response.set_header("Content-Type", "application/json")
        return cors_headers(response)
    
    elif req.method == "PUT":
        # 更新項目
        user_id = require_auth(req)
        
        item = _database.get(item_id)
        if not item:
            raise HTTPError(404, f"Item {item_id} not found")
        
        try:
            data = req.json()
        except ValueError:
            raise HTTPError(400, "Invalid JSON")
        
        # 更新欄位
        if "name" in data:
            item["name"] = data["name"]
        if "description" in data:
            item["description"] = data["description"]
        item["updated_by"] = user_id
        item["updated_at"] = datetime.utcnow().isoformat() + "Z"
        
        _database[item_id] = item
        
        response = HttpResponse(
            status=200,
            body=json.dumps(item).encode()
        )
        response.set_header("Content-Type", "application/json")
        return cors_headers(response)
    
    elif req.method == "DELETE":
        # 刪除項目
        require_auth(req)
        
        if item_id not in _database:
            raise HTTPError(404, f"Item {item_id} not found")
        
        del _database[item_id]
        
        return cors_headers(HttpResponse(status=204))

# ==================== 啟動 ====================

if __name__ == "__main__":
    print(f"Starting server on http://0.0.0.0:8080")
    print("Endpoints:")
    print("  GET    /              - API info")
    print("  GET    /api/items     - List items")
    print("  POST   /api/items     - Create item (requires auth)")
    print("  GET    /api/items/:id - Get item")
    print("  PUT    /api/items/:id - Update item (requires auth)")
    print("  DELETE /api/items/:id - Delete item (requires auth)")
    print()
    print("Example:")
    print('  curl -X POST http://localhost:8080/api/items \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -H "Authorization: Bearer user123" \\')
    print('    -d \'{"name": "Test Item"}\'')
    
    server.serve_forever()
```

### 靜態網站 + API 混合

```python
#!/usr/bin/env python3
"""
前後端分離架構範例：靜態檔案 + API
"""

from ASMhttpServerJIT import ASMHTTPServer, HttpRequest, HttpResponse
from pathlib import Path
import json
import os

# 確保靜態目錄存在
static_dir = Path("./frontend/dist")
static_dir.mkdir(parents=True, exist_ok=True)

# 建立範例 index.html
if not (static_dir / "index.html").exists():
    (static_dir / "index.html").write_text("""
<!DOCTYPE html>
<html>
<head>
    <title>My App</title>
    <style>
        body { font-family: sans-serif; margin: 40px; }
        #result { padding: 20px; background: #f0f0f0; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>My App</h1>
    <button onclick="fetchData()">Fetch Data</button>
    <div id="result"></div>
    <script>
        async function fetchData() {
            const res = await fetch('/api/data');
            const data = await res.json();
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        }
    </script>
</body>
</html>
""", encoding="utf-8")

server = ASMHTTPServer(
    host="0.0.0.0",
    port=3000,
    static_root=static_dir,
    docs_prefix="/",  # 根路徑提供靜態檔案
    enable_static=True,
    enable_builtin_routes=True
)

# API 路由
@server.route("/api/data", methods="GET")
def api_data(req: HttpRequest):
    return {
        "message": "Hello from API",
        "timestamp": __import__("time").time()
    }

@server.route("/api/echo", methods="POST")
def api_echo(req: HttpRequest):
    return {
        "received": req.json()
    }

if __name__ == "__main__":
    print(f"Server running at http://localhost:3000")
    server.serve_forever()
```

---

## 14. API 參考

### 函式

```python
def get_module_info() -> Dict[str, Any]:
    """
    回傳模組資訊，包括版本、Python 環境、後端可用性等。
    """

def testAll() -> bool:
    """
    執行完整測試套件。
    回傳 True 表示所有測試通過。
    """

def reason_phrase(status: int) -> str:
    """
    回傳 HTTP 狀態碼對應的原因短語。
    例如：reason_phrase(404) -> "Not Found"
    """
```

### 類別

```python
class ASMHTTPServer:
    """高效能 HTTP/HTTPS 伺服器"""
    
    def __init__(self, *, host, port, ...): ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def serve_forever(self) -> None: ...
    def route(self, path, methods="GET", *, exact=True) -> Callable: ...
    def add_route(self, path, methods, handler, *, exact=True) -> None: ...
    def mount_static(self, url_prefix, directory) -> None: ...

class HttpRequest:
    """HTTP 請求物件"""
    
    method: str
    path: str
    query_string: str
    headers: Dict[str, str]
    body: bytes
    client_addr: Optional[Tuple[str, int]]
    
    def header(self, name, default=None) -> Optional[str]: ...
    def json(self) -> Any: ...
    def text(self, encoding="utf-8") -> str: ...
    def form(self, encoding="utf-8") -> Dict[str, List[str]]: ...

class HttpResponse:
    """HTTP 回應物件"""
    
    status: int
    headers: Dict[str, str]
    body: Union[bytes, str, Iterable, None]
    
    def set_header(self, name, value) -> None: ...
    def get_header(self, name, default=None) -> Optional[str]: ...

class HttpRouter:
    """HTTP 路由器"""
    
    def add_route(self, path, methods, handler, *, exact=True) -> None: ...
    def resolve(self, path, method) -> HttpRoute: ...
    def list_routes(self) -> List[HttpRoute]: ...

class HTTPError(Exception):
    """HTTP 錯誤例外"""
    
    status: int
    message: str
    headers: Dict[str, str]
    
    def __init__(self, status, message=None, headers=None): ...

class HTTPJITPerformanceBenchmark:
    """效能基準測試"""
    
    @staticmethod
    def run_all(*, heavy=False) -> None: ...
    
    @staticmethod
    def benchmark_static_throughput(host, port, *, heavy=False) -> None: ...
    
    @staticmethod
    def benchmark_dynamic_json_close(host, port, *, heavy=False) -> None: ...
    
    @staticmethod
    def benchmark_dynamic_json_keep_alive(host, port, *, heavy=False) -> None: ...
    
    @staticmethod
    def benchmark_concurrent_clients_close(host, port, *, heavy=False) -> None: ...
    
    @staticmethod
    def benchmark_concurrent_clients_keep_alive(host, port, *, heavy=False) -> None: ...
```

### 常數

```python
__version__: str  # 模組版本 "2.1.1-HTTPJIT-BASE4.1"

HTTP_STATUS_TEXT: Dict[int, str]  # 狀態碼對應表
# {200: "OK", 404: "Not Found", ...}
```

---

## 附錄：錯誤處理對照表

| HTTP 狀態碼 | 觸發情況 |
|-------------|----------|
| 400 | 無效的請求行、malformed header、無效的 percent-encoding、無效的 UTF-8 |
| 401 | 應用程式拋出 `HTTPError(401)` |
| 403 | 靜態檔案路徑遍歷嘗試 |
| 404 | 找不到路由 |
| 405 | 路由存在但方法不允許 |
| 408 | 請求讀取逾時 |
| 413 | 請求 body 超過 `max_request_body` |
| 414 | URL 超過 `max_target_bytes` |
| 431 | Header 超過 `max_header_bytes` |
| 500 | 未捕捉的應用程式例外 |
| 501 | 請求使用 Transfer-Encoding: chunked（不支援） |
| 505 | HTTP 版本不是 1.0 或 1.1 |