
###server.py:

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import gradio as gr
import json
import asyncio
from typing import Dict, Any, Callable, Optional
from collections import deque
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SSEManager:
    """Manages Server-Sent Events (SSE) clients and heartbeats."""
    def __init__(self, heartbeat_interval: int = 30):
        self.clients: deque[asyncio.Queue] = deque()
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_task: Optional[asyncio.Task] = None

    async def add_client(self, q: asyncio.Queue) -> None:
        """Add a new SSE client and start heartbeat if needed."""
        self.clients.append(q)
        if self.heartbeat_task is None:
            self.heartbeat_task = asyncio.create_task(self._send_heartbeats())
        logger.info("New SSE client connected. Total clients: %d", len(self.clients))

    async def remove_client(self, q: asyncio.Queue) -> None:
        """Remove an SSE client and stop heartbeat if no clients remain."""
        if q in self.clients:
            self.clients.remove(q)
        if not self.clients and self.heartbeat_task:
            self.heartbeat_task.cancel()
            self.heartbeat_task = None
        logger.info("SSE client disconnected. Total clients: %d", len(self.clients))

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all connected SSE clients."""
        for q in list(self.clients):
            await q.put(message)

    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeat messages to keep connections alive."""
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                await self.broadcast({"type": "heartbeat", "timestamp": asyncio.get_event_loop().time()})
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled.")

class JSONRPCManager:
    """Manages JSON-RPC methods and request handling."""
    def __init__(self):
        self.methods: Dict[str, Callable] = {}

    def register_method(self, name: str, func: Callable) -> None:
        """Register a custom JSON-RPC method."""
        self.methods[name] = func
        logger.info("Registered JSON-RPC method: %s", name)

    async def handle_request(self, request: Dict[str, Any], gradio_fn: Callable) -> Dict[str, Any]:
        """Handle a JSON-RPC request."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method not in self.methods:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method '{method}' not found"}
            }

        try:
            if method == "gradio_predict":
                arguments = params.get("arguments", {})
                result = await asyncio.get_event_loop().run_in_executor(None, gradio_fn, *arguments.values())
            else:
                result = await self.methods[method](params)
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as e:
            logger.error("Error processing JSON-RPC request: %s", str(e))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(e)}
            }

class myGrMCPfastapi:
    """A FastAPI application integrating Gradio with SSE and JSON-RPC support."""
    def __init__(
        self,
        gradio_app: gr.Interface,
        mcp_path: str = "/mcp",
        gradio_mount_path: str = "/",
        sse_heartbeat_interval: int = 30
    ):
        """
        Initialize the application.

        Args:
            gradio_app: The Gradio interface to integrate.
            mcp_path: The endpoint path for SSE and JSON-RPC.
            gradio_mount_path: The mount path for the Gradio UI.
            sse_heartbeat_interval: Interval (seconds) for SSE heartbeats.
        """
        self.gradio_app = gradio_app
        self.mcp_path = mcp_path
        self.app = FastAPI(title="Gradio MCP Server")
        self.sse_manager = SSEManager(heartbeat_interval=sse_heartbeat_interval)
        self.jsonrpc_manager = JSONRPCManager()

        # Register default Gradio predict method
        self.jsonrpc_manager.register_method("gradio_predict", self._gradio_predict)

        # Setup routes and mount applications
        self._setup_routes()
        self.app.mount("/static", StaticFiles(directory="static", html=True), name="static")
        self.app = gr.mount_gradio_app(self.app, self.gradio_app, path=gradio_mount_path)

    def _setup_routes(self) -> None:
        """Configure the MCP endpoint for SSE (GET) and JSON-RPC (POST)."""
        async def mcp_endpoint(request: Request):
            if request.method == "GET":
                q: asyncio.Queue = asyncio.Queue()
                await self.sse_manager.add_client(q)
                async def event_stream():
                    try:
                        while True:
                            event = await q.get()
                            yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.CancelledError:
                        pass
                    finally:
                        await self.sse_manager.remove_client(q)
                return StreamingResponse(event_stream(), media_type="text/event-stream")
            
            # Handle POST (JSON-RPC)
            try:
                body = await request.json()
            except json.JSONDecodeError:
                return JSONResponse(
                    {"error": "Invalid JSON"}, status_code=400
                )
            resp = await self.jsonrpc_manager.handle_request(body, self.gradio_app.fn)
            await self.sse_manager.broadcast(resp)
            return JSONResponse(resp)

        self.app.add_api_route(
            self.mcp_path,
            mcp_endpoint,
            methods=["GET", "POST"],
            name="mcp_endpoint"
        )

    def register_jsonrpc_method(self, name: str, func: Callable) -> None:
        """Register a custom JSON-RPC method."""
        self.jsonrpc_manager.register_method(name, func)

    async def _gradio_predict(self, arguments: Dict[str, Any]) -> Any:
        """Default method to handle Gradio predictions."""
        inputs = [arguments.get(comp.label, "") for comp in self.gradio_app.input_components]
        return self.gradio_app.fn(*inputs)

    def run(self, host: str = "127.0.0.1", port: int = 7860) -> None:
        """Run the FastAPI application using uvicorn."""
        import uvicorn
        logger.info("Starting HTTP server at http://%s:%d", host, port)
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    import os
    os.makedirs("static", exist_ok=True)

    def hello_chatbot(user_input: str) -> str:
        return f"Hello, {user_input}!"

    gr_app = gr.Interface(
        fn=hello_chatbot,
        inputs=[gr.Textbox(label="Your Input")],
        outputs=gr.Textbox(label="Response"),
        title="Hello World Chatbot"
    )

    mcp_app = myGrMCPfastapi(
        gradio_app=gr_app,
        mcp_path="/mcp",
        gradio_mount_path="/",
        sse_heartbeat_interval=30
    )
    mcp_app.run(host="127.0.0.1", port=7860)
####test.py:
    
import asyncio
import aiohttp
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_mcp_client(mcp_url: str = "http://127.0.0.1:7860/mcp") -> None:
    """Test the MCP endpoint with SSE and JSON-RPC."""
    async with aiohttp.ClientSession() as session:
        # Test 1: Establish SSE connection and receive messages
        logger.info("Testing SSE connection...")
        async with session.get(mcp_url) as sse_resp:
            if sse_resp.status != 200:
                logger.error("Failed to establish SSE connection: %d", sse_resp.status)
                return

            # Test 2: Send a valid JSON-RPC POST request
            logger.info("Sending JSON-RPC POST request...")
            rpc_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "gradio_predict",
                "params": {"arguments": {"Your Input": "World"}}
            }
            async with session.post(mcp_url, json=rpc_request) as post_resp:
                if post_resp.status != 200:
                    logger.error("POST request failed: %d", post_resp.status)
                    return
                post_text = await post_resp.text()
                logger.info("POST response: %s", post_text)

            # Test 3: Receive result from SSE
            logger.info("Waiting for SSE message...")
            while True:
                raw = await sse_resp.content.readline()
                if not raw:
                    logger.warning("SSE connection closed unexpectedly")
                    break
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line[len("data:"):])
                if payload.get("id") == 1:
                    if "result" in payload:
                        logger.info("Received result from SSE: %s", payload["result"])
                    elif "error" in payload:
                        logger.error("Received error from SSE: %s", payload["error"])
                    break
                elif payload.get("type") == "heartbeat":
                    logger.debug("Received heartbeat: %s", payload)

async def run_tests() -> None:
    """Run all test cases."""
    mcp_url = "http://127.0.0.1:7860/mcp"
    try:
        await test_mcp_client(mcp_url)
    except Exception as e:
        logger.error("Test failed: %s", str(e))

if __name__ == "__main__":
    asyncio.run(run_tests())
    