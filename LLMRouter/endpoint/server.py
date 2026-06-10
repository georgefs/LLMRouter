"""LLM Router Endpoint Server

HTTP API for serving trained routers.
相容 semantic-router r1_server_url 協議。

Protocol:
  POST /route
    Request:  {"query": "text"}
    Response: {"selected_model": "gpt-4"}

  GET /health
    Response: {"status": "healthy", "router_type": "KNNRouter", "model_count": N}

  GET /models
    Response: {"models": [{"name": "gpt-4"}, ...]}
"""

import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

from ..router.base import BaseRouter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RouterHandler(BaseHTTPRequestHandler):
    """HTTP request handler for router endpoint."""

    router: Optional[BaseRouter] = None
    embedding_model: Optional[str] = None

    def do_POST(self):
        """Handle POST requests."""
        if self.path == "/route":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                data = json.loads(body)
                query = data.get("query", "")

                if not query:
                    self._send_error(400, "Missing 'query' field")
                    return

                if self.router is None:
                    self._send_error(500, "Router not initialized")
                    return

                # 路由 query
                selected_model = self.router.predict([query])[0]

                response = {"selected_model": selected_model}
                self._send_json(200, response)

            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
            except Exception as e:
                logger.exception("Error processing request")
                self._send_error(500, str(e))
        else:
            self._send_error(404, "Not found")

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            if self.router is None:
                self._send_json(503, {"status": "unavailable", "reason": "Router not initialized"})
            else:
                response = {
                    "status": "healthy",
                    "router_type": self.router.__class__.__name__,
                    "model_count": len(self.router.model_names or []),
                }
                if self.embedding_model:
                    response["embedding_model"] = self.embedding_model
                self._send_json(200, response)
        elif self.path == "/route":
            self._send_json(405, {"error": "Method Not Allowed"})
        elif self.path == "/models":
            if self.router is None or self.router.model_names is None:
                self._send_json(503, {"error": "Router not initialized"})
            else:
                models = [{"name": name} for name in self.router.model_names]
                response = {"models": models}
                self._send_json(200, response)
        else:
            self._send_error(404, "Not found")

    def _send_json(self, status: int, data: dict):
        """Send JSON response."""
        response = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(response))
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, status: int, message: str):
        """Send error response."""
        self._send_json(status, {"error": message})

    def log_message(self, format, *args):
        """Log HTTP requests."""
        logger.info(f"{self.client_address[0]} - {format % args}")


class LLMRouterEndpointServer:
    """HTTP server for serving trained LLM router.

    Args:
        router_path: Path to saved router (.pkl file)
        port: Server port (default 8888)
        host: Server host (default "0.0.0.0")
    """

    def __init__(
        self,
        router_path: str | Path,
        port: int = 8888,
        host: str = "0.0.0.0",
        embedding_model: Optional[str] = None,
    ):
        self.router_path = Path(router_path)
        self.port = port
        self.host = host
        self.embedding_model = embedding_model
        self.server: Optional[HTTPServer] = None
        self.router: Optional[BaseRouter] = None

        self._load_router()

    def _load_router(self):
        """Load router from file (single pickle.load)."""
        if not self.router_path.exists():
            raise FileNotFoundError(f"Router file not found: {self.router_path}")

        logger.info(f"Loading router from {self.router_path}")

        if self.router_path.is_dir():
            cls = self._detect_type_dir(self.router_path)
            self.router = cls.load(self.router_path)
        else:
            import pickle
            with open(self.router_path, "rb") as f:
                ck = pickle.load(f)
            cls = self._detect_type_checkpoint(ck)
            self.router = cls.load(ck)

        if self.router.model_names is None:
            raise ValueError("Router model_names not bound")

        logger.info(f"Router loaded: {self.router.__class__.__name__}, models={self.router.model_names}")

    def _detect_type_checkpoint(self, ck: dict) -> type:
        """Determine router class from a loaded checkpoint dict."""
        from ..router.registry import get as get_router

        name = ck.get("router_type")
        if name:
            try:
                cls, _ = get_router(name)
                return cls
            except KeyError:
                raise ValueError(f"未知的 router_type：{name!r}")

        # 向後相容：舊檔案無 router_type 欄位
        logger.warning("舊格式 checkpoint（無 router_type），使用特徵偵測")
        from ..router import KNNRouter, OracleRouter, RandomRouter, MFRouter, SWRankingRouter, GRPORouter
        if "_nn" in ck:
            return KNNRouter
        if "seed" in ck and "_n_models" in ck:
            return RandomRouter
        if "policy_state" in ck:
            return GRPORouter
        if "_X_train" in ck and "_Y_train" in ck:
            return SWRankingRouter
        if "model_state" in ck:
            return MFRouter
        return OracleRouter

    def _detect_type_dir(self, path: Path) -> type:
        """Determine router class from a directory (HuggingFace format)."""
        import json
        from ..router.registry import get as get_router

        config_path = path / "router_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            name = config.get("router_type")
            if name:
                cls, _ = get_router(name)
                return cls

        from ..router import RoBERTaMLCRouter
        return RoBERTaMLCRouter

    def start(self):
        """Start the HTTP server."""
        RouterHandler.router = self.router
        RouterHandler.embedding_model = self.embedding_model
        self.server = HTTPServer((self.host, self.port), RouterHandler)

        logger.info(f"Router endpoint server listening on {self.host}:{self.port}")
        logger.info(f"Available models: {self.router.model_names}")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down")
            self.shutdown()

    def shutdown(self):
        """Shutdown the server and close the socket."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("Server shut down")
