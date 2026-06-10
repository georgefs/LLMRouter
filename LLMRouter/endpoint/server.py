"""LLM Router Endpoint Server

HTTP API for serving trained routers.
相容 semantic-router r1_server_url 協議。

Protocol:
  POST /route
    Request:  {"query": "text"}
    Response: {"selected_model": "gpt-4"}

  GET /health
    Response: {"status": "ok", "router_type": "KNN"}

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
                    "status": "ok",
                    "router_type": self.router.__class__.__name__,
                }
                self._send_json(200, response)
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
    ):
        self.router_path = Path(router_path)
        self.port = port
        self.host = host
        self.server: Optional[HTTPServer] = None
        self.router: Optional[BaseRouter] = None

        self._load_router()

    def _load_router(self):
        """Load router from file."""
        if not self.router_path.exists():
            raise FileNotFoundError(f"Router file not found: {self.router_path}")

        logger.info(f"Loading router from {self.router_path}")

        # 自動推斷 router 類型並加載
        router_class = self._infer_router_type()
        self.router = router_class.load(self.router_path)

        if self.router.model_names is None:
            raise ValueError("Router model_names not bound")

        logger.info(
            f"Router loaded successfully. Models: {self.router.model_names}"
        )

    def _infer_router_type(self) -> type:
        """Infer router type from pickled object."""
        import pickle

        with open(self.router_path, "rb") as f:
            checkpoint = pickle.load(f)

        # 從 checkpoint 推斷 router 類型
        if isinstance(checkpoint, dict):
            # 新式 save/load 格式
            from ..router import (
                KNNRouter,
                OracleRouter,
                RandomRouter,
                MFRouter,
                SWRankingRouter,
                RoBERTaMLCRouter,
                GRPORouter,
            )

            # 檢查特徵來判斷 router 類型
            if "_nn" in checkpoint:
                return KNNRouter
            elif "seed" in checkpoint and "_n_models" in checkpoint:
                return RandomRouter
            elif "policy_state" in checkpoint:
                return GRPORouter
            elif "_X_train" in checkpoint and "_Y_train" in checkpoint:
                if "model_name" in checkpoint:
                    return RoBERTaMLCRouter
                else:
                    return SWRankingRouter
            elif "weights" in checkpoint:
                return MFRouter
            else:
                # 預設為 OracleRouter
                return OracleRouter
        else:
            raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    def start(self):
        """Start the HTTP server."""
        RouterHandler.router = self.router
        self.server = HTTPServer((self.host, self.port), RouterHandler)

        logger.info(f"Router endpoint server listening on {self.host}:{self.port}")
        logger.info(f"Available models: {self.router.model_names}")

        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down")
            self.shutdown()

    def shutdown(self):
        """Shutdown the server."""
        if self.server:
            self.server.shutdown()
            logger.info("Server shut down")
