"""
SemanticAPIRouter 測試
======================

分兩層：
  unit     — mock HTTP server，不需外部服務
  integration — 需要 localhost:8080 的 semantic-router 實際運行
                若服務不在線，整個 class 自動 skip
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import socket
import tempfile

import numpy as np
import pytest

from LLMRouter.router import RouterData, SemanticAPIRouter


# ── 共用 fixture ──────────────────────────────────────────────────────────────

MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3"]


@pytest.fixture
def sample_data():
    n, m = 20, 3
    rng = np.random.default_rng(0)
    prompts = [f"q{i}" for i in range(n)]
    return RouterData(
        train_prompt=prompts[:15],
        val_prompt=prompts[15:18],
        test_prompt=prompts[18:],
        train_score=rng.random((15, m)).astype("f4"),
        val_score=rng.random((3, m)).astype("f4"),
        test_score=rng.random((2, m)).astype("f4"),
        test_tokens=np.zeros((2, m)),
        test_time=np.zeros((2, m)),
        models=MODELS[:],
    )


# ── Mock HTTP server ──────────────────────────────────────────────────────────

def _make_mock_server(recommended_model: str = "gpt-4"):
    """在隨機 port 啟動一個只回 recommended_model 的 mock API。"""

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            body = json.dumps({
                "classification": {"category": "test", "confidence": 0.9},
                "recommended_model": recommended_model,
                "routing_decision": "test",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}"


# ── Unit tests（mock server）─────────────────────────────────────────────────

class TestSemanticAPIRouterUnit:

    def test_fit_succeeds_when_api_reachable(self, sample_data):
        srv, url = _make_mock_server("gpt-4")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            assert r.model_names == MODELS
        finally:
            srv.shutdown(); srv.server_close()

    def test_predict_probs_shape(self, sample_data):
        srv, url = _make_mock_server("gpt-4")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            probs = r.predict_probs(sample_data.test_prompt)
            assert probs.shape == (len(sample_data.test_prompt), len(MODELS))
        finally:
            srv.shutdown(); srv.server_close()

    def test_recommended_model_becomes_argmax(self, sample_data):
        """mock 始終回 gpt-4，argmax 應指向 gpt-4 的 index。"""
        srv, url = _make_mock_server("gpt-4")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            probs = r.predict_probs(sample_data.test_prompt)
            idx = np.argmax(probs, axis=1)
            gpt4_idx = MODELS.index("gpt-4")
            assert all(i == gpt4_idx for i in idx)
        finally:
            srv.shutdown(); srv.server_close()

    def test_unknown_model_fallback_to_uniform(self, sample_data):
        """API 回傳不在 model_names 裡的名稱，應 fallback 為均勻分布。"""
        srv, url = _make_mock_server("unknown-model-xyz")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            probs = r.predict_probs(["one prompt"])
            expected = 1.0 / len(MODELS)
            assert np.allclose(probs[0], expected)
        finally:
            srv.shutdown(); srv.server_close()

    def test_fit_raises_when_unreachable(self, sample_data):
        r = SemanticAPIRouter(base_url="http://127.0.0.1:1", timeout=1.0)
        with pytest.raises(RuntimeError, match="無法連線"):
            r.fit(sample_data)

    def test_save_load_roundtrip(self, sample_data):
        srv, url = _make_mock_server("gpt-4")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "sr.pkl"
                r.save(path)

                import pickle
                with open(path, "rb") as f:
                    ck = pickle.load(f)
                assert ck["router_type"] == "semantic_api"
                assert ck["base_url"] == url

                loaded = SemanticAPIRouter.load(path)
                assert loaded.model_names == MODELS
                assert loaded.base_url == url
        finally:
            srv.shutdown(); srv.server_close()

    def test_load_from_dict(self, sample_data):
        srv, url = _make_mock_server("gpt-3.5-turbo")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / "sr.pkl"
                r.save(path)
                import pickle
                with open(path, "rb") as f:
                    ck = pickle.load(f)
                loaded = SemanticAPIRouter.load(ck)
                assert loaded.model_names == MODELS
        finally:
            srv.shutdown(); srv.server_close()

    def test_registered_in_registry(self):
        from LLMRouter.router.registry import list_routers
        assert "semantic_api" in list_routers()

    def test_evaluate_returns_metrics(self, sample_data):
        srv, url = _make_mock_server("gpt-4")
        try:
            r = SemanticAPIRouter(base_url=url)
            r.fit(sample_data)
            metrics = r.evaluate(sample_data)
            for key in ("mu", "vb", "ep", "avg_tokens", "avg_latency"):
                assert key in metrics
        finally:
            srv.shutdown(); srv.server_close()


# ── Integration tests（需要 localhost:8080）───────────────────────────────────

def _check_live():
    """回傳 True 表示 localhost:8080 的 semantic-router 可連線。"""
    try:
        req = urllib.request.Request(
            "http://localhost:8080/api/v1/classify/intent",
            data=b'{"text":"ping"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False


import urllib.request  # noqa: E402（需在 _check_live 定義之前 import）

live_only = pytest.mark.skipif(
    not _check_live(),
    reason="localhost:8080 semantic-router 未運行",
)


@live_only
class TestSemanticAPIRouterIntegration:

    @pytest.fixture
    def live_data(self):
        """使用 semantic-router 實際回傳的模型名稱建立測試資料。"""
        # 先問一次 API 確認可用的模型名稱
        req = urllib.request.Request(
            "http://localhost:8080/api/v1/classify/intent",
            data=b'{"text":"hello"}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            rec_model = json.loads(resp.read()).get("recommended_model", "gpt-3.5-turbo")

        # 建立含有該模型的 RouterData
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
        if rec_model not in models:
            models.append(rec_model)
        n, m = 10, len(models)
        rng = np.random.default_rng(1)
        prompts = [f"integration query {i}" for i in range(n)]
        return RouterData(
            train_prompt=prompts[:8],
            val_prompt=prompts[8:9],
            test_prompt=prompts[9:],
            train_score=rng.random((8, m)).astype("f4"),
            val_score=rng.random((1, m)).astype("f4"),
            test_score=rng.random((1, m)).astype("f4"),
            test_tokens=np.zeros((1, m)),
            test_time=np.zeros((1, m)),
            models=models,
        )

    def test_fit_and_predict_live(self, live_data):
        r = SemanticAPIRouter(base_url="http://localhost:8080")
        r.fit(live_data)
        probs = r.predict_probs(live_data.test_prompt)
        assert probs.shape == (len(live_data.test_prompt), len(live_data.models))
        assert np.all(probs >= 0)
        # 每行應有一個 1.0（one-hot）或全均勻
        row_sums = probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)

    def test_evaluate_live(self, live_data):
        r = SemanticAPIRouter(base_url="http://localhost:8080")
        r.fit(live_data)
        metrics = r.evaluate(live_data)
        assert 0.0 <= metrics["mu"] <= 1.0
