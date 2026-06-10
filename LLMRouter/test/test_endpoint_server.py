"""Tests for LLMRouter HTTP endpoint server.

驗證：
1. 伺服器可以加載保存的 router
2. POST /route 端點返回正確的模型名稱
3. GET /health 和 GET /models 工作正常
4. 錯誤處理正常
"""

import json
import pytest
import numpy as np
import threading
import time
from pathlib import Path
import tempfile
import urllib.request
import urllib.error

from LLMRouter.router import RouterData, KNNRouter, OracleRouter, RandomRouter
from LLMRouter.endpoint import LLMRouterEndpointServer


@pytest.fixture
def sample_data():
    """建立簡單的 RouterData"""
    n_samples = 50
    n_models = 3
    embedding_dim = 384

    prompts = [f"query_{i}" for i in range(n_samples)]
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    scores = np.random.rand(n_samples, n_models).astype(np.float32)

    data = RouterData(
        train_prompt=prompts[:40],
        val_prompt=prompts[40:45],
        test_prompt=prompts[45:],
        train_score=scores[:40],
        val_score=scores[40:45],
        test_score=scores[45:],
        test_tokens=np.zeros((5, n_models)),
        test_time=np.zeros((5, n_models)),
        models=["gpt-4", "gpt-3.5-turbo", "claude-3"],
        train_embed=embeddings[:40],
        val_embed=embeddings[40:45],
        test_embed=embeddings[45:],
    )
    return data


@pytest.fixture
def saved_router(sample_data):
    """訓練並保存一個 router"""
    router = KNNRouter(k=5)
    router.fit(sample_data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "router.pkl"
        router.save(path)
        yield path


def test_endpoint_loads_router(saved_router):
    """伺服器可以加載保存的 router"""
    server = LLMRouterEndpointServer(saved_router, port=9999)

    assert server.router is not None
    assert server.router.model_names == ["gpt-4", "gpt-3.5-turbo", "claude-3"]


def test_endpoint_infers_router_type(sample_data):
    """自動推斷 router 類型"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 測試 KNNRouter
        knn = KNNRouter(k=5)
        knn.fit(sample_data)
        knn_path = Path(tmpdir) / "knn.pkl"
        knn.save(knn_path)

        server = LLMRouterEndpointServer(knn_path, port=9999)
        assert server.router.__class__.__name__ == "KNNRouter"

        # 測試 RandomRouter
        rand = RandomRouter(seed=42)
        rand.fit(sample_data)
        rand_path = Path(tmpdir) / "random.pkl"
        rand.save(rand_path)

        server = LLMRouterEndpointServer(rand_path, port=9999)
        assert server.router.__class__.__name__ == "RandomRouter"

        # 測試 OracleRouter
        oracle = OracleRouter()
        oracle.fit(sample_data)
        oracle_path = Path(tmpdir) / "oracle.pkl"
        oracle.save(oracle_path)

        server = LLMRouterEndpointServer(oracle_path, port=9999)
        assert server.router.__class__.__name__ == "OracleRouter"


def test_endpoint_health_check(saved_router):
    """GET /health 端點工作"""
    port = 9998
    server = LLMRouterEndpointServer(saved_router, port=port)

    # 在後臺執行伺服器
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)  # 等待伺服器啟動

    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/health")
        data = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert data["status"] == "ok"
        assert data["router_type"] == "KNNRouter"
    finally:
        server.shutdown()


def test_endpoint_models_list(saved_router):
    """GET /models 端點返回模型列表"""
    port = 9997
    server = LLMRouterEndpointServer(saved_router, port=port)

    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)

    try:
        response = urllib.request.urlopen(f"http://localhost:{port}/models")
        data = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert len(data["models"]) == 3
        model_names = [m["name"] for m in data["models"]]
        assert "gpt-4" in model_names
        assert "gpt-3.5-turbo" in model_names
        assert "claude-3" in model_names
    finally:
        server.shutdown()


def test_endpoint_route_request(saved_router):
    """POST /route 端點返回正確的模型名稱"""
    port = 9996
    server = LLMRouterEndpointServer(saved_router, port=port)

    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)

    try:
        # 測試路由請求
        request_data = json.dumps({"query": "What is machine learning?"}).encode(
            "utf-8"
        )
        req = urllib.request.Request(
            f"http://localhost:{port}/route",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert "selected_model" in data
        assert data["selected_model"] in [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3",
        ]
    finally:
        server.shutdown()


def test_endpoint_route_missing_query(saved_router):
    """POST /route 沒有 query 應返回 400"""
    port = 9995
    server = LLMRouterEndpointServer(saved_router, port=port)

    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)

    try:
        request_data = json.dumps({}).encode("utf-8")
        req = urllib.request.Request(
            f"http://localhost:{port}/route",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised HTTP 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
            data = json.loads(e.read().decode("utf-8"))
            assert "error" in data
    finally:
        server.shutdown()


def test_endpoint_invalid_json(saved_router):
    """POST /route 無效 JSON 應返回 400"""
    port = 9994
    server = LLMRouterEndpointServer(saved_router, port=port)

    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)

    try:
        request_data = b"not valid json"
        req = urllib.request.Request(
            f"http://localhost:{port}/route",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            urllib.request.urlopen(req)
            assert False, "Should have raised HTTP 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.shutdown()


def test_endpoint_not_found(saved_router):
    """不存在的路徑應返回 404"""
    port = 9993
    server = LLMRouterEndpointServer(saved_router, port=port)

    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(0.5)

    try:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/nonexistent")
            assert False, "Should have raised HTTP 404"
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.shutdown()


def test_endpoint_router_not_found():
    """Router 文件不存在應拋出 FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        LLMRouterEndpointServer("/nonexistent/router.pkl", port=9999)
