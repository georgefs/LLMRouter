"""
LLMRouter 測試共用 fixtures
============================

新增 router / annotator 測試時直接使用下列 fixtures，
不需在每個測試檔案重複定義 setUp 邏輯。

可用 fixtures
-------------
router_data      RouterData  50 筆、3 模型、384-dim 預計算嵌入（session 範圍）
trained_knn      已訓練的 KNNRouter（function 範圍）
saved_router_path  .pkl 暫存路徑（function 範圍，teardown 自動清除）
live_endpoint    執行中的 LLMRouterEndpointServer，提供 .base_url（function 範圍）

典型用法
--------
    def test_my_router(router_data, trained_knn):
        preds = trained_knn.predict(router_data.test_prompt)
        assert all(m in trained_knn.model_names for m in preds)

    def test_endpoint(live_endpoint):
        import httpx
        r = httpx.get(f"{live_endpoint.base_url}/health")
        assert r.json()["status"] == "healthy"
"""
from __future__ import annotations

import socket
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest

# ── 常數 ─────────────────────────────────────────────────────────────────────

MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
EMB_DIM = 384   # sentence-transformers/all-MiniLM-L6-v2 輸出維度
N_SAMPLES = 50


# ── RouterData ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def router_data():
    """
    標準測試用 RouterData（session 範圍，全套測試共用一份）。

    規格：
      - 50 筆：train 40 / val 5 / test 5
      - 3 個模型：gpt-4, gpt-3.5-turbo, claude-3
      - 384-dim 預計算嵌入（固定 seed=42，適合 snapshot 斷言）

    注意：session 範圍物件為共用，測試不應修改其欄位。
    若需要可變副本，請呼叫 dataclasses.replace(router_data, ...)。
    """
    from LLMRouter.router import RouterData

    rng = np.random.default_rng(42)
    prompts = [f"query_{i}" for i in range(N_SAMPLES)]
    embeddings = rng.standard_normal((N_SAMPLES, EMB_DIM)).astype(np.float32)
    scores = rng.random((N_SAMPLES, len(MODELS))).astype(np.float32)

    return RouterData(
        train_prompt=prompts[:40],
        val_prompt=prompts[40:45],
        test_prompt=prompts[45:],
        train_score=scores[:40],
        val_score=scores[40:45],
        test_score=scores[45:],
        test_tokens=np.zeros((5, len(MODELS))),
        test_time=np.zeros((5, len(MODELS))),
        models=MODELS[:],
        train_embed=embeddings[:40],
        val_embed=embeddings[40:45],
        test_embed=embeddings[45:],
    )


# ── Router ────────────────────────────────────────────────────────────────────


@pytest.fixture
def trained_knn(router_data):
    """
    已訓練的 KNNRouter（function 範圍，每個測試取得獨立實例）。

    使用 router_data 的預計算嵌入訓練，無需下載嵌入模型。
    k=5，模型：gpt-4 / gpt-3.5-turbo / claude-3。
    """
    from LLMRouter.router import KNNRouter

    r = KNNRouter(k=5)
    r.fit(router_data)
    return r


@pytest.fixture
def saved_router_path(trained_knn):
    """
    已儲存至暫存目錄的 KNNRouter .pkl 路徑。

    yield 後 tempdir 自動清除，不需手動 teardown。

    典型用法：
        def test_load(saved_router_path):
            from LLMRouter.router import KNNRouter
            r = KNNRouter.load(saved_router_path)
            assert r.model_names == MODELS
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "knn.pkl"
        trained_knn.save(path)
        yield path


# ── Endpoint Server ───────────────────────────────────────────────────────────


@pytest.fixture
def live_endpoint(saved_router_path):
    """
    在隨機 port 啟動 LLMRouterEndpointServer，測試結束後自動 shutdown。

    Attributes：
        base_url  e.g. "http://localhost:54321"
        config    空 dict（不含 embedding_model，跳過相關斷言）

    典型用法：
        def test_health(live_endpoint):
            import httpx
            r = httpx.get(f"{live_endpoint.base_url}/health")
            assert r.status_code == 200

    注意：若需要測試 embedding_model 欄位，
    請自行帶入 LLMRouterEndpointServer(path, embedding_model="xxx")。
    """
    from LLMRouter.endpoint import LLMRouterEndpointServer

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    srv = LLMRouterEndpointServer(saved_router_path, port=port)
    t = threading.Thread(target=srv.start, daemon=True)
    t.start()
    time.sleep(0.3)

    class _Info:
        base_url = f"http://localhost:{port}"
        config: dict = {}

    yield _Info()
    srv.shutdown()
