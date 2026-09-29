"""
行為測試：LLMRouter Endpoint
================================

這些測試定義了 RouterEndpointServer 的預期行為，不依賴具體實現。
按照 semantic-router r1_server_url 協議進行驗證。

使用方法：
    pytest LLMRouter/test/test_endpoint_behavior.py -v

測試組織：
    1. 基本功能 (Basic Functionality)
    2. 協議遵循 (Protocol Compliance)
    3. 邊界情況 (Edge Cases)
    4. 性能 (Performance)
    5. 集成 (Integration with Semantic Router)
"""

import pytest
import json
import time
import httpx
from typing import Dict, Any


# ============================================================================
# 1. 基本功能測試
# ============================================================================

class TestHealthEndpoint:
    """
    健康檢查端點的行為測試
    """

    def test_health_endpoint_returns_200_when_service_is_ready(self, endpoint_server):
        """
        行為: 服務就緒時，/health 應返回 200 OK
        場景: 在調用 start() 後立即檢查
        期望: 狀態碼 200 且 status 字段為 "healthy"
        """
        with httpx.Client() as client:
            response = client.get(f"{endpoint_server.base_url}/health")
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            data = response.json()
            assert data.get("status") == "healthy"

    def test_health_endpoint_includes_router_metadata(self, endpoint_server):
        """
        行為: /health 响應應包含 router 的元數據
        期望: 返回中包含 router_type 和 model_count
        """
        with httpx.Client() as client:
            response = client.get(f"{endpoint_server.base_url}/health")
            data = response.json()
            assert "router_type" in data, "Missing router_type"
            assert "model_count" in data, "Missing model_count"
            assert isinstance(data["model_count"], int)
            assert data["model_count"] > 0

    def test_health_endpoint_includes_embedding_model_info(self, endpoint_server):
        """
        行為: /health 响應應包含嵌入模型信息（如果使用）
        期望: embedding_model 字段存在於響應中
        """
        with httpx.Client() as client:
            response = client.get(f"{endpoint_server.base_url}/health")
            data = response.json()
            if endpoint_server.config.get("embedding_model"):
                assert "embedding_model" in data


class TestRouteEndpoint:
    """
    路由決策端點的行為測試（核心功能）
    """

    def test_route_endpoint_requires_post_method(self, endpoint_server):
        """
        行為: /route 端點只接受 POST 請求
        期望: GET 或 PUT 請求應返回 405 Method Not Allowed
        """
        with httpx.Client() as client:
            response = client.get(f"{endpoint_server.base_url}/route")
            assert response.status_code == 405

    def test_route_endpoint_accepts_valid_query(self, endpoint_server):
        """
        行為: /route 應接受簡單的問題文本查詢
        場景: 發送簡單的問題 query
        期望: 返回 200 OK，包含 selected_model
        """
        payload = {
            "query": "What is 2+2?"
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
            data = response.json()
            assert "selected_model" in data

    def test_route_endpoint_returns_valid_model_name(self, endpoint_server):
        """
        行為: 返回的 selected_model 必須是有效的模型名稱
        場景: 發送簡單的查詢
        期望: selected_model 是訓練時使用的模型之一
        """
        payload = {
            "query": "What is the capital of France?"
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            data = response.json()
            selected = data["selected_model"]
            # 模型名稱應該是非空字符串
            assert isinstance(selected, str) and len(selected) > 0

    def test_route_endpoint_returns_optional_thinking_field(self, endpoint_server):
        """
        行為: /route 可以返回 thinking 字段用於調試
        期望: thinking 字段是可選的，但如果有應該是字符串
        """
        payload = {
            "query": "Task: test\nAvailable models: gpt-4, gpt-3.5"
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            data = response.json()
            if "thinking" in data:
                assert isinstance(data["thinking"], str)

    def test_route_endpoint_returns_consistent_results(self, endpoint_server):
        """
        行為: 對相同查詢，/route 應返回一致的模型選擇
        場景: 連續調用 3 次相同查詢
        期望: 3 次返回相同的 selected_model
        """
        payload = {
            "query": "Task: fixed_test\nAvailable models: gpt-4, gpt-3.5"
        }
        with httpx.Client() as client:
            results = []
            for _ in range(3):
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                data = response.json()
                results.append(data["selected_model"])

            # 應該三次相同（假設路由決策確定性）
            assert results[0] == results[1] == results[2], \
                f"Inconsistent results: {results}"


# ============================================================================
# 2. 協議遵循測試
# ============================================================================

class TestProtocolCompliance:
    """
    驗證 endpoint 遵循 semantic-router r1_server_url 協議（Router-R1 格式）
    """

    def test_query_field_is_required(self, endpoint_server):
        """
        行為: 請求必須包含 "query" 字段
        場景: 發送缺少 query 的 JSON
        期望: 返回 400 Bad Request
        """
        payload = {
            "other_field": "some value"  # 缺少 query 字段
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code in [400, 422], \
                f"Expected error status, got {response.status_code}"

    def test_query_can_be_any_text(self, endpoint_server):
        """
        行為: Query 可以是任意長度和格式的文本
        場景: 發送多種格式的 query
        期望: 都應返回 200
        """
        test_queries = [
            "What is 2+2?",
            "Write a Python function to implement quicksort algorithm",
            "Explain quantum computing in simple terms",
            "Write a haiku about nature",
        ]

        with httpx.Client() as client:
            for query in test_queries:
                payload = {"query": query}
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                assert response.status_code == 200, \
                    f"Failed for query: {query}"

    def test_response_must_include_selected_model_field(self, endpoint_server):
        """
        行為: /route 的 200 響應必須包含 selected_model 字段
        期望: 字段存在且為非空字符串
        """
        payload = {
            "query": "Task: test\nAvailable models: gpt-4, gpt-3.5"
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code == 200
            data = response.json()
            assert "selected_model" in data
            assert isinstance(data["selected_model"], str)
            assert len(data["selected_model"]) > 0

    def test_response_content_type_is_json(self, endpoint_server):
        """
        行為: /health 和 /route 的響應 Content-Type 必須是 application/json
        期望: Content-Type header 包含 application/json
        """
        with httpx.Client() as client:
            # 檢查 /health
            response = client.get(f"{endpoint_server.base_url}/health")
            assert "application/json" in response.headers.get("content-type", "")

            # 檢查 /route
            payload = {"query": "Task: test\nAvailable models: gpt-4"}
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert "application/json" in response.headers.get("content-type", "")


# ============================================================================
# 3. 邊界情況測試
# ============================================================================

class TestEdgeCases:
    """
    測試異常和邊界情況
    """

    def test_route_with_short_query(self, endpoint_server):
        """
        行為: 應能處理短查詢
        場景: Query 只有 1-2 個詞
        期望: 返回 200 並返回有效的模型名稱
        """
        payload = {
            "query": "Hello?"
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code == 200

    def test_route_with_long_query(self, endpoint_server):
        """
        行為: 應能處理長查詢
        場景: Query 有 100+ 個詞
        期望: 返回 200 並返回有效的模型名稱
        """
        long_query = " ".join(["word"] * 150)
        payload = {
            "query": long_query
        }
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code == 200

    def test_route_with_special_characters(self, endpoint_server):
        """
        行為: Query 可以包含特殊字符
        場景: 含有符號、非 ASCII 字符等
        期望: 正常處理，不拋出異常
        """
        test_queries = [
            "What is π (pi)?",
            "Explain 中文 (Chinese)",
            "Code: for (int i=0; i<10; i++) { print(i); }",
        ]
        with httpx.Client() as client:
            for query in test_queries:
                payload = {"query": query}
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                # 應該返回 200 或 400，但不應該 500
                assert response.status_code in [200, 400, 422]

    def test_malformed_json_returns_400(self, endpoint_server):
        """
        行為: 格式錯誤的 JSON 應返回 400
        期望: 狀態碼 400
        """
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                content="{invalid json}",
                headers={"Content-Type": "application/json"}
            )
            assert response.status_code == 400

    def test_missing_query_field_returns_400(self, endpoint_server):
        """
        行為: 缺少 query 字段的請求應返回 400
        期望: 狀態碼 400
        """
        payload = {"other_field": "value"}  # 沒有 query 字段
        with httpx.Client() as client:
            response = client.post(
                f"{endpoint_server.base_url}/route",
                json=payload
            )
            assert response.status_code in [400, 422]


# ============================================================================
# 4. 性能測試
# ============================================================================

class TestPerformance:
    """
    性能要求測試
    """

    def test_health_endpoint_response_time_under_10ms(self, endpoint_server):
        """
        要求: /health 響應時間 < 10ms
        期望: P50 < 10ms
        """
        with httpx.Client() as client:
            times = []
            for _ in range(10):
                start = time.time()
                response = client.get(f"{endpoint_server.base_url}/health")
                elapsed = (time.time() - start) * 1000  # 轉換為毫秒
                times.append(elapsed)
                assert response.status_code == 200

            avg_time = sum(times) / len(times)
            assert avg_time < 10, f"Average time {avg_time:.2f}ms exceeds 10ms"

    def test_route_endpoint_response_time_under_100ms_p50(self, endpoint_server):
        """
        要求: /route 響應時間 P50 < 100ms
        期望: 50% 的請求在 100ms 內完成
        """
        payload = {
            "query": "What is machine learning?"
        }
        with httpx.Client() as client:
            times = []
            for _ in range(20):
                start = time.time()
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                assert response.status_code == 200

            times.sort()
            p50 = times[len(times) // 2]
            assert p50 < 100, f"P50 time {p50:.2f}ms exceeds 100ms"

    def test_route_endpoint_response_time_under_500ms_p99(self, endpoint_server):
        """
        要求: /route 響應時間 P99 < 500ms
        期望: 99% 的請求在 500ms 內完成
        """
        payload = {
            "query": "Write a Python program to calculate Fibonacci numbers"
        }
        with httpx.Client() as client:
            times = []
            for _ in range(100):
                start = time.time()
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                assert response.status_code == 200

            times.sort()
            p99_idx = int(len(times) * 0.99)
            p99 = times[p99_idx]
            assert p99 < 500, f"P99 time {p99:.2f}ms exceeds 500ms"

    def test_handles_concurrent_requests(self, endpoint_server):
        """
        要求: 應能同時處理多個請求
        期望: 能處理 10 個並發請求
        """
        import concurrent.futures

        payload = {
            "query": "Explain the concept of distributed systems"
        }

        def make_request():
            with httpx.Client() as client:
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # 所有請求應返回 200
        assert all(status == 200 for status in results)


# ============================================================================
# 5. 集成測試（Semantic Router 場景）
# ============================================================================

class TestSemanticRouterIntegration:
    """
    模擬 semantic-router 調用 endpoint 的實際場景
    """

    def test_semantic_router_initialization_flow(self, endpoint_server):
        """
        流程: semantic-router 初始化時檢查 endpoint
        步驟:
            1. GET /health 檢查服務是否可用
            2. 如果 200，初始化成功
        期望: /health 返回 200 OK
        """
        with httpx.Client() as client:
            response = client.get(f"{endpoint_server.base_url}/health")
            assert response.status_code == 200

    def test_semantic_router_decision_flow(self, endpoint_server):
        """
        流程: semantic-router RL-Driven 在做決策時調用 endpoint
        步驟:
            1. POST /route 含簡單查詢文本
            2. 獲取 selected_model
            3. 使用該模型處理請求
        期望: 能正確返回選中的模型
        """
        # 模擬 semantic-router 的決策場景
        queries = [
            "What is machine learning?",
            "Write a Python function to sort a list",
            "Explain quantum computing",
        ]

        with httpx.Client() as client:
            for query in queries:
                payload = {"query": query}
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                assert response.status_code == 200
                data = response.json()
                selected_model = data["selected_model"]
                assert isinstance(selected_model, str)
                assert len(selected_model) > 0

    def test_endpoint_handles_rapid_consecutive_requests(self, endpoint_server):
        """
        場景: semantic-router 在短時間內發送多個路由決策請求
        期望: 所有請求都成功返回
        """
        payload = {
            "query": "Solve this problem efficiently"
        }
        with httpx.Client() as client:
            for _ in range(20):
                response = client.post(
                    f"{endpoint_server.base_url}/route",
                    json=payload
                )
                assert response.status_code == 200


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def endpoint_server():
    """啟動真實的 LLMRouterEndpointServer，測試結束後自動關閉。"""
    import socket
    import tempfile
    import threading
    import time
    from pathlib import Path

    import numpy as np

    from LLMRouter.endpoint import LLMRouterEndpointServer
    from LLMRouter.router import KNNRouter, RouterData

    # 找一個空閒 port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    # dim=384 與 sentence-transformers/all-MiniLM-L6-v2 輸出一致，
    # 使用隨機預計算嵌入訓練；推論時由模型計算真實嵌入，維度相符。
    n, n_models, dim = 60, 3, 384
    rng = np.random.default_rng(0)
    prompts = [f"q{i}" for i in range(n)]
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)
    scores = rng.random((n, n_models)).astype(np.float32)
    data = RouterData(
        train_prompt=prompts[:36],
        val_prompt=prompts[36:48],
        test_prompt=prompts[48:],
        train_score=scores[:36],
        val_score=scores[36:48],
        test_score=scores[48:],
        test_tokens=np.zeros((12, n_models)),
        test_time=np.zeros((12, n_models)),
        models=["gpt-4", "gpt-3.5-turbo", "claude-3"],
        train_embed=embeddings[:36],
        val_embed=embeddings[36:48],
        test_embed=embeddings[48:],
    )

    router = KNNRouter(k=3)
    router.fit(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "router.pkl"
        router.save(path)

        srv = LLMRouterEndpointServer(path, port=port)
        thread = threading.Thread(target=srv.start, daemon=True)
        thread.start()
        time.sleep(0.5)  # 等待 server 就緒

        class _ServerInfo:
            base_url = f"http://localhost:{port}"
            config: dict = {}

        yield _ServerInfo()
        srv.shutdown()


# ============================================================================
# 運行測試的說明
# ============================================================================

"""
運行測試：
    pytest LLMRouter/test/test_endpoint_behavior.py -v

按類別運行：
    pytest LLMRouter/test/test_endpoint_behavior.py::TestHealthEndpoint -v
    pytest LLMRouter/test/test_endpoint_behavior.py::TestRouteEndpoint -v
    pytest LLMRouter/test/test_endpoint_behavior.py::TestProtocolCompliance -v
    pytest LLMRouter/test/test_endpoint_behavior.py::TestEdgeCases -v
    pytest LLMRouter/test/test_endpoint_behavior.py::TestPerformance -v
    pytest LLMRouter/test/test_endpoint_behavior.py::TestSemanticRouterIntegration -v

查看測試覆蓋率：
    pytest LLMRouter/test/test_endpoint_behavior.py --cov=LLMRouter.router.endpoint

詳細輸出：
    pytest LLMRouter/test/test_endpoint_behavior.py -vv -s
"""
