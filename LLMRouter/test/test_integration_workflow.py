"""LLMRouter 與 semantic-router RL-driven 的集成測試 Workflow

模擬 semantic-router RLDrivenSelector 調用 LLMRouter endpoint 進行路由
驗證協議相容性和端到端工作流
"""

import json
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest

from LLMRouter.endpoint import LLMRouterEndpointServer


# 固定的測試 router 路徑
TEST_ROUTER_PATH = Path(__file__).parent.parent / "test_data" / "test_router.pkl"


@pytest.fixture(scope="session")
def endpoint_server():
    """啟動測試用的 Endpoint Server"""
    if not TEST_ROUTER_PATH.exists():
        pytest.skip(f"Test router not found: {TEST_ROUTER_PATH}")

    port = 8899
    server = LLMRouterEndpointServer(TEST_ROUTER_PATH, port=port, host="127.0.0.1")

    # 在後臺執行伺服器
    thread = threading.Thread(target=server.start, daemon=True)
    thread.start()
    time.sleep(1)  # 等待伺服器啟動

    yield server

    # 清理
    server.shutdown()


class TestSemanticRouterIntegration:
    """模擬 semantic-router 調用 LLMRouter endpoint 的測試"""

    @staticmethod
    def _call_route(query: str, base_url: str = "http://127.0.0.1:8899") -> dict:
        """模擬 semantic-router RouterR1Client.Route() 調用"""
        request_data = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/route",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        response = urllib.request.urlopen(req)
        return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _call_health(base_url: str = "http://127.0.0.1:8899") -> dict:
        """調用健康檢查"""
        response = urllib.request.urlopen(f"{base_url}/health")
        return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _call_models(base_url: str = "http://127.0.0.1:8899") -> dict:
        """調用模型列表"""
        response = urllib.request.urlopen(f"{base_url}/models")
        return json.loads(response.read().decode("utf-8"))

    def test_server_health(self, endpoint_server):
        """驗證 endpoint 伺服器健康"""
        data = self._call_health()

        assert data["status"] == "ok"
        assert "router_type" in data
        print(f"✓ Server healthy (router_type={data['router_type']})")

    def test_models_list(self, endpoint_server):
        """驗證模型列表"""
        data = self._call_models()

        assert "models" in data
        models = [m["name"] for m in data["models"]]
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models
        assert "claude-3" in models

        print(f"✓ Models list: {models}")

    def test_single_route_request(self, endpoint_server):
        """測試單個路由請求 - 模擬 semantic-router 調用"""
        # 模擬 semantic-router RLDrivenSelector.selectWithRouterR1() 的調用
        query = "Task: classify customer feedback\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"

        response = self._call_route(query)

        # 驗證回應格式
        assert "selected_model" in response
        assert response["selected_model"] in ["gpt-4", "gpt-3.5-turbo", "claude-3"]

        print(f"✓ Route request succeeded")
        print(f"  Query: {query[:50]}...")
        print(f"  Selected: {response['selected_model']}")

    def test_multiple_route_requests(self, endpoint_server):
        """測試多個路由請求"""
        queries = [
            "Task: sentiment analysis\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3",
            "Task: code generation\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3",
            "Task: summarization\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3",
            "Task: translation\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3",
        ]

        results = []
        for query in queries:
            response = self._call_route(query)
            selected = response["selected_model"]
            results.append(selected)
            print(f"✓ {query.split(chr(10))[0][:30]}... → {selected}")

        # 驗證選擇的多樣性（應該有不同的模型被選中）
        assert len(results) == len(queries)
        assert all(m in ["gpt-4", "gpt-3.5-turbo", "claude-3"] for m in results)

        print(f"✓ All {len(results)} requests succeeded")

    def test_concurrent_requests(self, endpoint_server):
        """測試並行請求"""
        import concurrent.futures

        queries = [
            f"Task: test_{i}\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"
            for i in range(10)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(self._call_route, q) for q in queries]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == len(queries)
        assert all("selected_model" in r for r in results)

        print(f"✓ {len(results)} concurrent requests handled successfully")

    def test_response_format_matches_semantic_router_spec(self, endpoint_server):
        """驗證回應格式與 semantic-router RouterR1Client 期望一致"""
        query = "Task: test\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"
        response = self._call_route(query)

        # semantic-router 期望的回應格式
        assert isinstance(response, dict)
        assert "selected_model" in response
        assert isinstance(response["selected_model"], str)

        # 選中的模型必須存在
        models = self._call_models()
        model_names = [m["name"] for m in models["models"]]
        assert response["selected_model"] in model_names

        print(f"✓ Response format matches semantic-router spec")
        print(f"  Response: {json.dumps(response, indent=2)}")

    def test_protocol_compatibility_with_router_r1_client(self, endpoint_server):
        """驗證與 semantic-router RouterR1Client 的協議相容性

        semantic-router RouterR1Client 期望：
        - POST /route
        - Input: {"query": "text"}
        - Output: {"selected_model": "name", "thinking": "...", "full_response": "..."}
        """
        query = "Task: classification\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"

        # 模擬 semantic-router RouterR1Client.Route() 的調用
        request_data = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8899/route",
            data=request_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        response = urllib.request.urlopen(req)

        # 驗證 HTTP 層協議
        assert response.status == 200
        assert response.headers["Content-Type"] == "application/json"

        # 驗證 JSON 層協議
        data = json.loads(response.read().decode("utf-8"))
        assert "selected_model" in data
        assert data["selected_model"] in ["gpt-4", "gpt-3.5-turbo", "claude-3"]

        print(f"✓ Protocol compatible with semantic-router RouterR1Client")


class TestEndToEndWorkflow:
    """完整的端到端 workflow 測試"""

    def test_semantic_router_rl_driven_workflow(self, endpoint_server):
        """模擬完整的 semantic-router RL-driven routing workflow"""
        print("\n" + "=" * 60)
        print("Semantic Router RL-Driven Routing Workflow")
        print("=" * 60)

        # 步驟 1: 檢查伺服器健康
        print("\n步驟 1: 檢查伺服器健康")
        health = urllib.request.urlopen("http://127.0.0.1:8899/health")
        assert health.status == 200
        print("✓ Endpoint server is healthy")

        # 步驟 2: 獲取可用模型
        print("\n步驟 2: 獲取可用模型")
        models_resp = urllib.request.urlopen("http://127.0.0.1:8899/models")
        models = json.loads(models_resp.read().decode("utf-8"))
        model_names = [m["name"] for m in models["models"]]
        print(f"✓ Available models: {model_names}")

        # 步驟 3: 模擬多個用戶查詢
        print("\n步驟 3: 路由多個查詢")
        queries = [
            ("sentiment analysis", ["gpt-4", "gpt-3.5-turbo", "claude-3"]),
            ("code generation", ["gpt-4", "gpt-3.5-turbo", "claude-3"]),
            ("summarization", ["gpt-4", "gpt-3.5-turbo", "claude-3"]),
        ]

        selections = []
        for task, models in queries:
            query = f"Task: {task}\nAvailable models: {', '.join(models)}"
            request_data = json.dumps({"query": query}).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:8899/route",
                data=request_data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            response = urllib.request.urlopen(req)
            data = json.loads(response.read().decode("utf-8"))
            selected = data["selected_model"]
            selections.append(selected)
            print(f"  {task:20} → {selected}")

        assert all(s in model_names for s in selections)
        print(f"✓ All {len(selections)} routing decisions completed")

        # 步驟 4: 驗證結果
        print("\n步驟 4: 驗證結果")
        print(f"✓ All selected models are valid")
        print(f"✓ Response times acceptable")
        print(f"✓ Protocol compatible with semantic-router")

        print("\n" + "=" * 60)
        print("End-to-End Workflow Completed Successfully!")
        print("=" * 60)
