"""
測試 LLMRouter 資料集主要功能
"""
import pytest
import json
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestLoadDataset:
    """測試資料集載入功能"""

    def test_load_dataset_with_responses_and_evals(
        self, temp_dataset_dir, sample_qa_data,
        create_test_dataset, create_test_response, create_test_eval
    ):
        """測試載入包含回應和評估的完整資料集"""
        # 準備測試資料
        create_test_dataset("test_dataset", sample_qa_data)

        # 準備回應資料
        response_data = [
            {"key": "test_0", "text": "Answer is 4",
             "usage": {"total_tokens": 15}, "response_time": 1.5},
            {"key": "test_1", "text": "Answer is 15",
             "usage": {"total_tokens": 20}, "response_time": 2.0}
        ]
        create_test_response("test_dataset", "test-model", response_data)

        # 準備評估資料
        eval_data = [
            {"key": "test_0", "point": 0.95},
            {"key": "test_1", "point": 0.85}
        ]
        create_test_eval("test_dataset", "test-model", "similar", eval_data)

        # 執行測試
        with patch('LLMRouter.datasets.dataset_path', str(temp_dataset_dir)):
            from LLMRouter.datasets import load_dataset
            results = load_dataset(["test_dataset"], ["test-model"], ["similar"])

        # 驗證結果
        assert len(results) == 3
        assert results[0]["key"] == "test_0"
        assert results[0]["question"] == "What is 2 + 2?"
        assert "test-model_response" in results[0]
        assert results[0]["test-model_response"]["text"] == "Answer is 4"
        assert "test-model_similar_eval" in results[0]
        assert results[0]["test-model_similar_eval"]["point"] == 0.95

    def test_load_dataset_multiple_models(
        self, temp_dataset_dir, sample_qa_data,
        create_test_dataset, create_test_response, create_test_eval
    ):
        """測試載入多個模型的資料"""
        create_test_dataset("test_dataset", sample_qa_data[:2])

        # 為兩個模型準備資料
        for model in ["model-a", "model-b"]:
            response_data = [
                {"key": "test_0", "text": f"{model} answer",
                 "usage": {"total_tokens": 15}, "response_time": 1.5}
            ]
            create_test_response("test_dataset", model, response_data)
            eval_data = [{"key": "test_0", "point": 0.9}]
            create_test_eval("test_dataset", model, "similar", eval_data)

        # 執行測試
        with patch('LLMRouter.datasets.dataset_path', str(temp_dataset_dir)):
            from LLMRouter.datasets import load_dataset
            results = load_dataset(["test_dataset"], ["model-a", "model-b"], ["similar"])

        # 驗證結果
        assert len(results) == 2
        assert "model-a_response" in results[0]
        assert "model-b_response" in results[0]
        assert results[0]["model-a_response"]["text"] == "model-a answer"
        assert results[0]["model-b_response"]["text"] == "model-b answer"


@pytest.mark.unit
class TestAddModelResponse:
    """測試模型回應生成功能"""

    @patch('LLMRouter.datasets.litellm_router')
    def test_add_model_response(
        self, mock_router, temp_dataset_dir, sample_qa_data,
        create_test_dataset, mock_litellm_completion
    ):
        """測試生成模型回應"""
        create_test_dataset("test_dataset", sample_qa_data[:1])
        mock_router.completion.return_value = mock_litellm_completion

        with patch('LLMRouter.datasets.dataset_path', str(temp_dataset_dir)):
            from LLMRouter.datasets import add_model_response
            add_model_response("test_dataset", "test-model")

        # 驗證結果
        response_file = temp_dataset_dir / "responses" / "test_dataset" / "test-model.jsonl"
        assert response_file.exists()

        with open(response_file) as f:
            response = json.loads(f.readline())
            assert response["key"] == "test_0"
            assert response["text"] == "The answer is 4."
            assert "usage" in response
            assert "response_time" in response

    @patch('LLMRouter.datasets.litellm_router')
    def test_skip_existing_response(
        self, mock_router, temp_dataset_dir, sample_qa_data,
        create_test_dataset, create_test_response
    ):
        """測試跳過已存在的回應"""
        create_test_dataset("test_dataset", sample_qa_data[:1])

        existing_response = [
            {"key": "test_0", "text": "Existing answer",
             "usage": {"total_tokens": 10}, "response_time": 1.0}
        ]
        create_test_response("test_dataset", "test-model", existing_response)

        with patch('LLMRouter.datasets.dataset_path', str(temp_dataset_dir)):
            from LLMRouter.datasets import add_model_response
            add_model_response("test_dataset", "test-model")

        # 驗證沒有呼叫 API
        mock_router.completion.assert_not_called()


@pytest.mark.unit
class TestAddModelResponseEval:
    """測試評估生成功能"""

    def test_add_model_response_eval(
        self, temp_dataset_dir, sample_qa_data,
        create_test_dataset, create_test_response
    ):
        """測試生成評估結果"""
        create_test_dataset("test_dataset", sample_qa_data[:2])

        response_data = [
            {"key": "test_0", "text": "4",
             "usage": {"total_tokens": 15}, "response_time": 1.5},
            {"key": "test_1", "text": "15",
             "usage": {"total_tokens": 20}, "response_time": 2.0}
        ]
        create_test_response("test_dataset", "test-model", response_data)

        # Mock 評估方法
        mock_eval_module = Mock()
        mock_eval_module.eval.side_effect = [0.95, 0.85]

        with patch('LLMRouter.datasets.dataset_path', str(temp_dataset_dir)):
            with patch('LLMRouter.datasets.evals.random', mock_eval_module):
                from LLMRouter.datasets import add_model_response_eval
                add_model_response_eval("test_dataset", "test-model", "random")

        # 驗證結果
        eval_file = temp_dataset_dir / "evals" / "test_dataset" / "random" / "test-model.jsonl"
        assert eval_file.exists()

        with open(eval_file) as f:
            lines = f.readlines()
            assert len(lines) == 2

            eval_0 = json.loads(lines[0])
            assert eval_0["key"] == "test_0"
            assert eval_0["point"] == 0.95
