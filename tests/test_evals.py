"""
測試 LLMRouter 評估方法主要功能
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch


@pytest.mark.unit
class TestRandomEval:
    """測試隨機評估方法"""

    def test_random_eval(self):
        """測試隨機評估回傳有效分數"""
        from LLMRouter.datasets.evals import random

        dataset = {"answer": "test answer"}
        response = {"text": "test response"}

        result = random.eval(dataset, response)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


@pytest.mark.unit
class TestSimilarEval:
    """測試相似度評估方法"""

    @patch('LLMRouter.datasets.evals.similar.get_model')
    def test_similar_eval(self, mock_get_model):
        """測試相似度評估回傳有效分數"""
        mock_model = Mock()
        mock_model.encode.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.9, 0.1, 0.0])
        ]
        mock_get_model.return_value = mock_model

        from LLMRouter.datasets.evals import similar

        dataset = {"answer": "The answer is 42.\n#### 42"}
        response = {"text": "42"}

        result = similar.eval(dataset, response)

        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    @patch('LLMRouter.datasets.evals.similar.get_model')
    def test_similar_eval_extracts_answer(self, mock_get_model):
        """測試正確擷取 #### 後的答案"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
        mock_get_model.return_value = mock_model

        from LLMRouter.datasets.evals import similar

        dataset = {"answer": "Long explanation here.\n#### 42"}
        response = {"text": "42"}

        similar.eval(dataset, response)

        # 驗證 encode 被呼叫時使用 "42" (#### 後面的部分)
        calls = mock_model.encode.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0].strip() == "42"

    def test_cosine_similarity(self):
        """測試 cosine similarity 基本功能"""
        from LLMRouter.datasets.evals.similar import cosine_similarity

        # 相同向量應該得到 1.0
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(1.0, abs=0.01)

        # 正交向量應該得到 0.0
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(vec1, vec2) == pytest.approx(0.0, abs=0.01)
