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


@pytest.mark.unit
class TestSquadEval:
    """測試 SQuAD 風格評估方法"""

    def test_normalize_answer(self):
        """測試答案標準化功能"""
        from LLMRouter.datasets.evals.squad import normalize_answer

        # 測試移除標點符號
        assert normalize_answer("Hello, World!") == "hello world"

        # 測試移除冠詞
        assert normalize_answer("The quick brown fox") == "quick brown fox"
        assert normalize_answer("A cat and an elephant") == "cat and elephant"

        # 測試標準化空白
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_compute_exact(self):
        """測試精確匹配計算"""
        from LLMRouter.datasets.evals.squad import compute_exact

        # 完全相同
        assert compute_exact("42", "42") == 1

        # 大小寫不同但標準化後相同
        assert compute_exact("Hello World", "hello world") == 1

        # 標點符號不同但標準化後相同
        assert compute_exact("42!", "42") == 1

        # 完全不同
        assert compute_exact("42", "99") == 0

    def test_compute_f1(self):
        """測試 F1 分數計算"""
        from LLMRouter.datasets.evals.squad import compute_f1

        # 完全相同
        f1 = compute_f1("the quick brown fox", "the quick brown fox")
        assert f1 == pytest.approx(1.0)

        # 部分重疊
        f1 = compute_f1("the quick brown fox", "quick brown dog")
        assert 0.0 < f1 < 1.0

        # 完全不同
        f1 = compute_f1("cat", "dog")
        assert f1 == 0.0

        # 空答案
        assert compute_f1("", "") == 1
        assert compute_f1("test", "") == 0

    def test_eval_with_gsm8k_format(self):
        """測試處理 GSM8K 格式答案"""
        from LLMRouter.datasets.evals.squad import eval

        # GSM8K 格式：答案在 #### 之後
        dataset = {"answer": "The calculation shows...\n#### 42"}
        response = {"text": "42"}

        score = eval(dataset, response)
        assert isinstance(score, float)
        assert score > 0.5  # 應該有高分

    def test_eval_exact_match(self):
        """測試精確匹配評估函數"""
        from LLMRouter.datasets.evals.squad import eval_exact

        dataset = {"answer": "#### 42"}
        response = {"text": "42"}

        score = eval_exact(dataset, response)
        assert score == 1

        response = {"text": "99"}
        score = eval_exact(dataset, response)
        assert score == 0

    def test_eval_both(self):
        """測試同時返回兩種分數"""
        from LLMRouter.datasets.evals.squad import eval_both

        dataset = {"answer": "The answer is 42.\n#### 42"}
        response = {"text": "42"}

        result = eval_both(dataset, response)

        assert isinstance(result, dict)
        assert 'f1' in result
        assert 'exact' in result
        assert 0.0 <= result['f1'] <= 1.0
        assert result['exact'] in [0, 1]
