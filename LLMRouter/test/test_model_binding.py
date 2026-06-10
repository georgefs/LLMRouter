"""
測試 BaseRouter model_names 綁定機制

驗證：
1. fit() 時自動綁定 model_names
2. 重複 fit() 檢查一致性
3. save/load 序列化 model_names
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from LLMRouter.router import (
    RouterData,
    KNNRouter,
    OracleRouter,
    RandomRouter,
)


@pytest.fixture
def sample_data():
    """建立簡單的 RouterData 用於測試"""
    n_samples = 50
    n_models = 3
    # 必須與 KNNRouter 的預設 embedding 模型維度一致
    # sentence-transformers/all-MiniLM-L6-v2 → 384 維
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


class TestModelNamesBinding:
    """測試 model_names 綁定"""

    def test_model_names_none_before_fit(self, sample_data):
        """訓練前 model_names 應為 None"""
        router = KNNRouter(k=5)
        assert router.model_names is None

    def test_model_names_bound_after_fit(self, sample_data):
        """fit() 後 model_names 應被設置"""
        router = KNNRouter(k=5)
        router.fit(sample_data)
        assert router.model_names == ["gpt-4", "gpt-3.5-turbo", "claude-3"]

    def test_model_names_matches_data_models(self, sample_data):
        """router.model_names 應與 data.models 相同"""
        router = KNNRouter(k=5)
        router.fit(sample_data)
        assert router.model_names == sample_data.models

    def test_repeated_fit_with_same_models(self, sample_data):
        """重複 fit() 同一資料應無錯誤"""
        router = KNNRouter(k=5)
        router.fit(sample_data)
        first_names = router.model_names.copy()

        # 重複 fit() 同一資料應成功
        router.fit(sample_data)
        assert router.model_names == first_names

    def test_repeated_fit_with_different_models_raises_error(self, sample_data):
        """fit() 不同 models 應拋出錯誤"""
        router = KNNRouter(k=5)
        router.fit(sample_data)

        # 建立不同 models 的 data
        data_diff = sample_data
        data_diff.models = ["model-a", "model-b"]

        # 應拋出 ValueError
        with pytest.raises(ValueError, match="Model mismatch"):
            router.fit(data_diff)


class TestSaveLoad:
    """測試 save/load 序列化"""

    def test_knn_save_load_preserves_model_names(self, sample_data):
        """KNNRouter save/load 應保留 model_names"""
        router = KNNRouter(k=5)
        router.fit(sample_data)
        original_names = router.model_names.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knn.pkl"
            router.save(path)

            loaded = KNNRouter.load(path)
            assert loaded.model_names == original_names

    def test_knn_save_load_preserves_hyperparameters(self, sample_data):
        """KNNRouter save/load 應保留超參數"""
        router = KNNRouter(k=7, emb_model="test-model")
        router.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knn.pkl"
            router.save(path)

            loaded = KNNRouter.load(path)
            assert loaded.k == 7
            assert loaded.emb_model == "test-model"

    def test_knn_loaded_router_can_predict(self, sample_data):
        """載入的 KNNRouter 應能進行預測"""
        router = KNNRouter(k=5)
        router.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knn.pkl"
            router.save(path)

            loaded = KNNRouter.load(path)
            predictions = loaded.predict_indices(sample_data.test_prompt)
            assert len(predictions) == len(sample_data.test_prompt)
            assert all(0 <= idx < len(loaded.model_names) for idx in predictions)

    def test_oracle_save_load(self, sample_data):
        """OracleRouter save/load 應保留 model_names"""
        router = OracleRouter()
        router.fit(sample_data)
        original_names = router.model_names.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.pkl"
            router.save(path)

            loaded = OracleRouter.load(path)
            assert loaded.model_names == original_names

    def test_random_save_load(self, sample_data):
        """RandomRouter save/load 應保留 model_names 和 seed"""
        router = RandomRouter(seed=123)
        router.fit(sample_data)
        original_names = router.model_names.copy()
        original_seed = router.seed

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random.pkl"
            router.save(path)

            loaded = RandomRouter.load(path)
            assert loaded.model_names == original_names
            assert loaded.seed == original_seed


class TestCompleteWorkflow:
    """測試完整的訓練→保存→載入→預測流程"""

    def test_knn_complete_workflow(self, sample_data):
        """完整流程：訓練 → 保存 → 載入 → 預測"""
        # 1. 訓練
        router = KNNRouter(k=5)
        assert router.model_names is None

        router.fit(sample_data)
        assert router.model_names == sample_data.models
        assert len(router.model_names) == 3

        # 2. 預測（訓練後）- 直接返回模型名稱
        pred1 = router.predict(sample_data.test_prompt)
        assert len(pred1) == len(sample_data.test_prompt)
        assert all(model in ["gpt-4", "gpt-3.5-turbo", "claude-3"] for model in pred1)

        # 3. 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knn.pkl"
            router.save(path)
            assert path.exists()

            # 4. 載入
            loaded = KNNRouter.load(path)
            assert loaded.model_names == sample_data.models
            assert loaded.k == router.k

            # 5. 預測（載入後）- 直接返回模型名稱
            pred2 = loaded.predict(sample_data.test_prompt)
            assert len(pred2) == len(sample_data.test_prompt)
            assert all(model in ["gpt-4", "gpt-3.5-turbo", "claude-3"] for model in pred2)

    def test_oracle_complete_workflow(self, sample_data):
        """OracleRouter 完整流程"""
        router = OracleRouter()
        router.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.pkl"
            router.save(path)

            loaded = OracleRouter.load(path)
            assert loaded.model_names == sample_data.models

            # Oracle 的 evaluate 應工作正常
            metrics = loaded.evaluate(sample_data)
            assert "mu" in metrics
            assert "vb" in metrics

    def test_random_complete_workflow(self, sample_data):
        """RandomRouter 完整流程"""
        router = RandomRouter(seed=42)
        router.fit(sample_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random.pkl"
            router.save(path)

            loaded = RandomRouter.load(path)
            assert loaded.model_names == sample_data.models
            assert loaded.seed == 42

            # RandomRouter 的 evaluate 應工作正常
            metrics = loaded.evaluate(sample_data)
            assert "mu" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
