"""
測試 Router Model Names 綁定 + Save/Load 完整工作流

驗證：
1. fit() 時自動綁定 model_names
2. predict() 返回模型名稱
3. save/load 保留 model_names + 超參數 + 訓練狀態
4. 重複 fit() 檢查一致性
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


class TestRouterWorkflow:
    """Router Model Names 綁定 + Save/Load 工作流測試"""

    def test_knn_router_workflow(self, sample_data):
        """KNNRouter: 訓練 → 預測 → 保存 → 載入 → 預測 → 重複訓練"""
        # 1. 初始化時 model_names 為 None
        router = KNNRouter(k=5)
        assert router.model_names is None

        # 2. fit() 自動綁定 model_names
        router.fit(sample_data)
        assert router.model_names == ["gpt-4", "gpt-3.5-turbo", "claude-3"]

        # 3. predict() 返回模型名稱
        pred1 = router.predict(sample_data.test_prompt)
        assert len(pred1) == len(sample_data.test_prompt)
        assert all(name in router.model_names for name in pred1)

        # 4. 保存並載入
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "knn.pkl"
            router.save(path)

            loaded = KNNRouter.load(path)
            assert loaded.model_names == router.model_names
            assert loaded.k == 5

            # 5. 載入後的 router 可以預測
            pred2 = loaded.predict(sample_data.test_prompt)
            assert len(pred2) == len(sample_data.test_prompt)
            assert all(name in loaded.model_names for name in pred2)

        # 6. 重複訓練同一資料（應成功）
        router.fit(sample_data)
        assert router.model_names == ["gpt-4", "gpt-3.5-turbo", "claude-3"]

    def test_oracle_router_workflow(self, sample_data):
        """OracleRouter: 訓練 → 保存 → 載入 → 評估"""
        router = OracleRouter()
        assert router.model_names is None

        router.fit(sample_data)
        assert router.model_names == sample_data.models

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "oracle.pkl"
            router.save(path)

            loaded = OracleRouter.load(path)
            assert loaded.model_names == sample_data.models

            # 載入後能評估
            metrics = loaded.evaluate(sample_data)
            assert "mu" in metrics and "vb" in metrics

    def test_random_router_workflow(self, sample_data):
        """RandomRouter: 訓練 → 保存 → 載入 → 評估"""
        router = RandomRouter(seed=42)
        assert router.model_names is None

        router.fit(sample_data)
        assert router.model_names == sample_data.models

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random.pkl"
            router.save(path)

            loaded = RandomRouter.load(path)
            assert loaded.model_names == sample_data.models
            assert loaded.seed == 42

            # 載入後能評估
            metrics = loaded.evaluate(sample_data)
            assert "mu" in metrics

    def test_model_mismatch_error(self, sample_data):
        """防止綁定到不同的 model 列表"""
        router = KNNRouter(k=5)
        router.fit(sample_data)

        # 建立不同 models 的資料
        data_diff = sample_data
        data_diff.models = ["model-a", "model-b"]

        # 應拋出 ValueError
        with pytest.raises(ValueError, match="Model mismatch"):
            router.fit(data_diff)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
