"""
Tests for LLMRouter.router module.
覆蓋 DataPreparer, RouterData (save/load), OracleRouter, RandomRouter.
KNN / MF / SW / RoBERTa 需要 heavy deps（torch, transformers）— 僅做 import 檢查。
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── fixtures ─────────────────────────────────────────────────────────────────


def _make_records(n_samples=30, n_models=3, seed=0):
    """建立假的 extract() 記錄。"""
    rng = np.random.default_rng(seed)
    model_names = [f"model_{i}" for i in range(n_models)]
    records = []
    for i in range(n_samples):
        for m in model_names:
            records.append(
                {
                    "key": str(i),
                    "question": f"question {i}",
                    "answer": f"answer {i}",
                    "model": m,
                    "score": float(rng.integers(0, 2)),
                }
            )
    return records, model_names


@pytest.fixture()
def small_data():
    """RouterData with 30 samples, 3 models."""
    from LLMRouter.router import DataPreparer

    records, _ = _make_records(30, 3, seed=7)
    return DataPreparer().prepare(records, seed=42)


# ── DataPreparer ──────────────────────────────────────────────────────────────


class TestDataPreparer:
    def test_basic_shape(self, small_data):
        d = small_data
        total = len(d.train_prompt) + len(d.val_prompt) + len(d.test_prompt)
        assert total == 30
        assert d.train_score.shape == (len(d.train_prompt), 3)
        assert d.val_score.shape == (len(d.val_prompt), 3)
        assert d.test_score.shape == (len(d.test_prompt), 3)

    def test_default_split(self, small_data):
        d = small_data
        # 60 / 20 / 20 split → 30 samples: train=18, val=6, test=6
        assert len(d.train_prompt) == 18
        assert len(d.val_prompt) == 6
        assert len(d.test_prompt) == 6

    def test_models_sorted(self, small_data):
        assert small_data.models == sorted(small_data.models)

    def test_tokens_zeros_when_not_provided(self, small_data):
        assert np.all(small_data.test_tokens == 0.0)
        assert np.all(small_data.test_time == 0.0)

    def test_tokens_populated(self):
        from LLMRouter.router import DataPreparer

        records, model_names = _make_records(20, 2, seed=1)
        tokens = {(r["key"], r["model"]): 100.0 for r in records}
        times = {(r["key"], r["model"]): 0.5 for r in records}
        data = DataPreparer().prepare(records, tokens_by_key_model=tokens, times_by_key_model=times, seed=42)
        assert np.all(data.test_tokens == 100.0)
        assert np.all(data.test_time == 0.5)

    def test_empty_records_raises(self):
        from LLMRouter.router import DataPreparer

        with pytest.raises(ValueError, match="空"):
            DataPreparer().prepare([])

    def test_bad_ratio_raises(self):
        from LLMRouter.router import DataPreparer

        records, _ = _make_records(20, 2)
        with pytest.raises(ValueError, match="test set"):
            DataPreparer().prepare(records, train_ratio=0.7, val_ratio=0.3)

    def test_custom_split_ratio(self):
        from LLMRouter.router import DataPreparer

        records, _ = _make_records(100, 2, seed=5)
        data = DataPreparer().prepare(records, train_ratio=0.7, val_ratio=0.1, seed=0)
        assert len(data.train_prompt) == 70
        assert len(data.val_prompt) == 10
        assert len(data.test_prompt) == 20

    def test_from_manager(self, tmp_path):
        """from_manager 整合 DatasetManager 的 stub。"""
        from LLMRouter.router import DataPreparer

        mgr = MagicMock()
        records, _ = _make_records(20, 2, seed=3)
        mgr.extract.return_value = records
        mgr.get_responses.side_effect = FileNotFoundError("no file")

        data = DataPreparer().from_manager(
            mgr, datasets=["ds1"], models=["m0", "m1"], strategies="llm", seed=42
        )
        assert len(data.models) == 2
        assert data.train_score.shape[1] == 2


# ── RouterData save / load ────────────────────────────────────────────────────


class TestRouterDataSaveLoad:
    def test_roundtrip(self, small_data, tmp_path):
        from LLMRouter.router import RouterData

        path = tmp_path / "data.npz"
        small_data.save(path)
        loaded = RouterData.load(path)

        assert loaded.models == small_data.models
        assert loaded.train_prompt == small_data.train_prompt
        assert np.allclose(loaded.train_score, small_data.train_score)
        assert np.allclose(loaded.test_tokens, small_data.test_tokens)

    def test_creates_parent_dirs(self, small_data, tmp_path):
        from LLMRouter.router import RouterData

        path = tmp_path / "deep" / "nested" / "data.npz"
        small_data.save(path)
        assert path.exists()


# ── OracleRouter ─────────────────────────────────────────────────────────────


class TestOracleRouter:
    def test_vb_equals_one(self, small_data):
        from LLMRouter.router import OracleRouter

        r = OracleRouter()
        r.fit(small_data)
        m = r.evaluate(small_data)
        assert m["vb"] == pytest.approx(1.0, abs=1e-9)

    def test_mu_is_oracle_performance(self, small_data):
        from LLMRouter.router import OracleRouter

        r = OracleRouter()
        r.fit(small_data)
        m = r.evaluate(small_data)
        expected = float(np.mean(np.max(small_data.test_score, axis=1)))
        assert m["mu"] == pytest.approx(expected, abs=1e-6)

    def test_predict_probs_raises(self, small_data):
        from LLMRouter.router import OracleRouter

        r = OracleRouter()
        r.fit(small_data)
        with pytest.raises(NotImplementedError):
            r.predict_probs(["hello"])

    def test_all_correct_ep_zero(self):
        """If every sample has a clear winner (score=1, rest=0), ep should be 0."""
        from LLMRouter.router import OracleRouter, RouterData

        n, m = 10, 3
        Y = np.zeros((n, m), dtype=np.float32)
        Y[:, 0] = 1.0  # model 0 always best
        data = RouterData(
            train_prompt=["q"] * n,
            val_prompt=[],
            test_prompt=["q"] * n,
            train_score=Y,
            val_score=np.zeros((0, m)),
            test_score=Y,
            test_tokens=np.zeros((n, m)),
            test_time=np.zeros((n, m)),
            models=["a", "b", "c"],
        )
        r = OracleRouter()
        r.fit(data)
        m_res = r.evaluate(data)
        assert m_res["ep"] == pytest.approx(0.0, abs=1e-9)

    def test_metrics_keys(self, small_data):
        from LLMRouter.router import OracleRouter

        r = OracleRouter()
        r.fit(small_data)
        keys = set(r.evaluate(small_data).keys())
        assert keys == {"mu", "vb", "ep", "avg_tokens", "avg_latency"}


# ── RandomRouter ─────────────────────────────────────────────────────────────


class TestRandomRouter:
    def test_vb_less_than_oracle(self, small_data):
        from LLMRouter.router import OracleRouter, RandomRouter

        rnd = RandomRouter(seed=0)
        rnd.fit(small_data)
        rnd_m = rnd.evaluate(small_data)

        oracle = OracleRouter()
        oracle.fit(small_data)
        oracle_m = oracle.evaluate(small_data)

        assert rnd_m["mu"] <= oracle_m["mu"] + 1e-6

    def test_predict_probs_uniform(self, small_data):
        from LLMRouter.router import RandomRouter

        rnd = RandomRouter(seed=0)
        rnd.fit(small_data)
        probs = rnd.predict_probs(small_data.test_prompt)
        assert probs.shape == (len(small_data.test_prompt), 3)
        # All rows should be uniform
        assert np.allclose(probs, probs[0])
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_metrics_keys(self, small_data):
        from LLMRouter.router import RandomRouter

        rnd = RandomRouter(seed=0)
        rnd.fit(small_data)
        keys = set(rnd.evaluate(small_data).keys())
        assert keys == {"mu", "vb", "ep", "avg_tokens", "avg_latency"}

    def test_not_fitted_raises(self):
        from LLMRouter.router import RandomRouter

        rnd = RandomRouter()
        with pytest.raises(RuntimeError, match="fit"):
            rnd.predict_probs(["hello"])


# ── BaseRouter.evaluate via subclass ─────────────────────────────────────────


class TestBaseRouterEvaluate:
    """驗證 BaseRouter.evaluate 的指標計算邏輯（透過 RandomRouter）。"""

    def test_mu_in_valid_range(self, small_data):
        from LLMRouter.router import RandomRouter

        rnd = RandomRouter(seed=1)
        rnd.fit(small_data)
        m = rnd.evaluate(small_data)
        assert 0.0 <= m["mu"] <= 1.0

    def test_ep_non_negative(self, small_data):
        from LLMRouter.router import RandomRouter

        rnd = RandomRouter(seed=1)
        rnd.fit(small_data)
        m = rnd.evaluate(small_data)
        assert m["ep"] >= 0.0


# ── RouterData.subsample_train ────────────────────────────────────────────────


class TestSubsampleTrain:
    def test_test_val_unchanged(self, small_data):
        sub = small_data.subsample_train(0.5, seed=0)
        assert sub.test_prompt == small_data.test_prompt
        assert sub.val_prompt == small_data.val_prompt
        assert np.allclose(sub.test_score, small_data.test_score)

    def test_train_size_reduced(self, small_data):
        n_full = len(small_data.train_prompt)
        sub = small_data.subsample_train(0.5, seed=0)
        assert len(sub.train_prompt) == max(1, round(n_full * 0.5))

    def test_fraction_one_same_size(self, small_data):
        sub = small_data.subsample_train(1.0, seed=0)
        assert len(sub.train_prompt) == len(small_data.train_prompt)

    def test_different_seeds_different_subsets(self, small_data):
        sub0 = small_data.subsample_train(0.5, seed=0)
        sub1 = small_data.subsample_train(0.5, seed=1)
        # 不同 seed 應選出不同子集（機率極低會相同）
        assert sub0.train_prompt != sub1.train_prompt

    def test_bad_fraction_raises(self, small_data):
        with pytest.raises(ValueError, match="float size"):
            small_data.subsample_train(0.0)
        with pytest.raises(ValueError, match="float size"):
            small_data.subsample_train(1.1)

    def test_int_size_fixed_count(self, small_data):
        sub = small_data.subsample_train(5)
        assert len(sub.train_prompt) == 5

    def test_int_size_capped_at_n(self, small_data):
        n = len(small_data.train_prompt)
        sub = small_data.subsample_train(n + 100)
        assert len(sub.train_prompt) == n

    def test_int_size_bad_raises(self, small_data):
        with pytest.raises(ValueError, match="int size"):
            small_data.subsample_train(0)

    def test_models_preserved(self, small_data):
        sub = small_data.subsample_train(0.3, seed=0)
        assert sub.models == small_data.models


# ── filter_by_variance ───────────────────────────────────────────────────────


class TestFilterByVariance:
    def _data_with_known_scores(self):
        """手工建立 RouterData，train_score 有可控的 variance。"""
        from LLMRouter.router import RouterData

        n, m = 10, 3
        # 前 4 筆：全 0（var=0）、全 1（var=0）交替 → 無鑑別度
        # 後 6 筆：[0,1,0]、[1,0,1] 交替 → var≈0.22
        scores = np.zeros((n, m), dtype=np.float32)
        scores[0] = [0, 0, 0]
        scores[1] = [1, 1, 1]
        scores[2] = [0, 0, 0]
        scores[3] = [1, 1, 1]
        for i in range(4, n):
            scores[i] = [0, 1, 0] if i % 2 == 0 else [1, 0, 1]

        return RouterData(
            train_prompt=[f"q{i}" for i in range(n)],
            val_prompt=[],
            test_prompt=["t0"],
            train_score=scores,
            val_score=np.zeros((0, m)),
            test_score=np.ones((1, m)),
            test_tokens=np.zeros((1, m)),
            test_time=np.zeros((1, m)),
            models=["a", "b", "c"],
        )

    def test_removes_zero_variance(self):
        data = self._data_with_known_scores()
        filtered = data.filter_by_variance(min_var=0.0)
        assert len(filtered.train_prompt) == 6

    def test_higher_threshold_removes_more(self):
        data = self._data_with_known_scores()
        filtered = data.filter_by_variance(min_var=0.2)
        assert len(filtered.train_prompt) == 6  # var≈0.22 > 0.2

    def test_test_val_unchanged(self):
        data = self._data_with_known_scores()
        filtered = data.filter_by_variance(0.0)
        assert filtered.test_prompt == data.test_prompt
        assert filtered.val_prompt == data.val_prompt

    def test_zero_threshold_keeps_all_discriminative(self):
        data = self._data_with_known_scores()
        # min_var=0 → only exact-tie samples removed
        filtered = data.filter_by_variance(0.0)
        variances = np.var(filtered.train_score, axis=1)
        assert np.all(variances > 0)


# ── deduplicate_train ─────────────────────────────────────────────────────────


class TestDeduplicateTrain:
    def _data_with_duplicates(self):
        """train_prompt 中有重複（完全相同的文字）。"""
        from LLMRouter.router import RouterData

        prompts = ["unique A", "duplicate", "duplicate", "unique B", "duplicate"]
        n, m = len(prompts), 2
        scores = np.random.default_rng(0).random((n, m)).astype(np.float32)
        return RouterData(
            train_prompt=prompts,
            val_prompt=[],
            test_prompt=["test"],
            train_score=scores,
            val_score=np.zeros((0, m)),
            test_score=np.ones((1, m)),
            test_tokens=np.zeros((1, m)),
            test_time=np.zeros((1, m)),
            models=["x", "y"],
        )

    def test_removes_duplicates(self):
        data = self._data_with_duplicates()
        # 預先提供 embedding（相同文字給完全相同的向量來模擬 duplicate）
        emb = np.array([
            [1.0, 0.0],   # unique A
            [0.0, 1.0],   # duplicate (all same)
            [0.0, 1.0],   # duplicate
            [0.5, 0.5],   # unique B
            [0.0, 1.0],   # duplicate
        ], dtype=np.float32)
        deduped = data.deduplicate_train(eps=0.01, min_samples=2, embeddings=emb)
        # "duplicate" cluster → 保留 1 筆；unique A, B → 雜訊點保留
        assert len(deduped.train_prompt) == 3

    def test_test_val_unchanged(self):
        data = self._data_with_duplicates()
        emb = np.eye(len(data.train_prompt), dtype=np.float32)  # all unique → no clusters
        deduped = data.deduplicate_train(eps=0.01, embeddings=emb)
        assert deduped.test_prompt == data.test_prompt
        assert deduped.val_prompt == data.val_prompt

    def test_all_unique_keeps_all(self):
        data = self._data_with_duplicates()
        # 每筆 embedding 都不同 → DBSCAN 全部視為雜訊 → 全部保留
        emb = np.eye(len(data.train_prompt), dtype=np.float32)
        deduped = data.deduplicate_train(eps=0.01, embeddings=emb)
        assert len(deduped.train_prompt) == len(data.train_prompt)


# ── Router class imports (no heavy deps required) ────────────────────────────


class TestRouterImports:
    def test_knn_importable(self):
        from LLMRouter.router import KNNRouter
        assert KNNRouter is not None

    def test_mf_importable(self):
        from LLMRouter.router import MFRouter
        assert MFRouter is not None

    def test_sw_importable(self):
        from LLMRouter.router import SWRankingRouter
        assert SWRankingRouter is not None

    def test_roberta_importable(self):
        from LLMRouter.router import RoBERTaMLCRouter
        assert RoBERTaMLCRouter is not None

    def test_grpo_importable(self):
        from LLMRouter.router import GRPORouter
        assert GRPORouter is not None
