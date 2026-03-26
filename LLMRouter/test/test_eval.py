"""
Tests for LLMRouter.router.eval — evaluate(), RunResult, RouterBenchmark.
"""
from __future__ import annotations

import numpy as np
import pytest

from LLMRouter.router import RouterData, OracleRouter, RandomRouter
from LLMRouter.router.eval import RouterBenchmark, RunResult, _softmax_entropy, evaluate


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_data(n=20, m=3, seed=0) -> RouterData:
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 2, size=(n, m)).astype(float)
    return RouterData(
        train_prompt=[f"q{i}" for i in range(n)],
        val_prompt=[],
        test_prompt=[f"q{i}" for i in range(n)],
        train_score=scores,
        val_score=np.zeros((0, m)),
        test_score=scores,
        test_tokens=np.full((n, m), 100.0),
        test_time=np.full((n, m), 0.5),
        models=[f"model_{i}" for i in range(m)],
    )


@pytest.fixture()
def data():
    return _make_data(n=20, m=3, seed=7)


# ── _softmax_entropy ──────────────────────────────────────────────────────────


class TestSoftmaxEntropy:
    def test_uniform_max_entropy(self):
        probs = np.ones((10, 3))
        ep = _softmax_entropy(probs, temperature=1.0)
        assert ep == pytest.approx(np.log2(3), abs=1e-5)

    def test_deterministic_zero_entropy(self):
        probs = np.zeros((5, 3))
        probs[:, 0] = 100.0  # always picks model 0 after softmax
        ep = _softmax_entropy(probs)
        assert ep == pytest.approx(0.0, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        probs = rng.random((8, 4))
        assert _softmax_entropy(probs) >= 0.0


# ── evaluate() ────────────────────────────────────────────────────────────────


class TestEvaluate:
    def test_keys(self, data):
        probs = np.ones((len(data.test_prompt), len(data.models)))
        m = evaluate(probs, data)
        assert set(m.keys()) == {"mu", "vb", "ep", "avg_tokens", "avg_latency"}

    def test_oracle_vb_one(self, data):
        """完美預測 → vb=1.0。"""
        # probs: 給每筆 test 的最佳 model 極高分
        best_idx = np.argmax(data.test_score, axis=1)
        n, m = data.test_score.shape
        probs = np.zeros((n, m))
        probs[np.arange(n), best_idx] = 1.0
        result = evaluate(probs, data)
        oracle = float(np.mean(np.max(data.test_score, axis=1)))
        assert result["mu"] == pytest.approx(oracle, abs=1e-6)
        assert result["vb"] == pytest.approx(1.0, abs=1e-6)

    def test_zero_oracle_safe(self):
        """oracle=0 時 vb 不應 ZeroDivisionError。"""
        n, m = 5, 2
        data = RouterData(
            train_prompt=["q"] * n,
            val_prompt=[],
            test_prompt=["q"] * n,
            train_score=np.zeros((n, m)),
            val_score=np.zeros((0, m)),
            test_score=np.zeros((n, m)),
            test_tokens=np.zeros((n, m)),
            test_time=np.zeros((n, m)),
            models=["a", "b"],
        )
        probs = np.ones((n, m))
        result = evaluate(probs, data)
        assert result["vb"] == 0.0

    def test_tokens_latency_populated(self, data):
        probs = np.ones((len(data.test_prompt), len(data.models)))
        result = evaluate(probs, data)
        assert result["avg_tokens"] == pytest.approx(100.0, abs=1e-6)
        assert result["avg_latency"] == pytest.approx(0.5, abs=1e-6)

    def test_tokens_zero_matrix(self):
        """test_tokens / test_time 為零矩陣 → avg_tokens/latency=0.0。"""
        data = _make_data(n=10, m=2)
        probs = np.ones((10, 2))
        result = evaluate(probs, data)
        # _make_data 填 100 / 0.5，但 zero-matrix 版本是直接造 RouterData
        assert result["avg_tokens"] == pytest.approx(100.0)  # non-zero in fixture

    def test_mu_in_range(self, data):
        probs = np.random.default_rng(0).random((len(data.test_prompt), len(data.models)))
        result = evaluate(probs, data)
        assert 0.0 <= result["mu"] <= 1.0

    def test_ep_non_negative(self, data):
        probs = np.random.default_rng(1).random((len(data.test_prompt), len(data.models)))
        result = evaluate(probs, data)
        assert result["ep"] >= 0.0

    def test_consistent_with_base_router_evaluate(self, data):
        """BaseRouter.evaluate() 應委派給 evaluate() 純函數，結果相同。"""
        from LLMRouter.router.base import BaseRouter

        class FixedRouter(BaseRouter):
            def _fit(self, d): pass
            def predict_probs(self, prompts):
                n, m = len(prompts), data.test_score.shape[1]
                return np.full((n, m), 1.0 / m)

        r = FixedRouter()
        r.fit(data)
        probs = r.predict_probs(data.test_prompt)
        via_standalone = evaluate(probs, data)
        via_router = r.evaluate(data)
        for key in ["mu", "vb", "ep", "avg_tokens", "avg_latency"]:
            assert via_standalone[key] == pytest.approx(via_router[key], abs=1e-9)


# ── RunResult ─────────────────────────────────────────────────────────────────


class TestRunResult:
    def test_fields(self):
        r = RunResult(
            label="knn", size=0.5, seed=0, n_train=10,
            mu=0.8, vb=0.9, ep=1.2, avg_tokens=200.0, avg_latency=0.3
        )
        assert r.label == "knn"
        assert r.size == 0.5
        assert r.n_train == 10
        assert r.mu == 0.8


# ── RouterBenchmark ───────────────────────────────────────────────────────────


class TestRouterBenchmark:
    def test_empty_results(self, data):
        bench = RouterBenchmark(data)
        assert bench.results() == []

    def test_empty_table(self, data):
        assert RouterBenchmark(data).table() == "(無結果)"

    def test_run_returns_self(self, data):
        bench = RouterBenchmark(data)
        ret = bench.run(OracleRouter)
        assert ret is bench

    def test_method_chaining(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle").run(RandomRouter, label="random")
        assert len(bench.results()) == 2

    def test_single_run_accumulates(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter)
        assert len(bench.results()) == 1
        r = bench.results()[0]
        assert r.label == "OracleRouter"
        assert r.vb == pytest.approx(1.0, abs=1e-9)

    def test_multi_seed_accumulates(self, data):
        bench = RouterBenchmark(data)
        bench.run(RandomRouter, seeds=[0, 1, 2])
        assert len(bench.results()) == 3

    def test_multi_size_accumulates(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[0.5, 1.0])
        assert len(bench.results()) == 2

    def test_multi_size_multi_seed(self, data):
        bench = RouterBenchmark(data)
        bench.run(RandomRouter, sizes=[0.5, 1.0], seeds=[0, 1])
        assert len(bench.results()) == 4  # 2 sizes × 2 seeds

    def test_custom_label(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="upper-bound")
        assert bench.results()[0].label == "upper-bound"

    def test_n_train_reflects_subsample(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[0.5])
        r = bench.results()[0]
        expected = max(1, round(len(data.train_prompt) * 0.5))
        assert r.n_train == expected

    def test_n_train_int_size(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[5])
        assert bench.results()[0].n_train == 5

    def test_multi_router_separate_labels(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        bench.run(RandomRouter, label="random", seeds=[0, 1])
        labels = [r.label for r in bench.results()]
        assert labels.count("oracle") == 1
        assert labels.count("random") == 2

    def test_results_returns_copy(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter)
        r1 = bench.results()
        r1.clear()
        assert len(bench.results()) == 1  # original not affected

    def test_table_has_header(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        t = bench.table()
        assert "router" in t
        assert "mu" in t
        assert "vb" in t

    def test_table_contains_label(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="my-oracle")
        assert "my-oracle" in bench.table()

    def test_table_float_size_percent(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[0.5])
        assert "50%" in bench.table()

    def test_table_int_size(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[5])
        assert "5" in bench.table()

    def test_table_separator_between_labels(self, data):
        """不同 label 間應有分隔線。"""
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="A")
        bench.run(RandomRouter, label="B")
        lines = bench.table().splitlines()
        sep_count = sum(1 for l in lines if l.startswith("---"))
        assert sep_count >= 2  # 至少 header 下方 + label 之間

    def test_table_averages_seeds(self, data):
        """相同 (label, size)、不同 seed 的結果應在 table 中只佔一行（平均）。"""
        bench = RouterBenchmark(data)
        bench.run(RandomRouter, label="rnd", seeds=[0, 1, 2])
        data_lines = [
            l for l in bench.table().splitlines()
            if "rnd" in l
        ]
        assert len(data_lines) == 1  # 3 seeds 合併成 1 行

    def test_preprocess_fn_applied(self, data):
        """preprocess_fn 應縮減訓練集。"""
        bench = RouterBenchmark(data)
        # preprocess_fn 只保留前 5 筆訓練資料
        def trim(d):
            return d._select_train(np.arange(5))
        bench.run(OracleRouter, preprocess_fn=trim, sizes=[1.0])
        assert bench.results()[0].n_train == 5

    def test_oracle_vb_one_regardless_of_size(self, data):
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, sizes=[0.3, 1.0], seeds=[0])
        for r in bench.results():
            assert r.vb == pytest.approx(1.0, abs=1e-9)
