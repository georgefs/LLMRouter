"""
§4.3 評估指標單元測試

涵蓋：
  - compute_hit_rate   (HR)
  - compute_cost       (Cost)
  - compute_ter        (TER: Dominant / Inv / float)
  - compute_nbs        (NBS: 3×ΔHR + ΔCost_Savings)
  - model_unit_costs   (定價查表)
  - evaluate()         (回傳 hr 欄位)
  - evaluate_full()    (新增 cost / ter / nbs)
  - RouterBenchmark.table(show_cost_metrics=True)
  - RouterBenchmark.strongest_baseline()

執行: pytest LLMRouter/test/test_eval_metrics.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from LLMRouter.router.eval import (
    MODEL_PRICING,
    compute_cost,
    compute_hit_rate,
    compute_nbs,
    compute_ter,
    evaluate,
    evaluate_full,
    model_unit_costs,
)
from LLMRouter.router import RouterData, RouterBenchmark, OracleRouter, RandomRouter


# ── 工具函數 ─────────────────────────────────────────────────────────────────

def _probs_onehot(n: int, m: int, choices: list[int]) -> np.ndarray:
    """建立 one-hot probs；choices[i] 為第 i 筆選的 model index。"""
    p = np.zeros((n, m), dtype=float)
    for i, c in enumerate(choices):
        p[i, c] = 1.0
    return p


def _make_data(n: int = 50, m: int = 3, seed: int = 0) -> RouterData:
    rng = np.random.default_rng(seed)
    scores = rng.integers(0, 2, size=(n, m)).astype(float)
    tokens = rng.integers(100, 500, size=(n, m)).astype(float)
    times  = rng.random((n, m)).astype(float)
    prompts = [f"p{i}" for i in range(n)]
    return RouterData(
        train_prompt=prompts, val_prompt=prompts[:10], test_prompt=prompts,
        train_score=scores, val_score=scores[:10], test_score=scores,
        test_tokens=tokens, test_time=times,
        models=[f"model_{k}" for k in range(m)],
    )


# ── compute_hit_rate ─────────────────────────────────────────────────────────

class TestComputeHitRate:
    def test_perfect_oracle(self):
        """永遠選最高分 → HR = 1.0"""
        scores = np.array([[1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0]])
        probs = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, 1.0]])
        assert np.isclose(compute_hit_rate(probs, scores), 1.0)

    def test_always_wrong(self):
        """永遠選最低分（且 scores 各異）→ HR = 0.0"""
        scores = np.array([[0.0, 1.0],
                           [0.0, 1.0]])
        probs  = np.array([[1.0, 0.0],
                           [1.0, 0.0]])
        assert np.isclose(compute_hit_rate(probs, scores), 0.0)

    def test_partial_hits(self):
        """2/4 命中 → HR = 0.5"""
        scores = np.array([[1.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0],
                           [0.0, 1.0]])
        # always pick model 0
        probs = _probs_onehot(4, 2, [0, 0, 0, 0])
        # queries 0,1: hit; queries 2,3: miss
        assert np.isclose(compute_hit_rate(probs, scores), 0.5)

    def test_all_models_equal(self):
        """所有 model 分數相同 → 任何選法 HR = 1.0"""
        scores = np.ones((5, 3))
        probs  = _probs_onehot(5, 3, [0, 1, 2, 1, 0])
        assert np.isclose(compute_hit_rate(probs, scores), 1.0)

    def test_range(self):
        data = _make_data()
        probs = np.random.default_rng(1).random((50, 3))
        hr = compute_hit_rate(probs, data.test_score)
        assert 0.0 <= hr <= 1.0


# ── compute_cost ─────────────────────────────────────────────────────────────

class TestComputeCost:
    def test_uniform_cost_equals_avg_tokens(self):
        """model_costs=None → 回傳平均 token 數（= 論文 Cost 欄位）"""
        tokens = np.array([[100.0, 200.0],
                           [300.0, 400.0]])
        probs = _probs_onehot(2, 2, [0, 1])   # 選 100 和 400
        expected = (100.0 + 400.0) / 2
        assert np.isclose(compute_cost(probs, tokens), expected)

    def test_with_model_costs(self):
        """model_costs=[1.0, 2.0] → 第 2 個 model 的 token 乘以 2"""
        tokens = np.array([[100.0, 100.0],
                           [100.0, 100.0]])
        probs = _probs_onehot(2, 2, [0, 1])
        # query 0: 100×1=100; query 1: 100×2=200; avg=150
        assert np.isclose(compute_cost(probs, tokens, [1.0, 2.0]), 150.0)

    def test_all_same_model(self):
        """所有 query 都選同一 model → cost = 該 model 的平均 tokens"""
        tokens = np.array([[100.0, 999.0],
                           [200.0, 999.0],
                           [300.0, 999.0]])
        probs = _probs_onehot(3, 2, [0, 0, 0])
        assert np.isclose(compute_cost(probs, tokens), 200.0)

    def test_usd_cost(self):
        """model_costs 為無量綱乘數：傳入 price/1e6 即可得 USD"""
        tokens = np.array([[1_000_000.0, 0.0]])   # 1M tokens, model 0
        probs  = _probs_onehot(1, 2, [0])
        # 0.5 $/1M tokens → unit multiplier = 0.5/1_000_000
        unit = 0.5 / 1_000_000
        assert np.isclose(compute_cost(probs, tokens, [unit, unit]), 0.5)


# ── compute_ter ──────────────────────────────────────────────────────────────

class TestComputeTer:
    def test_dominant_both_improved(self):
        assert compute_ter(0.92, 190.0, 0.90, 200.0) == "Dominant"

    def test_dominant_equal_metrics(self):
        assert compute_ter(0.90, 200.0, 0.90, 200.0) == "Dominant"

    def test_dominant_hr_improved_cost_equal(self):
        assert compute_ter(0.92, 200.0, 0.90, 200.0) == "Dominant"

    def test_inv_both_worsened(self):
        assert compute_ter(0.80, 210.0, 0.90, 200.0) == "Inv"

    def test_valid_trade_off(self):
        """HR 下降 10%，Cost 節省 50% → TER = 5.0"""
        # baseline: HR=1.0, cost=200; router: HR=0.9, cost=100
        # hr_sacrifice = (1.0-0.9)/1.0*100 = 10%
        # cost_savings = (200-100)/200*100 = 50%
        # TER = 50/10 = 5.0
        result = compute_ter(0.9, 100.0, 1.0, 200.0)
        assert isinstance(result, float)
        assert np.isclose(result, 5.0)

    def test_ter_positive_means_worth_it(self):
        """TER > 0 代表每 1% HR 犧牲可節省正的 Cost"""
        result = compute_ter(0.85, 150.0, 0.90, 200.0)
        assert isinstance(result, float)
        assert result > 0

    def test_ter_negative_inv_boundary(self):
        """Cost 上升但 HR 下降 → Inv"""
        result = compute_ter(0.85, 220.0, 0.90, 200.0)
        assert result == "Inv"


# ── compute_nbs ──────────────────────────────────────────────────────────────

class TestComputeNbs:
    def test_nbs_dominant_positive(self):
        """HR 提升 1%，Cost 節省 5% → NBS = 3*1 + 5 = +8"""
        nbs = compute_nbs(0.91, 190.0, 0.90, 200.0)
        # ΔHR = (0.91-0.90)*100 = 1.0%
        # cost_savings = (200-190)/200*100 = 5.0%
        # NBS = 3*1 + 5 = 8
        assert np.isclose(nbs, 8.0, atol=0.01)

    def test_nbs_always_cheapest_hacks_metric(self):
        """
        若 router 把所有 query 都送最便宜 model（HR 大降），
        NBS 應該是嚴重負值，反映 3:1 懲罰。
        """
        # baseline HR=0.90, cost=200; hacked: HR=0.50, cost=50
        # ΔHR = -40%; cost_savings = 75%
        # NBS = 3*(-40) + 75 = -45
        nbs = compute_nbs(0.50, 50.0, 0.90, 200.0)
        assert nbs < 0, f"NBS 應為負值，得 {nbs}"

    def test_nbs_zero_baseline_cost(self):
        """baseline_cost=0 不應 raise；cost_savings 回傳 0"""
        nbs = compute_nbs(0.90, 0.0, 0.90, 0.0)
        assert np.isfinite(nbs)

    def test_nbs_same_as_baseline(self):
        """完全相同 → NBS = 0"""
        nbs = compute_nbs(0.90, 200.0, 0.90, 200.0)
        assert np.isclose(nbs, 0.0)

    def test_nbs_weighting_3to1(self):
        """驗證 3:1 比例：HR 損失 1% ≡ Cost 損失 3%"""
        # router A：只節省成本，HR 損失 1%
        nbs_a = compute_nbs(0.89, 150.0, 0.90, 200.0)
        # router B：完全不節省成本，HR 也損失更多
        nbs_b = compute_nbs(0.87, 200.0, 0.90, 200.0)
        # 驗證 nbs_a 好於 nbs_b（同 HR 損失，nbs_a 有更多 cost_savings）
        assert nbs_a > nbs_b


# ── model_unit_costs ─────────────────────────────────────────────────────────

class TestModelUnitCosts:
    def test_known_models(self):
        names = ["llama-4-maverick", "microsoft-phi-4"]
        costs = model_unit_costs(names)
        assert costs.shape == (2,)
        # maverick 比 phi-4 貴
        assert costs[0] > costs[1]

    def test_unknown_model_returns_zero(self):
        costs = model_unit_costs(["totally-unknown-model-xyz"])
        assert costs[0] == 0.0

    def test_partial_match(self):
        """部分匹配：'mistral-small' 應能查到定價"""
        costs = model_unit_costs(["mistral-small"])
        assert costs[0] > 0.0

    def test_all_report_models(self):
        """Technical Report §4.2.4 六個模型都應可查到定價"""
        names = [
            "llama-4-maverick-17b",
            "llama-4-scout-17b",
            "gpt-oss-20b",
            "mistral-small-3.1",
            "google-gemma-3-27b",
            "microsoft-phi-4",
        ]
        costs = model_unit_costs(names)
        assert all(c > 0 for c in costs), f"有 model 定價為 0: {list(zip(names, costs))}"


# ── evaluate() ───────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_hr_field(self):
        data = _make_data()
        probs = np.random.default_rng(7).random((50, 3))
        result = evaluate(probs, data)
        assert "hr" in result
        assert 0.0 <= result["hr"] <= 1.0

    def test_oracle_hr_is_one(self):
        """Oracle probs（完美知道最佳 model）→ HR = 1.0"""
        rng = np.random.default_rng(0)
        scores = rng.integers(0, 2, size=(20, 3)).astype(float)
        # Make sure each row has a unique best
        for i in range(20):
            scores[i] = 0.0
            scores[i, i % 3] = 1.0
        data = RouterData(
            train_prompt=["p"] * 20, val_prompt=["p"], test_prompt=["p"] * 20,
            train_score=scores, val_score=scores[:1], test_score=scores,
            test_tokens=np.ones((20, 3)) * 100, test_time=np.ones((20, 3)) * 0.1,
            models=["a", "b", "c"],
        )
        # oracle probs = scores
        result = evaluate(scores, data)
        assert np.isclose(result["hr"], 1.0)

    def test_backward_compat_keys(self):
        """舊版鍵仍在（不破壞既有程式碼）"""
        data = _make_data()
        probs = np.ones((50, 3)) / 3
        result = evaluate(probs, data)
        for key in ("mu", "vb", "ep", "avg_tokens", "avg_latency"):
            assert key in result, f"missing key: {key}"


# ── evaluate_full() ──────────────────────────────────────────────────────────

class TestEvaluateFull:
    def test_cost_included_when_tokens_exist(self):
        data = _make_data()
        probs = np.random.default_rng(3).random((50, 3))
        result = evaluate_full(probs, data)
        assert "cost" in result
        assert result["cost"] > 0

    def test_ter_nbs_when_baseline_provided(self):
        data = _make_data()
        probs = np.random.default_rng(4).random((50, 3))
        result = evaluate_full(probs, data, baseline_hr=0.9, baseline_cost=300.0)
        assert "ter" in result
        assert "nbs" in result

    def test_ter_nbs_absent_without_baseline(self):
        data = _make_data()
        probs = np.random.default_rng(5).random((50, 3))
        result = evaluate_full(probs, data)
        assert "ter" not in result
        assert "nbs" not in result

    def test_with_model_costs_usd(self):
        """傳入 unit_cost 後 cost 應為 USD 量級（極小值）"""
        data = _make_data()
        probs = np.ones((50, 3)) / 3
        mc = model_unit_costs(["llama-4-maverick", "gpt-oss-20b", "phi-4"]) / 1_000_000
        result = evaluate_full(probs, data, model_costs=mc)
        assert result["cost"] < 1.0, f"Cost 應為 USD 量級，得 {result['cost']}"


# ── RouterBenchmark ──────────────────────────────────────────────────────────

class TestRouterBenchmark:
    def test_show_cost_metrics_table(self):
        data = _make_data(n=30, m=2)
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        bench.run(RandomRouter, label="random")
        table = bench.table(show_cost_metrics=True)
        assert "HR" in table
        assert "Cost" in table
        assert "NBS" in table

    def test_standard_table_has_hr_column(self):
        data = _make_data(n=30, m=2)
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        table = bench.table()
        assert "HR" in table

    def test_strongest_baseline_has_highest_hr(self):
        data = _make_data(n=30, m=2)
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        bench.run(RandomRouter, label="random")
        best = bench.strongest_baseline()
        assert best is not None
        assert best.label == "oracle"

    def test_results_have_hr(self):
        data = _make_data(n=20, m=2)
        bench = RouterBenchmark(data)
        bench.run(OracleRouter, label="oracle")
        for r in bench.results():
            assert 0.0 <= r.hr <= 1.0

    def test_with_baseline_ter_nbs(self):
        """傳入 baseline_hr / baseline_cost → RunResult 帶有 ter / nbs"""
        data = _make_data(n=30, m=2)
        bench = RouterBenchmark(data)
        bench.run(
            RandomRouter, label="random",
            baseline_hr=0.9, baseline_cost=300.0,
        )
        r = bench.results()[0]
        assert r.ter is not None
        assert r.nbs is not None

    def test_empty_benchmark(self):
        data = _make_data()
        bench = RouterBenchmark(data)
        assert bench.table() == "(無結果)"
        assert bench.strongest_baseline() is None
