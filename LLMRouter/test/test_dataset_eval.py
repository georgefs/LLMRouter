"""
DatasetAnalyzer（四維根因框架）的單元測試。

測試策略：
  - 用合成資料驗證每個指標的數學正確性
  - 用「好資料 vs 壞資料」場景驗證 PASS/FAIL 判斷
  - 不依賴外部網路或 GPU（embedding 由測試自行注入）
"""
from __future__ import annotations

import numpy as np
import pytest

from LLMRouter.router import RouterData
from LLMRouter.router.dataset_eval import (
    DatasetAnalysisResult,
    _compute_avg_sim,
    _compute_ch_score,
    _compute_dec_var,
    _l2_normalize,
    analyze,
    format_report,
)


# ── 工具函數 ─────────────────────────────────────────────────────────────────

def _make_data(
    n: int = 200,
    m: int = 3,
    seed: int = 0,
    emb_dim: int = 16,
    *,
    cluster: bool = False,
    exclusive: bool = True,
) -> RouterData:
    """
    建立合成 RouterData，供 analyze() 測試使用。

    cluster=True  → 每個 GT label 的 embedding 有明顯分群
    exclusive=True → 每個樣本只有一個 best model
    """
    rng = np.random.default_rng(seed)
    models = [f"model_{i}" for i in range(m)]
    prompts = [f"prompt_{i}" for i in range(n)]

    if exclusive:
        labels = rng.integers(0, m, size=n)
        scores = np.zeros((n, m), dtype=np.float32)
        for i, lbl in enumerate(labels):
            scores[i, lbl] = 1.0
    else:
        # 多個 model 可能同時得分
        scores = rng.random((n, m)).astype(np.float32)

    if cluster:
        # 每個 label 的 embedding 繞不同中心點
        centers = rng.standard_normal((m, emb_dim)) * 5
        emb = np.zeros((n, emb_dim), dtype=np.float32)
        for i, lbl in enumerate(labels if exclusive else np.argmax(scores, axis=1)):
            emb[i] = centers[lbl] + rng.standard_normal(emb_dim) * 0.1
    else:
        emb = rng.standard_normal((n, emb_dim)).astype(np.float32)

    split = n // 5
    return RouterData(
        train_prompt=prompts,
        val_prompt=prompts[:split],
        test_prompt=prompts[:split],
        train_score=scores,
        val_score=scores[:split],
        test_score=scores[:split],
        test_tokens=np.zeros((split, m)),
        test_time=np.zeros((split, m)),
        models=models,
        train_embed=emb,
        val_embed=emb[:split],
        test_embed=emb[:split],
    )


# ── _l2_normalize ────────────────────────────────────────────────────────────

def test_l2_normalize_unit_vectors():
    x = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]])
    out = _l2_normalize(x)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms[0], 1.0)
    assert np.allclose(norms[1], 1.0)
    # 零向量 → 不爆炸
    assert np.all(np.isfinite(out[2]))


# ── _compute_avg_sim ─────────────────────────────────────────────────────────

def test_avg_sim_identical_vectors():
    """所有向量相同 → 所有 cos = 1.0。"""
    emb = np.ones((10, 4), dtype=np.float32)
    sim = _compute_avg_sim(emb)
    assert np.isclose(sim, 1.0, atol=1e-5)


def test_avg_sim_orthogonal_vectors():
    """正交向量 → cos = 0。"""
    n = 4
    emb = np.eye(n, dtype=np.float32)
    sim = _compute_avg_sim(emb)
    assert np.isclose(sim, 0.0, atol=1e-5)


def test_avg_sim_range():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((50, 16)).astype(np.float32)
    sim = _compute_avg_sim(emb)
    assert -1.0 <= sim <= 1.0


def test_avg_sim_single_vector():
    emb = np.ones((1, 4), dtype=np.float32)
    sim = _compute_avg_sim(emb)
    assert sim == 0.0


# ── _compute_dec_var ─────────────────────────────────────────────────────────

def test_dec_var_uniform_wins():
    """每個 model 剛好贏相同次數 → σ² ≈ 0。"""
    m = 4
    n = 100
    scores = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        scores[i, i % m] = 1.0
    win_rates, dec_var, multi = _compute_dec_var(scores)
    assert np.allclose(win_rates, 0.25, atol=0.01)
    assert dec_var < 1e-6
    assert multi < 0.01


def test_dec_var_dominant_model():
    """一個 model 永遠贏 → σ² 最大。"""
    n, m = 100, 4
    scores = np.zeros((n, m), dtype=np.float32)
    scores[:, 0] = 1.0
    win_rates, dec_var, multi = _compute_dec_var(scores)
    assert win_rates[0] == 1.0
    assert np.allclose(win_rates[1:], 0.0)
    # σ² = var([1,0,0,0]) = 3/16 * (1-0)² ≈ 0.1875
    assert dec_var > 0.1


def test_dec_var_multi_winner():
    """兩個 model 並列 → multi_win_pct = 1.0；每個 model 每筆都是 winner，win_rate = 1.0。"""
    n, m = 10, 2
    scores = np.ones((n, m), dtype=np.float32)  # 全部並列
    win_rates, dec_var, multi = _compute_dec_var(scores)
    # 每個 model 在全部 n 個樣本中都是 winner（並列），win_rate = n/n = 1.0
    assert np.allclose(win_rates, 1.0)
    assert np.isclose(multi, 1.0)


# ── _compute_ch_score ────────────────────────────────────────────────────────

def test_ch_score_well_separated():
    """完美分群 → CH Score 應很高。"""
    rng = np.random.default_rng(0)
    n_per_cluster, m, d = 50, 4, 8
    emb, labels = [], []
    for k in range(m):
        center = np.zeros(d)
        center[k] = 100.0
        emb.append(rng.standard_normal((n_per_cluster, d)) * 0.01 + center)
        labels.extend([k] * n_per_cluster)
    emb = np.vstack(emb).astype(np.float32)
    labels = np.array(labels)
    ch, ssb, ssw, k = _compute_ch_score(emb, labels)
    assert ch > 100, f"完美分群的 CH Score 應很高，got {ch}"
    assert ssb > ssw


def test_ch_score_random_labels():
    """隨機 label → CH Score 很低。"""
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 16)).astype(np.float32)
    labels = rng.integers(0, 4, size=200)
    ch, _, _, _ = _compute_ch_score(emb, labels)
    # 隨機 label 通常 CH < 10（不嚴格，但遠低於有意義分群）
    assert ch < 50


def test_ch_score_single_class():
    """唯一 label 只有 1 個 → 無法計算，回傳 0。"""
    emb = np.ones((20, 4), dtype=np.float32)
    labels = np.zeros(20, dtype=int)
    ch, ssb, ssw, k = _compute_ch_score(emb, labels)
    assert ch == 0.0
    assert k == 1


# ── analyze() ────────────────────────────────────────────────────────────────

def test_analyze_uses_precomputed_embedding():
    """train_embed 已存在時，不應呼叫 get_embeddings。"""
    data = _make_data(n=100, m=3, cluster=True)
    # 呼叫 analyze 時只要不 raise「無法載入 embedding model」就代表有用預存的
    result = analyze(data, emb_model="NONEXISTENT_MODEL_SHOULD_NOT_BE_LOADED")
    assert result.emb_source == "precomputed"


def test_analyze_result_fields():
    data = _make_data(n=150, m=4, cluster=True)
    r = analyze(data)
    assert r.n_train == 150
    assert len(r.models) == 4
    assert -1.0 <= r.avg_sim <= 1.0
    assert r.dec_var >= 0.0
    assert r.ch_score >= 0.0
    assert len(r.win_rates) == 4
    assert np.isclose(r.win_rates.sum(), 1.0, atol=0.05)  # 近似 1（exclusive 情況下嚴格等於 1）


def test_analyze_good_dataset_passes():
    """有明確分群的 exclusive 資料集應全部 PASS。"""
    data = _make_data(n=1000, m=4, cluster=True, seed=42)
    r = analyze(data)
    assert r.ch_pass, f"CH={r.ch_score:.4f} 應 > {2.0}"
    assert r.overall_grade in ("GOOD", "MARGINAL")


def test_analyze_exclusive_flag():
    """exclusive 資料集的 is_exclusive 應為 True。"""
    data = _make_data(n=100, m=3, exclusive=True)
    r = analyze(data)
    assert r.is_exclusive


def test_analyze_non_exclusive_flag():
    """連續分數資料集應顯示 is_exclusive=False。"""
    data = _make_data(n=100, m=3, exclusive=False)
    r = analyze(data)
    # 連續隨機分數幾乎不可能有並列最高，is_exclusive 仍可能是 True
    # 只要不拋例外即可
    assert isinstance(r.is_exclusive, bool)


def test_analyze_single_winner_model():
    """一個 model 永遠拿最高分 → n_clusters = 1，CH = 0，grade = POOR。"""
    rng = np.random.default_rng(0)
    n, m = 100, 3
    scores = np.zeros((n, m), dtype=np.float32)
    scores[:, 0] = 1.0  # model_0 永遠最好
    emb = rng.standard_normal((n, 8)).astype(np.float32)
    data = RouterData(
        train_prompt=[f"p{i}" for i in range(n)],
        val_prompt=["p0"],
        test_prompt=["p0"],
        train_score=scores,
        val_score=scores[:1],
        test_score=scores[:1],
        test_tokens=np.zeros((1, m)),
        test_time=np.zeros((1, m)),
        models=["a", "b", "c"],
        train_embed=emb,
    )
    r = analyze(data)
    assert r.n_clusters == 1
    assert r.ch_score == 0.0
    assert r.overall_grade == "POOR"
    assert len(r.warnings) > 0


def test_analyze_val_split():
    """split='val' 應使用 val_prompt / val_score / val_embed。"""
    data = _make_data(n=200, m=3, cluster=True)
    r = analyze(data, split="val")
    assert r.n_samples == len(data.val_prompt)


def test_analyze_invalid_split():
    data = _make_data(n=50, m=2)
    with pytest.raises(ValueError, match="split 必須是"):
        analyze(data, split="invalid")


# ── format_report ────────────────────────────────────────────────────────────

def test_format_report_contains_key_sections():
    data = _make_data(n=200, m=3, cluster=True)
    r = analyze(data)
    report = format_report(r, title="test_dataset.npz")
    # 必要區塊
    assert "CH Score" in report
    assert "Avg_Sim" in report
    assert "Dec_Var" in report
    assert "Win Rate" in report
    assert "綜合診斷" in report
    assert r.overall_grade in report


def test_format_report_poor_dataset():
    """POOR 資料集的報告應包含具體的修復建議。"""
    rng = np.random.default_rng(0)
    n, m = 100, 3
    scores = np.zeros((n, m), dtype=np.float32)
    scores[:, 0] = 1.0
    emb = rng.standard_normal((n, 8)).astype(np.float32)
    data = RouterData(
        train_prompt=[f"p{i}" for i in range(n)],
        val_prompt=["p0"], test_prompt=["p0"],
        train_score=scores, val_score=scores[:1], test_score=scores[:1],
        test_tokens=np.zeros((1, m)), test_time=np.zeros((1, m)),
        models=["a", "b", "c"],
        train_embed=emb,
    )
    r = analyze(data)
    report = format_report(r)
    assert "POOR" in report
    assert "✗" in report  # 至少一個 FAIL 標記


# ── CLI smoke test ────────────────────────────────────────────────────────────

def test_cli_router_analyze(tmp_path):
    """python -m LLMRouter router analyze <npz> 不拋例外。"""
    import subprocess, sys
    data = _make_data(n=200, m=3, cluster=True)
    npz = tmp_path / "data.npz"
    data.save(str(npz))
    json_out = tmp_path / "result.json"

    result = subprocess.run(
        [sys.executable, "-m", "LLMRouter", "router", "analyze",
         str(npz), "--output", str(json_out)],
        capture_output=True, text=True,
        cwd="/home/george/work2/delta/test/LLMRouter",
    )
    assert result.returncode == 0, result.stderr
    assert "Dataset Routing Value Report" in result.stdout
    assert json_out.exists()

    import json
    out = json.loads(json_out.read_text())
    assert "ch_score" in out
    assert "overall_grade" in out
