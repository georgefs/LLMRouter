"""
Dataset Routing Value Analyzer
================================
實作 Technical Report Section 7 的四維根因分析框架（Four-Dimensional Root-Cause Framework），
量化一個 RouterData 資料集是否適合作為 routing 訓練資料。

四個指標：
  1. N          — 樣本數（統計基線）
  2. Avg_Sim    — 語意一致性（平均成對 cosine 相似度）
  3. Dec_Var    — 鑑別增益（各模型 win rate 的變異數 σ²）
  4. CH Score   — 標籤-特徵對齊度（Calinski-Harabasz Index）

推薦閾值（Technical Report Section 8）：
  CH Score > 2.0   (Priority 1：最關鍵，決定 learnability)
  Avg_Sim  > 0.025 (Priority 2：特徵空間的結構性)
  σ²       > 0.015 (Priority 3：模型能力差距的鑑別性)

使用方式（CLI）：
  python -m LLMRouter router analyze data.npz
  python -m LLMRouter router analyze data.npz --emb-model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .data import RouterData

# ── 閾值 ────────────────────────────────────────────────────────────────────────
_THRESHOLD_CH       = 2.0
_THRESHOLD_AVG_SIM  = 0.025
_THRESHOLD_DEC_VAR  = 0.015
_N_ADEQUATE         = 3_000  # 建議最低樣本數


@dataclass
class DatasetAnalysisResult:
    """四維分析結果容器。"""

    # ── 基本資訊 ──────────────────────────────────────────────────────────────
    n_train: int
    n_val:   int
    n_test:  int
    models: List[str]

    # ── 四維指標 ──────────────────────────────────────────────────────────────
    n_samples:  int            # = n_train
    avg_sim:    float          # Semantic Coherence
    dec_var:    float          # Discriminative Gain (σ²)
    ch_score:   float          # Calinski-Harabasz Index

    # ── 輔助資訊 ──────────────────────────────────────────────────────────────
    win_rates:     np.ndarray  # (M,) — per-model exclusive win rate
    model_acc:     np.ndarray  # (M,) — per-model average score (可能非 exclusive)
    is_exclusive:  bool        # True = 每個樣本只有一個 best model
    multi_win_pct: float       # 有多個 best model 的樣本比例

    # ── CH 分解 ──────────────────────────────────────────────────────────────
    n_clusters:  int           # GT label 的唯一類別數 (K)
    ssb:         float         # Between-cluster variance
    ssw:         float         # Within-cluster variance

    # ── 狀態 ──────────────────────────────────────────────────────────────────
    emb_source: str            # "precomputed" | "computed:<model>"

    # ── 診斷 ──────────────────────────────────────────────────────────────────
    warnings: List[str] = field(default_factory=list)

    # ────────────────────────────────────────────────────────────────────────
    # 衍生屬性：Pass / Fail
    # ────────────────────────────────────────────────────────────────────────
    @property
    def ch_pass(self)      -> bool: return self.ch_score  >= _THRESHOLD_CH
    @property
    def sim_pass(self)     -> bool: return self.avg_sim   >= _THRESHOLD_AVG_SIM
    @property
    def var_pass(self)     -> bool: return self.dec_var   >= _THRESHOLD_DEC_VAR
    @property
    def n_adequate(self)   -> bool: return self.n_samples >= _N_ADEQUATE

    @property
    def overall_grade(self) -> str:
        """GOOD / MARGINAL / POOR — 依 Technical Report 優先順序判斷。"""
        if not self.ch_pass:
            return "POOR"
        if not self.sim_pass:
            return "MARGINAL"
        if not self.var_pass:
            return "MARGINAL"
        return "GOOD"


# ── 核心計算函數 ────────────────────────────────────────────────────────────────

def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-9)


def _compute_avg_sim(emb: np.ndarray) -> float:
    """
    平均成對 cosine 相似度 = 1/(N*(N-1)) * Σ_{i≠j} cos(v_i, v_j)

    用矩陣乘法實作，時間複雜度 O(N²D)，空間 O(N²)。
    N=3500, D=384 時 Gram matrix ≈ 47 MB，可接受。
    """
    n = len(emb)
    if n < 2:
        return 0.0
    en = _l2_normalize(emb)
    # Gram matrix（已 L2 normalize，dot product = cosine similarity）
    g = en @ en.T          # (N, N)
    # 去除對角線自身相似度 1.0
    total = g.sum() - np.trace(g)
    return float(total / (n * (n - 1)))


def _compute_dec_var(train_score: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    σ² = 1/M * Σ(Acc_k - mean_Acc)²

    Acc_k = 樣本中 model k 是 best（最高分）的比例。
    若多個 model 並列最高，各自都計一次 win。

    回傳 (win_rates, dec_var, multi_win_pct)
    """
    n, m = train_score.shape
    max_per_row = train_score.max(axis=1, keepdims=True)      # (N, 1)
    is_winner   = (train_score >= max_per_row - 1e-9)         # (N, M) bool

    win_counts  = is_winner.sum(axis=0).astype(float)         # (M,) — 可能有小數
    win_rates   = win_counts / n                               # (M,)
    dec_var     = float(np.var(win_rates))

    multi_win   = (is_winner.sum(axis=1) > 1).sum()
    multi_win_pct = float(multi_win / n) if n > 0 else 0.0

    return win_rates, dec_var, multi_win_pct


def _compute_ch_score(emb: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, int]:
    """
    Calinski-Harabasz Index，使用 sklearn。

    回傳 (ch_score, SSB, SSW, K)，若唯一 label 只有 1 個則回傳 (0, 0, 0, 1)。
    """
    from sklearn.metrics import calinski_harabasz_score

    unique_labels = np.unique(labels)
    k = len(unique_labels)
    if k < 2:
        return 0.0, 0.0, 0.0, k

    en = _l2_normalize(emb)
    ch = float(calinski_harabasz_score(en, labels))

    # 手動計算 SSB / SSW（用於報告細節）
    n = len(en)
    c_global = en.mean(axis=0)
    ssb = 0.0
    ssw = 0.0
    for lbl in unique_labels:
        idx = np.where(labels == lbl)[0]
        c_k = en[idx].mean(axis=0)
        ssb += len(idx) * float(np.sum((c_k - c_global) ** 2))
        ssw += float(np.sum((en[idx] - c_k) ** 2))

    return ch, ssb, ssw, k


# ── 主要分析器 ─────────────────────────────────────────────────────────────────

def analyze(
    data: RouterData,
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    emb_batch_size: int = 32,
    split: str = "train",
) -> DatasetAnalysisResult:
    """
    執行四維根因分析。

    Args:
        data:          RouterData 物件
        emb_model:     若資料集未預存 embedding，用此模型即時計算
        emb_batch_size: embedding 計算 batch size
        split:         分析哪個 split（"train" / "val" / "test"，預設 "train"）

    Returns:
        DatasetAnalysisResult
    """
    warnings: List[str] = []

    # ── 選取 split ────────────────────────────────────────────────────────────
    if split == "train":
        prompts = data.train_prompt
        scores  = data.train_score
        embeds  = data.train_embed
    elif split == "val":
        prompts = data.val_prompt
        scores  = data.val_score
        embeds  = data.val_embed
    elif split == "test":
        prompts = data.test_prompt
        scores  = data.test_score
        embeds  = data.test_embed
    else:
        raise ValueError(f"split 必須是 'train'、'val' 或 'test'，got '{split}'")

    n = len(prompts)

    # ── 取得 / 計算 embedding ─────────────────────────────────────────────────
    if embeds is not None:
        emb = embeds.astype(np.float32)
        emb_source = "precomputed"
    else:
        print(f"[dataset_eval] 計算 {split} set embedding（{n} 筆，model={emb_model}）...")
        from ._embeddings import get_embeddings
        emb = get_embeddings(prompts, emb_model, emb_batch_size)
        emb_source = f"computed:{emb_model}"

    # ── Metric 2: Avg_Sim ────────────────────────────────────────────────────
    avg_sim = _compute_avg_sim(emb)

    # ── Metric 3: Dec_Var ────────────────────────────────────────────────────
    win_rates, dec_var, multi_win_pct = _compute_dec_var(scores)
    model_acc = scores.mean(axis=0)
    is_exclusive = multi_win_pct < 0.01  # < 1% 視為 exclusive

    # ── Metric 4: CH Score ────────────────────────────────────────────────────
    gt_labels = np.argmax(scores, axis=1)  # best model index per sample
    ch_score, ssb, ssw, n_clusters = _compute_ch_score(emb, gt_labels)

    if n_clusters < 2:
        warnings.append(
            f"GT labels 的唯一類別數不足（K={n_clusters}），"
            "所有樣本都被同一個 model 拿下，CH Score 無法計算（設為 0）。"
        )

    # ── 樣本數警告 ───────────────────────────────────────────────────────────
    if n < _N_ADEQUATE:
        warnings.append(
            f"樣本數 N={n} 低於建議下限 {_N_ADEQUATE}，"
            "可能導致 router 過擬合或無法泛化。"
        )

    return DatasetAnalysisResult(
        n_train=len(data.train_prompt),
        n_val=len(data.val_prompt),
        n_test=len(data.test_prompt),
        models=data.models,
        n_samples=n,
        avg_sim=avg_sim,
        dec_var=dec_var,
        ch_score=ch_score,
        win_rates=win_rates,
        model_acc=model_acc,
        is_exclusive=is_exclusive,
        multi_win_pct=multi_win_pct,
        n_clusters=n_clusters,
        ssb=ssb,
        ssw=ssw,
        emb_source=emb_source,
        warnings=warnings,
    )


# ── 報告格式化 ─────────────────────────────────────────────────────────────────

def _status(passed: bool) -> str:
    return "✓ PASS" if passed else "✗ FAIL"


def _grade_label(grade: str) -> str:
    labels = {
        "GOOD":     "GOOD     — 具備高路由價值，適合訓練 SFT+GRPO",
        "MARGINAL": "MARGINAL — 路由訊號偏弱，訓練可能不穩定",
        "POOR":     "POOR     — 路由價值低，訓練預期失敗",
    }
    return labels.get(grade, grade)


def format_report(result: DatasetAnalysisResult, title: str = "") -> str:
    lines: List[str] = []
    W = 64
    sep = "─" * W

    def h(text: str):
        lines.append(f"\n── {text} {'─' * max(0, W - len(text) - 4)}")

    lines.append("=" * W)
    if title:
        lines.append(f" {title}")
    lines.append(" Dataset Routing Value Report（技術報告 §7 四維框架）")
    lines.append("=" * W)

    # ── 基本資訊 ──────────────────────────────────────────────────────────────
    h("基本資訊")
    lines.append(f"  模型池 ({len(result.models)} 個)：{', '.join(result.models)}")
    lines.append(
        f"  資料分割：train={result.n_train}  val={result.n_val}  test={result.n_test}"
    )
    lines.append(
        f"  分析 split：train（N={result.n_samples}）  "
        f"embedding 來源：{result.emb_source}"
    )
    ds_type = "Exclusive" if result.is_exclusive else f"Original（{result.multi_win_pct:.1%} 多贏家）"
    lines.append(f"  資料集類型：{ds_type}")

    # ── 四維指標 ──────────────────────────────────────────────────────────────
    h("四維指標（優先順序由高到低）")
    lines.append(
        f"  {'指標':<20} {'數值':>10}  {'閾值':>10}  {'狀態'}"
    )
    lines.append(f"  {sep[:60]}")
    lines.append(
        f"  {'CH Score（P1）':<20} {result.ch_score:>10.4f}  {f'> {_THRESHOLD_CH}':>10}  "
        f"{_status(result.ch_pass)}"
    )
    lines.append(
        f"  {'Avg_Sim（P2）':<20} {result.avg_sim:>10.4f}  {f'> {_THRESHOLD_AVG_SIM}':>10}  "
        f"{_status(result.sim_pass)}"
    )
    lines.append(
        f"  {'Dec_Var σ²（P3）':<20} {result.dec_var:>10.4f}  {f'> {_THRESHOLD_DEC_VAR}':>10}  "
        f"{_status(result.var_pass)}"
    )
    lines.append(
        f"  {'N 樣本數':<20} {result.n_samples:>10,}  {f'≥ {_N_ADEQUATE:,}':>10}  "
        f"{'✓ 足夠' if result.n_adequate else '⚠ 偏少'}"
    )

    # ── CH 分解 ──────────────────────────────────────────────────────────────
    h("CH Score 分解（標籤-特徵對齊）")
    lines.append(f"  K（GT label 類別數） = {result.n_clusters}")
    lines.append(f"  SSB（between-cluster） = {result.ssb:,.2f}"
                 "  ← 越大代表不同專家語意疆域越分離")
    lines.append(f"  SSW（within-cluster）  = {result.ssw:,.2f}"
                 "  ← 越小代表同一專家的 prompt 越緊湊")
    if result.ssw > 0:
        ratio = result.ssb / result.ssw
        lines.append(f"  SSB/SSW 比例 = {ratio:.4f}")

    # ── Win Rates ─────────────────────────────────────────────────────────────
    h("各模型 Win Rate（訓練集）")
    lines.append(f"  {'模型':<45} {'Win Rate':>9}  {'Avg Score':>9}")
    lines.append(f"  {sep[:60]}")
    order = np.argsort(result.win_rates)[::-1]
    for idx in order:
        lines.append(
            f"  {result.models[idx]:<45} {result.win_rates[idx]:>9.2%}  "
            f"{result.model_acc[idx]:>9.4f}"
        )
    lines.append(f"\n  Dec_Var σ² = {result.dec_var:.4f}  "
                 f"（平均 win rate = {result.win_rates.mean():.2%}）")

    # ── 綜合診斷 ──────────────────────────────────────────────────────────────
    h("綜合診斷與建議（Technical Report §8）")

    grade = result.overall_grade
    lines.append(f"\n  綜合評分：{_grade_label(grade)}\n")

    # Priority 1: CH
    if result.ch_pass:
        lines.append(f"  ✓ [P1] CH Score = {result.ch_score:.4f} > {_THRESHOLD_CH}")
        lines.append("       GT labels 在 embedding 空間形成可分離的 cluster，routing 可行。")
    else:
        lines.append(f"  ✗ [P1] CH Score = {result.ch_score:.4f} ≤ {_THRESHOLD_CH}  ← 最嚴重問題")
        lines.append("       GT labels 與文字特徵幾乎無對應，routing 邊界無法學習。")
        lines.append("       建議：（1）換更敏感的 embedding 模型重新計算，")
        lines.append("              （2）合併能力重疊的 GT 類別降低 K，")
        lines.append("              （3）加入 domain-specific 篩選。")

    # Priority 2: Avg_Sim
    if result.sim_pass:
        lines.append(f"\n  ✓ [P2] Avg_Sim = {result.avg_sim:.4f} > {_THRESHOLD_AVG_SIM}")
        lines.append("       特徵空間具備足夠的語意結構。")
    else:
        lines.append(f"\n  ✗ [P2] Avg_Sim = {result.avg_sim:.4f} ≤ {_THRESHOLD_AVG_SIM}")
        lines.append("       Prompt 語意過於分散（hairball），特徵無法形成穩定 routing trigger。")
        lines.append("       建議：縮小 domain 範圍，或增加同一類別的 prompt density。")

    # Priority 3: Dec_Var
    if result.var_pass:
        lines.append(f"\n  ✓ [P3] Dec_Var σ² = {result.dec_var:.4f} > {_THRESHOLD_DEC_VAR}")
        lines.append("       模型間能力差距顯著，routing 的 ROI 足夠。")
    else:
        lines.append(f"\n  ✗ [P3] Dec_Var σ² = {result.dec_var:.4f} ≤ {_THRESHOLD_DEC_VAR}")
        lines.append("       所有模型 win rate 過於平均，RL 獎勵訊號將過於平坦。")
        lines.append("       建議：剔除能力重疊的模型，保留能力差距最大的子集。")

    # N
    if not result.n_adequate:
        lines.append(
            f"\n  ⚠ [N]  樣本數 {result.n_samples:,} < {_N_ADEQUATE:,}，"
            "建議增加資料量以確保 SFT 泛化能力。"
        )

    # 其他警告
    if result.warnings:
        lines.append("\n  其他警告：")
        for w in result.warnings:
            lines.append(f"    • {w}")

    lines.append("\n" + "=" * W)
    return "\n".join(lines)
