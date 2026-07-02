from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from .data import RouterData


# ── 核心指標函數（舊版，保持向下相容） ────────────────────────────────────────


def _softmax_entropy(probs: np.ndarray) -> float:
    """
    Softmax entropy（bits/sample），與 RouterEval3 計算方式一致。

    Args:
        probs: (N, M) 分數矩陣（不必是機率）
    """
    p = np.exp(probs - np.max(probs, axis=1, keepdims=True))
    p /= np.sum(p, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(p > 1e-10, np.log2(np.where(p > 1e-10, p, 1.0)), 0.0)
    terms = np.where(p > 1e-10, p * log_p, 0.0)
    return float(-np.sum(terms) / p.shape[0])


# ── §4.3 Technical Report Evaluation Metrics ─────────────────────────────────
#
# 來源：Technical Report §4.3「Evaluation Metrics」
#
# Model API pricing ($/1M tokens) — Technical Report §4.2.4 Candidate LLM Pool
# Tuple: (prompt_price, completion_price)

MODEL_PRICING: Dict[str, tuple] = {
    "llama-4-maverick-17b-128e-instruct-fp8": (0.50, 0.77),
    "llama-4-maverick-17b":                   (0.50, 0.77),
    "llama-4-maverick":                        (0.50, 0.77),
    "llama-4-scout-17b-16e-instruct-fp8":     (0.18, 0.18),
    "llama-4-scout-17b":                       (0.18, 0.18),
    "llama-4-scout":                           (0.18, 0.18),
    "gpt-oss-20b":                             (0.30, 0.30),
    "mistral-small-3.1-24b-instruct-2503":    (0.10, 0.30),
    "mistral-small-3.1":                       (0.10, 0.30),
    "mistral-small":                           (0.10, 0.30),
    "google-gemma-3-27b":                      (0.10, 0.20),
    "gemma-3-27b":                             (0.10, 0.20),
    "microsoft-phi-4":                         (0.07, 0.14),
    "phi-4":                                   (0.07, 0.14),
}


def model_unit_costs(model_names: List[str]) -> np.ndarray:
    """
    將 model 名稱清單轉換為每 token 平均單價向量。

    單價定義（§3.2）：
        unit_cost = (Prompt_Price + Completion_Price) / 2  （$/1M tokens）

    用於 compute_cost()；當 test_tokens 單位為 「token 數」 時，
    乘以 unit_cost / 1_000_000 即可得 USD 費用。

    Returns:
        np.ndarray of shape (M,), 單位 $/1M tokens
    """
    costs = []
    for name in model_names:
        key = name.lower().strip()
        if key in MODEL_PRICING:
            pp, cp = MODEL_PRICING[key]
        else:
            pp, cp = next(
                ((p, c) for k, (p, c) in MODEL_PRICING.items()
                 if k in key or key in k),
                (0.0, 0.0),
            )
        costs.append((pp + cp) / 2.0)
    return np.array(costs, dtype=float)


def compute_hit_rate(probs: np.ndarray, test_scores: np.ndarray) -> float:
    """
    Hit Rate（HR，↑）— Technical Report §4.3

    A routing decision is a "Hit" if the selected model's score equals the
    maximum achievable score among all candidate models for that query.

        Hit ⟺ Score_selected = max(Score_all_models)

    Args:
        probs:       (N, M) prediction matrix; argmax → selected model
        test_scores: (N, M) ground-truth score matrix

    Returns:
        HR ∈ [0, 1]
    """
    idx = np.argmax(probs, axis=1)
    n = len(idx)
    selected = test_scores[np.arange(n), idx]
    best = np.max(test_scores, axis=1)
    return float(np.mean(selected >= best - 1e-9))


def compute_cost(
    probs: np.ndarray,
    test_tokens: np.ndarray,
    model_costs: "Optional[Union[List[float], np.ndarray]]" = None,
) -> float:
    """
    Average routing Cost（↓）— Technical Report §4.3

        Cost = Avg_Tokens × (Prompt_Price + Completion_Price) / 2

    When model_costs is None (all 1.0), returns plain average token count,
    which reproduces the Cost column values in the paper's tables.

    To get USD cost, pass model_unit_costs(model_names) / 1_000_000.

    Args:
        probs:       (N, M) prediction matrix
        test_tokens: (N, M) token counts per (query, model)
        model_costs: M-length cost multiplier per model (default: 1.0 each)

    Returns:
        mean cost per query
    """
    idx = np.argmax(probs, axis=1)
    n = len(idx)
    mc = np.ones(test_tokens.shape[1]) if model_costs is None else np.asarray(model_costs, dtype=float)
    per_query = test_tokens[np.arange(n), idx] * mc[idx]
    return float(np.mean(per_query))


def compute_ter(
    router_hr: float,
    router_cost: float,
    baseline_hr: float,
    baseline_cost: float,
) -> "Union[float, str]":
    """
    Trade-off Efficiency Ratio（TER，↑）— Technical Report §4.3

        TER = Cost_Savings(%) / Hit_Rate_Sacrifice(%)

    Special return values:
        "Dominant" — router improves (or matches) BOTH HR and Cost vs baseline
        "Inv"      — router worsens BOTH HR and Cost vs baseline (Invalid)

    Args:
        router_hr, router_cost:     metrics of the router under evaluation
        baseline_hr, baseline_cost: metrics of the strongest baseline

    Returns:
        "Dominant", "Inv", or float TER
    """
    hr_improved  = router_hr   >= baseline_hr   - 1e-9
    cost_improved = router_cost <= baseline_cost + 1e-9

    if hr_improved and cost_improved:
        return "Dominant"

    if not hr_improved and not cost_improved:
        return "Inv"

    # Mixed: HR worsened but cost saved (the typical trade-off case)
    hr_sacrifice_pct  = (baseline_hr - router_hr)     / baseline_hr   * 100 if baseline_hr   > 0 else 0.0
    cost_savings_pct  = (baseline_cost - router_cost) / baseline_cost * 100 if baseline_cost > 0 else 0.0

    if abs(hr_sacrifice_pct) < 1e-9:
        return "Dominant"

    return cost_savings_pct / hr_sacrifice_pct


def compute_nbs(
    router_hr: float,
    router_cost: float,
    baseline_hr: float,
    baseline_cost: float,
) -> float:
    """
    Net Benefit Score（NBS，↑）— Technical Report §4.3

        NBS = 3 × ΔHit_Rate(%) + ΔCost_Savings(%)

    The 3:1 accuracy-to-cost weighting prevents "gaming" by routing everything
    to the cheapest model regardless of accuracy loss.

    Δ values are relative to the strongest baseline in the candidate pool.

    Args:
        router_hr, router_cost:     metrics of the router under evaluation
        baseline_hr, baseline_cost: metrics of the strongest baseline (highest HR)

    Returns:
        NBS (positive = net gain, negative = net harm)
    """
    delta_hr_pct      = (router_hr - baseline_hr)      * 100
    cost_savings_pct  = (baseline_cost - router_cost) / baseline_cost * 100 if baseline_cost > 0 else 0.0
    return 3.0 * delta_hr_pct + cost_savings_pct


# ── evaluate() — 主要評估函數 ─────────────────────────────────────────────────


def evaluate(
    probs: np.ndarray,
    data: RouterData,
) -> dict:
    """
    純函數：給定 (N_test, N_models) 預測分數矩陣，在 RouterData.test_* 上計算評估指標。

    與 router 實例無關，可單獨使用於任意預測矩陣。

    Args:
        probs: (N_test, N_models) 預測分數矩陣（argmax 為所選模型）
        data:  RouterData（只使用 test_score / test_tokens / test_time）

    Returns:
        dict with keys:
            mu, vb, ep          — legacy metrics (continuous score, ratio, entropy)
            hr                  — Hit Rate §4.3 (binary)
            avg_tokens          — average tokens of selected model (= Cost when model_costs=1)
            avg_latency         — average latency of selected model
    """
    idx = np.argmax(probs, axis=1)
    Y = data.test_score
    n = len(idx)

    selected = Y[np.arange(n), idx]
    mu = float(np.mean(selected))
    oracle_score = float(np.mean(np.max(Y, axis=1)))
    vb = mu / oracle_score if oracle_score > 0 else 0.0
    ep = _softmax_entropy(probs)
    hr = compute_hit_rate(probs, Y)

    avg_tokens = avg_latency = 0.0
    T, L = data.test_tokens, data.test_time
    if T is not None and L is not None and T.shape[0] == n:
        avg_tokens = float(np.mean(T[np.arange(n), idx]))
        avg_latency = float(np.mean(L[np.arange(n), idx]))

    return {
        "mu":          mu,
        "vb":          vb,
        "ep":          ep,
        "hr":          hr,
        "avg_tokens":  avg_tokens,
        "avg_latency": avg_latency,
    }


def evaluate_full(
    probs: np.ndarray,
    data: RouterData,
    *,
    model_costs: "Optional[Union[List[float], np.ndarray]]" = None,
    baseline_hr: Optional[float] = None,
    baseline_cost: Optional[float] = None,
) -> dict:
    """
    完整 §4.3 評估：在 evaluate() 的基礎上額外計算 Cost、TER、NBS。

    Args:
        probs:         (N_test, N_models) 預測分數矩陣
        data:          RouterData
        model_costs:   M-length cost multiplier per model；傳 model_unit_costs(names) / 1e6
                       取得 USD 費用；預設 None（= 1.0，回傳原始 avg_tokens，重現論文 Cost 欄位）
        baseline_hr:   最強基準（Strongest Baseline）的 HR；提供時才計算 TER / NBS
        baseline_cost: 最強基準的 Cost；需與 model_costs 使用相同單位

    Returns:
        evaluate() 的所有欄位 + cost [+ ter + nbs]
    """
    base = evaluate(probs, data)
    T = data.test_tokens

    cost: Optional[float] = None
    if T is not None and T.shape[0] == len(np.argmax(probs, axis=1)):
        cost = compute_cost(probs, T, model_costs)
        base["cost"] = cost

    if baseline_hr is not None and baseline_cost is not None and cost is not None:
        base["ter"] = compute_ter(base["hr"], cost, baseline_hr, baseline_cost)
        base["nbs"] = compute_nbs(base["hr"], cost, baseline_hr, baseline_cost)

    return base


# ── RunResult ─────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """單次 router train + eval 的結果記錄。"""

    label:       str
    size:        "float | int"
    seed:        int
    n_train:     int
    mu:          float
    vb:          float
    ep:          float
    hr:          float          = 0.0   # §4.3 Hit Rate（新增）
    avg_tokens:  float          = 0.0
    avg_latency: float          = 0.0
    cost:        Optional[float] = None  # §4.3 Cost（需 model_costs）
    ter:  "Optional[Union[float, str]]" = None   # §4.3 TER
    nbs:         Optional[float] = None  # §4.3 NBS


# ── RouterBenchmark ───────────────────────────────────────────────────────────


class RouterBenchmark:
    """
    多 router × 多訓練配置的對比評估框架。

    兩種對比維度可自由組合：
    - 同一 router，不同 train_size（縮放實驗）
    - 不同 router，相同 train_size（橫向對比）

    Usage::

        from LLMRouter.router import RouterData, RouterBenchmark, KNNRouter, OracleRouter

        data = RouterData.load("data.npz")
        bench = RouterBenchmark(data)

        # 縮放實驗：同一 router，不同訓練量，每個大小重複 3 個 seed
        bench.run(KNNRouter, {"k": 10},
                  sizes=[0.1, 0.3, 0.5, 1.0], seeds=[0, 1, 2])

        # 橫向對比：多個 router，相同訓練量
        bench.run(KNNRouter, {"k": 10},  label="knn")
        bench.run(OracleRouter,           label="oracle")

        bench.print_table()

        # 完整 §4.3 指標（需提供 model_costs 與最強基準）
        bench.print_table(show_cost_metrics=True, model_costs=[1.0, 1.0, ...])
    """

    def __init__(self, data: RouterData) -> None:
        self.data = data
        self._results: List[RunResult] = []

    def run(
        self,
        router_cls: type,
        router_kwargs: Optional[dict] = None,
        *,
        sizes: "float | int | List[float | int]" = 1.0,
        seeds: "int | List[int]" = 42,
        preprocess_fn: Optional[Callable[[RouterData], RouterData]] = None,
        label: Optional[str] = None,
        model_costs: "Optional[Union[List[float], np.ndarray]]" = None,
        baseline_hr: Optional[float] = None,
        baseline_cost: Optional[float] = None,
    ) -> "RouterBenchmark":
        """
        以指定配置訓練並評估 router，結果累積到 self._results。

        Args:
            router_cls:    router 類別（繼承 BaseRouter）
            router_kwargs: router 建構子參數（預設 {}）
            sizes:         訓練集大小；float=比例(0~1]，int=固定筆數；可傳清單
            seeds:         每個 size 重複的隨機種子；int 表示單一 seed；清單表示多 seed
            preprocess_fn: 對 data 的前處理（filter_by_variance、deduplicate_train 等）
                           只影響訓練集，test / val 不受影響
            label:         顯示標籤（預設為 router_cls.__name__）
            model_costs:   M-length cost multiplier；傳入則同時計算 Cost
            baseline_hr:   最強基準 HR；與 baseline_cost 同時提供時計算 TER / NBS
            baseline_cost: 最強基準 Cost

        Returns:
            self（支援 method chaining）
        """
        if router_kwargs is None:
            router_kwargs = {}
        if not isinstance(sizes, list):
            sizes = [sizes]
        if isinstance(seeds, int):
            seeds = [seeds]
        if label is None:
            label = router_cls.__name__

        data = preprocess_fn(self.data) if preprocess_fn else self.data

        for size in sizes:
            for seed in seeds:
                sub = data.subsample_train(size, seed=seed)
                n_train = len(sub.train_prompt)
                r = router_cls(**router_kwargs)
                r.fit(sub)

                # Oracle / Random override predict_probs → 使用各自的 evaluate(data)
                try:
                    # 若 router 有 _predict_for_eval，優先用它（可利用預存 embedding）
                    if hasattr(r, "_predict_for_eval"):
                        probs = r._predict_for_eval(data)
                    else:
                        probs = r.predict_probs(data.test_prompt)
                    metrics = evaluate_full(
                        probs, data,
                        model_costs=model_costs,
                        baseline_hr=baseline_hr,
                        baseline_cost=baseline_cost,
                    )
                except NotImplementedError:
                    # 特殊 router（OracleRouter、RandomRouter）有自己的 evaluate()
                    metrics = r.evaluate(data)
                    # 若有 tokens 資料，補算 cost/ter/nbs
                    T = data.test_tokens
                    if T is not None and model_costs is not None:
                        # 用 argmax(test_score) 作為 oracle 的決策
                        idx = np.argmax(data.test_score, axis=1)
                        mc = np.asarray(model_costs, dtype=float)
                        cost = float(np.mean(T[np.arange(len(idx)), idx] * mc[idx]))
                        metrics["cost"] = cost
                        if baseline_hr is not None and baseline_cost is not None:
                            metrics["ter"] = compute_ter(
                                metrics["hr"], cost, baseline_hr, baseline_cost
                            )
                            metrics["nbs"] = compute_nbs(
                                metrics["hr"], cost, baseline_hr, baseline_cost
                            )

                self._results.append(RunResult(
                    label=label,
                    size=size,
                    seed=seed,
                    n_train=n_train,
                    mu=metrics["mu"],
                    vb=metrics["vb"],
                    ep=metrics["ep"],
                    hr=metrics.get("hr", 0.0),
                    avg_tokens=metrics.get("avg_tokens", 0.0),
                    avg_latency=metrics.get("avg_latency", 0.0),
                    cost=metrics.get("cost"),
                    ter=metrics.get("ter"),
                    nbs=metrics.get("nbs"),
                ))

        return self

    def results(self) -> List[RunResult]:
        """回傳所有已累積的 RunResult。"""
        return list(self._results)

    def table(self, show_cost_metrics: bool = False) -> str:
        """
        回傳格式化的對比表格字串。

        以 (label, size) 分組，同一組的多個 seed 結果取平均。
        同一 label 的不同 size 連續顯示；不同 label 間加分隔線。

        Args:
            show_cost_metrics: 若 True，額外顯示 HR / Cost / TER / NBS 欄位
        """
        if not self._results:
            return "(無結果)"

        from collections import defaultdict

        groups: dict = defaultdict(list)
        order: list = []
        seen_keys: set = set()
        for r in self._results:
            key = (r.label, r.size)
            groups[key].append(r)
            if key not in seen_keys:
                seen_keys.add(key)
                order.append(key)

        if show_cost_metrics:
            header = (
                f"{'router':<22}  {'size':>8}  {'n_train':>7}"
                f"  {'HR':>7}  {'Cost':>10}  {'TER':>10}  {'NBS':>8}"
                f"  {'mu':>7}  {'vb':>7}"
            )
        else:
            header = (
                f"{'router':<22}  {'size':>10}  {'n_train':>8}"
                f"  {'HR':>7}  {'mu':>8}  {'vb':>8}  {'ep':>8}"
                f"  {'tokens':>10}  {'latency':>10}"
            )
        sep = "-" * len(header)
        lines = [header, sep]

        prev_label: Optional[str] = None
        for (label, size) in order:
            runs = groups[(label, size)]
            if prev_label is not None and label != prev_label:
                lines.append(sep)
            prev_label = label

            n = len(runs)
            mu   = sum(r.mu   for r in runs) / n
            vb   = sum(r.vb   for r in runs) / n
            ep   = sum(r.ep   for r in runs) / n
            hr   = sum(r.hr   for r in runs) / n
            tok  = sum(r.avg_tokens  for r in runs) / n
            lat  = sum(r.avg_latency for r in runs) / n
            n_tr = round(sum(r.n_train for r in runs) / n)

            # Cost / TER / NBS — take mean of numeric values only
            costs = [r.cost for r in runs if r.cost is not None]
            avg_cost = sum(costs) / len(costs) if costs else None

            size_str = f"{size:.0%}" if isinstance(size, float) else str(size)

            if show_cost_metrics:
                cost_str = f"{avg_cost:>10.2f}" if avg_cost is not None else f"{'—':>10}"

                # TER: average numeric TERs; if all same string → show that string
                ters = [r.ter for r in runs if r.ter is not None]
                ter_str = "—"
                if ters:
                    str_ters = [t for t in ters if isinstance(t, str)]
                    num_ters = [t for t in ters if isinstance(t, (int, float))]
                    if str_ters and not num_ters:
                        ter_str = str_ters[0]
                    elif num_ters:
                        ter_str = f"{sum(num_ters) / len(num_ters):.2f}"

                nbss = [r.nbs for r in runs if r.nbs is not None]
                nbs_str = f"{sum(nbss)/len(nbss):+.2f}" if nbss else "—"

                lines.append(
                    f"{label:<22}  {size_str:>8}  {n_tr:>7d}"
                    f"  {hr:>7.4f}  {cost_str}  {ter_str:>10}  {nbs_str:>8}"
                    f"  {mu:>7.4f}  {vb:>7.4f}"
                )
            else:
                lines.append(
                    f"{label:<22}  {size_str:>10}  {n_tr:>8d}"
                    f"  {hr:>7.4f}  {mu:>8.4f}  {vb:>8.4f}  {ep:>8.4f}"
                    f"  {tok:>10.1f}  {lat:>10.4f}"
                )

        return "\n".join(lines)

    def print_table(self, show_cost_metrics: bool = False) -> None:
        """印出對比表格到 stdout。"""
        print(self.table(show_cost_metrics=show_cost_metrics))

    def strongest_baseline(self) -> Optional[RunResult]:
        """
        回傳已累積結果中 HR 最高的 RunResult（作為 §4.3 NBS / TER 基準）。
        若有多個 HR 相同，取 cost 最低者（較保守基準）。
        """
        if not self._results:
            return None
        return max(
            self._results,
            key=lambda r: (r.hr, -(r.cost or float("inf"))),
        )
