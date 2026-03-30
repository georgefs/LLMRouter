from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .data import RouterData


# ── 核心指標函數 ───────────────────────────────────────────────────────────────


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
        {"mu", "vb", "ep", "avg_tokens", "avg_latency"}
    """
    idx = np.argmax(probs, axis=1)
    Y = data.test_score
    n = len(idx)

    selected = Y[np.arange(n), idx]
    mu = float(np.mean(selected))
    oracle_score = float(np.mean(np.max(Y, axis=1)))
    vb = mu / oracle_score if oracle_score > 0 else 0.0
    ep = _softmax_entropy(probs)

    avg_tokens = avg_latency = 0.0
    T, L = data.test_tokens, data.test_time
    if T is not None and L is not None and T.shape[0] == n:
        avg_tokens = float(np.mean(T[np.arange(n), idx]))
        avg_latency = float(np.mean(L[np.arange(n), idx]))

    return {
        "mu": mu,
        "vb": vb,
        "ep": ep,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
    }


# ── RunResult ─────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    """單次 router train + eval 的結果記錄。"""

    label:       str            # router 名稱或自訂標籤
    size:        "float | int"  # 訓練集大小（比例或筆數）
    seed:        int
    n_train:     int            # 實際使用的訓練筆數
    mu:          float
    vb:          float
    ep:          float
    avg_tokens:  float
    avg_latency: float


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
                # 使用多型 evaluate()：一般 router 委派給純函數，
                # OracleRouter / RandomRouter 使用各自的特殊邏輯
                metrics = r.evaluate(data)
                self._results.append(RunResult(
                    label=label,
                    size=size,
                    seed=seed,
                    n_train=n_train,
                    **metrics,
                ))

        return self

    def results(self) -> List[RunResult]:
        """回傳所有已累積的 RunResult。"""
        return list(self._results)

    def table(self) -> str:
        """
        回傳格式化的對比表格字串。

        以 (label, size) 分組，同一組的多個 seed 結果取平均。
        同一 label 的不同 size 連續顯示；不同 label 間加分隔線。
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

        header = (
            f"{'router':<22}  {'size':>10}  {'n_train':>8}"
            f"  {'mu':>8}  {'vb':>8}  {'ep':>8}  {'tokens':>10}  {'latency':>10}"
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
            mu  = sum(r.mu  for r in runs) / n
            vb  = sum(r.vb  for r in runs) / n
            ep  = sum(r.ep  for r in runs) / n
            tok = sum(r.avg_tokens  for r in runs) / n
            lat = sum(r.avg_latency for r in runs) / n
            n_tr = round(sum(r.n_train for r in runs) / n)

            size_str = f"{size:.0%}" if isinstance(size, float) else str(size)
            lines.append(
                f"{label:<22}  {size_str:>10}  {n_tr:>8d}"
                f"  {mu:>8.4f}  {vb:>8.4f}  {ep:>8.4f}  {tok:>10.1f}  {lat:>10.4f}"
            )

        return "\n".join(lines)

    def print_table(self) -> None:
        """印出對比表格到 stdout。"""
        print(self.table())
