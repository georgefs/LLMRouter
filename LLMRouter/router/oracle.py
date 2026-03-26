from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseRouter
from .data import RouterData


class OracleRouter(BaseRouter):
    """
    Oracle 基準線：每次都選分數最高的模型（需要測試集 ground truth）。

    僅供評估參考，不適用於生產環境。
    Entropy 計算採用 L1-normalized 分數（與 RouterEval r_o_router.py 一致）。
    """

    def _fit(self, data: RouterData) -> None:
        pass  # Oracle 不需要訓練

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "OracleRouter 需要 ground truth，請改用 evaluate(data) 直接評估。"
        )

    def evaluate(self, data: RouterData) -> dict:
        Y = data.test_score  # (N, M)
        n, m = Y.shape

        idx = np.argmax(Y, axis=1)
        mu = float(np.mean(Y[np.arange(n), idx]))
        oracle = float(np.mean(np.max(Y, axis=1)))
        vb = mu / oracle if oracle > 0 else 1.0

        # Entropy: L1-normalize per sample（與 r_o_router.py 一致）
        row_sums = Y.sum(axis=1, keepdims=True)
        safe_denom = np.where(row_sums > 0, row_sums, 1.0)
        probs = np.where(row_sums > 0, Y / safe_denom, np.ones_like(Y) / m)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.where(probs > 1e-10, np.log2(np.where(probs > 1e-10, probs, 1.0)), 0.0)
        terms = np.where(probs > 1e-10, probs * log_p, 0.0)
        ep = float(-np.sum(terms) / n)

        avg_tokens = 0.0
        avg_latency = 0.0
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


class RandomRouter(BaseRouter):
    """
    Random 基準線：隨機選擇模型。

    評估時模擬 1000 次並取平均（與 RouterEval r_o_router.py 一致）。
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._n_models: int = 0

    def _fit(self, data: RouterData) -> None:
        self._n_models = data.train_score.shape[1]

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if not self._n_models:
            raise RuntimeError("請先呼叫 fit()")
        n, m = len(prompts), self._n_models
        return np.ones((n, m), dtype=np.float32) / m

    def evaluate(self, data: RouterData) -> dict:
        Y = data.test_score
        n, m = Y.shape

        rng = np.random.default_rng(self.seed)
        acc_list = []
        for _ in range(1000):
            idx = rng.integers(0, m, size=n)
            acc_list.append(float(np.mean(Y[np.arange(n), idx])))
        mu = float(np.mean(acc_list))

        oracle = float(np.mean(np.max(Y, axis=1)))
        vb = mu / oracle if oracle > 0 else 0.0

        # Entropy: uniform
        probs = np.ones((n, m), dtype=np.float32) / m
        terms = probs * np.log2(probs)
        ep = float(-np.sum(terms) / n)

        avg_tokens = float(np.mean(data.test_tokens)) if data.test_tokens is not None else 0.0
        avg_latency = float(np.mean(data.test_time)) if data.test_time is not None else 0.0

        return {
            "mu": mu,
            "vb": vb,
            "ep": ep,
            "avg_tokens": avg_tokens,
            "avg_latency": avg_latency,
        }
