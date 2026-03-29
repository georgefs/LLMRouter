from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .data import RouterData
from .eval import evaluate as _evaluate_fn


class BaseRouter(ABC):
    """
    Router 基底類別。

    子類別需實作：
        fit(data)            — 以 RouterData.train_* 訓練
        predict_probs(prompts) — 回傳 (N, M) 分數矩陣（argmax 為選擇的模型）
    """

    def fit(
        self,
        data: RouterData,
        train_size: "float | int" = 1.0,
        seed: int = 42,
    ) -> None:
        """
        以訓練集訓練 router。

        Args:
            data: RouterData
            train_size: 訓練資料大小
                        float → 比例（0 < train_size ≤ 1.0，預設 1.0 使用全部）
                        int   → 固定筆數（≥ 1）
            seed: subsample 的隨機種子
        """
        if train_size != 1.0:
            data = data.subsample_train(train_size, seed=seed)
        self._fit(data)

    @abstractmethod
    def _fit(self, data: RouterData) -> None:
        """子類別實作訓練邏輯。"""

    @abstractmethod
    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        """
        回傳 (N, M) 分數矩陣。
        分數不必是機率，argmax 將用於選擇模型。
        """

    def predict(self, prompts: List[str]) -> np.ndarray:
        """回傳每個 prompt 對應的模型 index（shape: (N,)）。"""
        return np.argmax(self.predict_probs(prompts), axis=1)

    def _predict_for_eval(self, data: RouterData) -> np.ndarray:
        """
        evaluate() 內部使用的預測入口。

        預設呼叫 predict_probs(data.test_prompt)；embedding-based 子類別
        可覆寫此方法，在 data.test_embed 存在時直接使用，省略重複計算。
        """
        return self.predict_probs(data.test_prompt)

    def evaluate(self, data: RouterData) -> dict:
        """
        在 RouterData.test_* 上評估，回傳指標字典。

        Keys: mu, vb, ep, avg_tokens, avg_latency
        """
        return _evaluate_fn(self._predict_for_eval(data), data)
