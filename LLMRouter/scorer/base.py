from __future__ import annotations

from abc import ABC, abstractmethod


class BaseScorer(ABC):
    """
    評分策略基底類別。

    子類別實作 __call__，從 (dataset_item, response, annotations) 計算出一個 float 分數。

    Args（傳入 __call__）:
        dataset_item: 原始 dataset 記錄（question, answer, …）
        response:     model response 記錄（text, usage, response_time, …）
        annotations:  {strategy_name: {fields}} — 各 strategy 的 annotation 資料
    """

    @abstractmethod
    def __call__(
        self,
        dataset_item: dict,
        response: dict,
        annotations: dict,
    ) -> float:
        ...
