from __future__ import annotations

from .base import BaseScorer


class FieldScorer(BaseScorer):
    """
    從 annotations 取特定欄位的值作為分數（最常用的預設實作）。

    Args:
        field:    要取的欄位名稱（預設 "point"）
        strategy: 若指定，只從該 strategy 取值；
                  若不指定，依序搜尋所有 strategy 直到找到該欄位
    """

    def __init__(self, field: str = "point", strategy: str | None = None) -> None:
        self.field = field
        self.strategy = strategy

    def __call__(self, dataset_item: dict, response: dict, annotations: dict) -> float:
        if self.strategy:
            return float(annotations.get(self.strategy, {}).get(self.field, 0.0))
        for strat_data in annotations.values():
            if self.field in strat_data:
                return float(strat_data[self.field])
        return 0.0
