from __future__ import annotations

from typing import Dict

from .base import BaseScorer

_registry: Dict[str, BaseScorer] = {}


def register(name: str, scorer: BaseScorer) -> BaseScorer:
    """
    將 scorer 實例以 name 註冊到全域 registry。

        register("point", FieldScorer("point"))
        register("cost",  FieldScorer("cost", strategy="cost"))

        my_scorer = register("custom", MyScorer())

    CLI 透過 --scorer <name> 取用已註冊的 scorer。
    """
    _registry[name] = scorer
    return scorer


def get(name: str) -> BaseScorer:
    """
    依名稱取得已註冊的 scorer。

    Raises:
        KeyError: 若 name 尚未註冊
    """
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "（無）"
        raise KeyError(f"Scorer '{name}' 未註冊。已有：{available}")
    return _registry[name]


def list_scorers() -> list[str]:
    """回傳所有已註冊的 scorer 名稱清單（已排序）。"""
    return sorted(_registry.keys())
