"""
Annotator Registry

新增 annotator 只需在模組末尾呼叫 register()，不需要修改 __main__.py。

factory_fn 簽名：(args: Namespace, config: dict) -> BaseAnnotator

Usage::

    # 在 my_annotator.py 末尾
    from .registry import register

    def _factory(args, config):
        return MyAnnotator(param=args.my_param)

    register("my_strategy", _factory)
"""

from __future__ import annotations

import argparse
from typing import Callable

from .base import BaseAnnotator

# {strategy: factory_fn}
_REGISTRY: dict[str, Callable] = {}


def register(strategy: str, factory: Callable) -> Callable:
    """
    將 annotator factory 以 strategy 名稱註冊。

    factory 簽名：(args: argparse.Namespace, config: dict) -> BaseAnnotator
    """
    _REGISTRY[strategy] = factory
    return factory


def get(strategy: str) -> Callable:
    """
    回傳對應的 factory function。

    Raises:
        KeyError: 若 strategy 尚未註冊
    """
    if strategy not in _REGISTRY:
        available = ", ".join(list_strategies()) or "（無）"
        raise KeyError(f"Annotator strategy '{strategy}' 未註冊。已有：{available}")
    return _REGISTRY[strategy]


def build(strategy: str, args: argparse.Namespace, config: dict) -> BaseAnnotator:
    """依 strategy 建立並回傳 annotator 實例。"""
    return get(strategy)(args, config)


def list_strategies() -> list[str]:
    """回傳所有已註冊的 strategy 名稱（已排序）。"""
    return sorted(_REGISTRY)
