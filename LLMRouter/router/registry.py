"""
Router Registry

新增 router 只需在 router 模組末尾呼叫 register()，不需要修改 __main__.py。

Usage::

    # 在 my_router.py 末尾
    from .registry import register

    def _kwargs(args):
        return {"k": args.k, "emb_model": args.emb_model}

    register("my_router", MyRouter, _kwargs)
"""

from __future__ import annotations

import argparse
from typing import Callable, Type

from .base import BaseRouter

# {name: (cls, kwargs_fn)}
# kwargs_fn(args: argparse.Namespace) -> dict
_REGISTRY: dict[str, tuple[Type[BaseRouter], Callable]] = {}


def register(
    name: str,
    cls: Type[BaseRouter],
    kwargs_fn: Callable[[argparse.Namespace], dict] | None = None,
) -> Type[BaseRouter]:
    """
    將 router class 以 name 註冊到全域 registry。

    Args:
        name:      CLI 使用的名稱（e.g. "knn"）
        cls:       BaseRouter 子類別
        kwargs_fn: (args: Namespace) -> dict，將 CLI args 轉成 __init__ kwargs。
                   不指定時預設回傳 {}。
    """
    _REGISTRY[name] = (cls, kwargs_fn or (lambda _: {}))
    return cls


def get(name: str) -> tuple[Type[BaseRouter], Callable]:
    """
    回傳 (RouterClass, kwargs_fn)。

    Raises:
        KeyError: 若 name 尚未註冊
    """
    if name not in _REGISTRY:
        available = ", ".join(list_routers()) or "（無）"
        raise KeyError(f"Router '{name}' 未註冊。已有：{available}")
    return _REGISTRY[name]


def build(name: str, args: argparse.Namespace) -> BaseRouter:
    """依 name 建立並回傳 router 實例。"""
    cls, kwargs_fn = get(name)
    return cls(**kwargs_fn(args))


def list_routers() -> list[str]:
    """回傳所有已註冊的 router 名稱清單（已排序）。"""
    return sorted(_REGISTRY)
