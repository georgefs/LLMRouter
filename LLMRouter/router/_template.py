"""
新 Router 實作範本
==================
複製此檔案，全域搜尋 MyRouter / my_router 替換為實際名稱。

最少需要實作
------------
  _fit(data)              訓練邏輯，可讀取 data.train_embed（預計算嵌入）
  predict_probs(prompts)  推論，回傳 shape (N, M) 的分數矩陣

可選覆寫
--------
  _predict_for_eval(data) 若有預計算嵌入可避免重複計算（見 KNNRouter 範例）
  save(path) / load(path) 若需要持久化超出 pickle 預設的欄位

新增後不需修改任何其他檔案
--------------------------
  register() 呼叫完成後，CLI 即可使用：
    python -m LLMRouter router train my_router --data data.npz
    python -m LLMRouter router eval  my_router --data data.npz --model r.pkl
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import numpy as np

from .base import BaseRouter
from .data import RouterData


class MyRouter(BaseRouter):
    """
    一句話說明這個 router 的策略。

    Args:
        param_a: 說明用途和預設值
        param_b: 說明用途和預設值
    """

    def __init__(
        self,
        param_a: int = 10,
        param_b: float = 0.1,
    ) -> None:
        super().__init__()
        self.param_a = param_a
        self.param_b = param_b
        # 內部訓練狀態，初始化為 None
        self._state = None

    # ── 訓練 ──────────────────────────────────────────────────────────────────

    def _fit(self, data: RouterData) -> None:
        """
        以 data.train_* 訓練 router。

        data.train_embed 是預計算嵌入（若 prepare 時有指定 --emb-model），
        否則為 None，需自行呼叫 get_embeddings()。
        """
        if data.train_embed is not None:
            X = data.train_embed          # shape (N, D)
        else:
            from ._embeddings import get_embeddings
            X = get_embeddings(data.train_prompt, model="sentence-transformers/all-MiniLM-L6-v2")

        Y = data.train_score              # shape (N, M)

        # TODO: 在此實作訓練邏輯，結果存入 self._state
        raise NotImplementedError

    # ── 推論 ──────────────────────────────────────────────────────────────────

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        """
        回傳 shape (N, M) 分數矩陣。
        argmax(axis=1) 為選中的模型索引。
        分數不需要是機率，相對大小即可。
        """
        if self._state is None:
            raise RuntimeError("請先呼叫 fit()")
        # TODO: 嵌入 prompts 並計算分數
        raise NotImplementedError

    def _predict_for_eval(self, data: RouterData) -> np.ndarray:
        """
        （可選）評估時若有預計算嵌入，可直接使用以加速。
        不覆寫時預設呼叫 predict_probs(data.test_prompt)。
        """
        if data.test_embed is not None:
            # 直接使用預計算嵌入，跳過重新計算
            pass  # TODO: 實作
        return self.predict_probs(data.test_prompt)

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_names": self.model_names,
            "param_a": self.param_a,
            "param_b": self.param_b,
            "_state": self._state,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"[MyRouter] Saved to {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "MyRouter":
        path = Path(path)
        with open(path, "rb") as f:
            ck = pickle.load(f)
        r = cls(param_a=ck["param_a"], param_b=ck["param_b"])
        r.model_names = ck["model_names"]
        r._state = ck["_state"]
        print(f"[MyRouter] Loaded from {path}")
        return r


# ── Registry 登記（放在檔案最末，import 時自動執行）──────────────────────────
#
# kwargs_fn 將 CLI args 轉換成 __init__ 所需的 dict。
# 對應的 --param-a 參數已在 __main__._add_router_args() 中統一定義；
# 若需要新的 CLI 參數，請在該函式中新增，並在 kwargs_fn 中取用。
#
from .registry import register  # noqa: E402

register("my_router", MyRouter, lambda a: {
    "param_a": a.param_a,
    "param_b": a.param_b,
})
