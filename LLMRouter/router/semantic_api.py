from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import List

import numpy as np

from .base import BaseRouter
from .data import RouterData


class SemanticAPIRouter(BaseRouter):
    """
    將 semantic-router 的 /api/v1/classify/intent HTTP API 包裝成 BaseRouter。

    不需本地訓練——遠端 semantic-router 自己維護分類邏輯（RL-driven 或規則式）。
    _fit() 只驗證 API 可連線並綁定 model_names。
    predict_probs() 對每個 prompt 發一次 HTTP 請求，將 recommended_model 轉成
    one-hot 機率向量（若 API 回傳未知 model 則 fallback 為均勻分布）。

    Args:
        base_url: semantic-router API server 的 base URL（預設 http://localhost:8080）
        timeout:  每次請求的 timeout（秒，預設 10.0）
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ── 訓練（連線驗證）────────────────────────────────────────────────────────

    def _fit(self, data: RouterData) -> None:
        url = f"{self.base_url}/api/v1/classify/intent"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": "ping"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout):
                pass
        except Exception as e:
            raise RuntimeError(f"[SemanticAPI] 無法連線到 {url}：{e}")
        print(f"[SemanticAPI] 連線成功：{self.base_url}，模型：{self.model_names}")

    # ── 推論 ──────────────────────────────────────────────────────────────────

    def _call_api(self, text: str) -> str | None:
        """呼叫一次 /api/v1/classify/intent，回傳 recommended_model 或 None。"""
        url = f"{self.base_url}/api/v1/classify/intent"
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": text}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
                return body.get("recommended_model")
        except Exception as e:
            print(f"[SemanticAPI] 請求失敗：{e}")
            return None

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self.model_names is None:
            raise RuntimeError("請先呼叫 fit()")
        n, m = len(prompts), len(self.model_names)
        probs = np.zeros((n, m), dtype=np.float32)
        model_index = {name: i for i, name in enumerate(self.model_names)}
        fallback = 0
        for i, prompt in enumerate(prompts):
            rec = self._call_api(prompt)
            if rec is not None and rec in model_index:
                probs[i, model_index[rec]] = 1.0
            else:
                probs[i] = 1.0 / m
                fallback += 1
        if fallback:
            print(f"[SemanticAPI] {fallback}/{n} 筆 fallback 為均勻分布（API 失敗或模型名稱不符）")
        return probs

    # ── 持久化 ────────────────────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "router_type": "semantic_api",
            "model_names": self.model_names,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)
        print(f"[SemanticAPI] Saved to {path}")

    @classmethod
    def load(cls, path_or_ck: "str | Path | dict") -> "SemanticAPIRouter":
        if isinstance(path_or_ck, dict):
            ck = path_or_ck
        else:
            import pickle
            from pathlib import Path
            with open(Path(path_or_ck), "rb") as f:
                ck = pickle.load(f)
            print(f"[SemanticAPI] Loaded from {path_or_ck}")
        r = cls(base_url=ck["base_url"], timeout=ck["timeout"])
        r.model_names = ck["model_names"]
        return r


# ── Registry ──────────────────────────────────────────────────────────────────

from .registry import register  # noqa: E402

register("semantic_api", SemanticAPIRouter, lambda a: {
    "base_url": getattr(a, "semantic_api_url", "http://localhost:8080"),
    "timeout":  getattr(a, "semantic_api_timeout", 10.0),
})
