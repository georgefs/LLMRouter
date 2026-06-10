from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData
from ._embeddings import get_embeddings


class SWRankingRouter(BaseRouter):
    """
    Similarity-Weighted Ranking Router（改編自 RouterEval SW-Ranking）。

    計算測試 prompt 與訓練集的 cosine similarity，以 softmax 加權平均
    訓練集的分數，預測各模型的表現。

    Args:
        k: Top-K 近鄰數量（預設 50）
        temperature: Softmax 溫度（越小越聚焦於最相似的近鄰，預設 0.1）
        emb_model: 嵌入模型名稱
        emb_batch_size: 嵌入計算的 batch size
    """

    def __init__(
        self,
        k: int = 50,
        temperature: float = 0.1,
        emb_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        emb_batch_size: int = 16,
    ) -> None:
        super().__init__()
        self.k = k
        self.temperature = temperature
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def _fit(self, data: RouterData) -> None:
        import torch
        import torch.nn.functional as F

        if data.train_embed is not None:
            print(f"[SW] 使用預存訓練集嵌入（{len(data.train_prompt)} 筆）...")
            X_raw = data.train_embed
        else:
            print(f"[SW] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
            X_raw = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)

        X_t = torch.FloatTensor(X_raw)
        self._X_train = F.normalize(X_t, p=2, dim=1).numpy()
        self._Y_train = data.train_score.astype(np.float32)
        print(f"[SW] 準備完成（k={self.k}, temp={self.temperature}）。")

    def _predict_from_embed(self, X_raw: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_test_t = F.normalize(torch.FloatTensor(X_raw), p=2, dim=1).to(device)
        X_train_t = torch.FloatTensor(self._X_train).to(device)
        Y_train_t = torch.FloatTensor(self._Y_train).to(device)

        sim = torch.matmul(X_test_t, X_train_t.T)
        k = min(self.k, X_train_t.shape[0])
        topk_sims, topk_idx = torch.topk(sim, k=k, dim=1)
        weights = F.softmax(topk_sims / self.temperature, dim=1).unsqueeze(-1)
        neighbor_scores = Y_train_t[topk_idx]
        predicted = torch.sum(weights * neighbor_scores, dim=1)
        return predicted.cpu().numpy().astype(np.float32)

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("請先呼叫 fit()")
        print(f"[SW] 計算測試集嵌入（{len(prompts)} 筆）...")
        X_raw = get_embeddings(prompts, self.emb_model, self.emb_batch_size)
        return self._predict_from_embed(X_raw)

    def _predict_for_eval(self, data: RouterData) -> np.ndarray:
        if self._X_train is None:
            raise RuntimeError("請先呼叫 fit()")
        if data.test_embed is not None:
            return self._predict_from_embed(data.test_embed)
        return self.predict_probs(data.test_prompt)

    def save(self, path: "str | Path") -> None:
        """
        保存訓練好的 router（包含訓練集 embeddings + model_names）。

        Args:
            path: 保存路徑 (.pkl 檔案)
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_names': self.model_names,
            'k': self.k,
            'temperature': self.temperature,
            'emb_model': self.emb_model,
            'emb_batch_size': self.emb_batch_size,
            '_X_train': self._X_train,
            '_Y_train': self._Y_train,
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[SW] Saved to {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "SWRankingRouter":
        """
        載入訓練好的 router（恢復訓練集 embeddings + model_names）。

        Args:
            path: 載入路徑 (.pkl 檔案)

        Returns:
            SWRankingRouter 實例，包含已恢復的 model_names
        """
        import pickle
        from pathlib import Path

        path = Path(path)

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        router = cls(
            k=checkpoint['k'],
            temperature=checkpoint['temperature'],
            emb_model=checkpoint['emb_model'],
            emb_batch_size=checkpoint['emb_batch_size'],
        )

        # 恢復 model_names 和訓練數據
        router.model_names = checkpoint['model_names']
        router._X_train = checkpoint['_X_train']
        router._Y_train = checkpoint['_Y_train']

        print(f"[SW] Loaded from {path}")
        return router


from .registry import register

register("sw", SWRankingRouter, lambda a: {
    "k": a.k, "temperature": a.temperature, "emb_model": a.emb_model,
})
