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
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batch_size: int = 32,
    ) -> None:
        self.k = k
        self.temperature = temperature
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def _fit(self, data: RouterData) -> None:
        import torch
        import torch.nn.functional as F

        print(f"[SW] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
        X_raw = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)

        # L2 normalize for cosine similarity via dot product
        X_t = torch.FloatTensor(X_raw)
        self._X_train = F.normalize(X_t, p=2, dim=1).numpy()
        self._Y_train = data.train_score.astype(np.float32)
        print(f"[SW] 準備完成（k={self.k}, temp={self.temperature}）。")

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self._X_train is None:
            raise RuntimeError("請先呼叫 fit()")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[SW] 計算測試集嵌入（{len(prompts)} 筆）...")
        X_raw = get_embeddings(prompts, self.emb_model, self.emb_batch_size)

        X_test_t = F.normalize(torch.FloatTensor(X_raw), p=2, dim=1).to(device)
        X_train_t = torch.FloatTensor(self._X_train).to(device)
        Y_train_t = torch.FloatTensor(self._Y_train).to(device)

        # Cosine similarity via dot product of L2-normalized vectors
        sim = torch.matmul(X_test_t, X_train_t.T)  # (N_test, N_train)

        k = min(self.k, X_train_t.shape[0])
        topk_sims, topk_idx = torch.topk(sim, k=k, dim=1)

        # Softmax weights (N_test, k, 1)
        weights = F.softmax(topk_sims / self.temperature, dim=1).unsqueeze(-1)

        # Weighted average of neighbor scores
        neighbor_scores = Y_train_t[topk_idx]           # (N_test, k, N_models)
        predicted = torch.sum(weights * neighbor_scores, dim=1)  # (N_test, N_models)

        return predicted.cpu().numpy().astype(np.float32)
