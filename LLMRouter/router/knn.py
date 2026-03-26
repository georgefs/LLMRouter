from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData
from ._embeddings import get_embeddings


class KNNRouter(BaseRouter):
    """
    KNN Router（改編自 RouterEval PRKnn-knn）。

    對測試 prompt 找最近的 k 個訓練樣本，以其平均分數預測最佳模型。

    Args:
        k: 近鄰數量（預設 10）
        emb_model: 嵌入模型名稱（預設 all-MiniLM-L6-v2）
        emb_batch_size: 嵌入計算的 batch size
    """

    def __init__(
        self,
        k: int = 10,
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batch_size: int = 32,
    ) -> None:
        self.k = k
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self._nn = None
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def _fit(self, data: RouterData) -> None:
        from sklearn.neighbors import NearestNeighbors

        print(f"[KNN] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
        self._X_train = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)
        self._Y_train = data.train_score.astype(np.float32)

        k = min(self.k, len(data.train_prompt))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self._nn.fit(self._X_train)
        print(f"[KNN] 訓練完成（k={k}）。")

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("請先呼叫 fit()")

        print(f"[KNN] 計算測試集嵌入（{len(prompts)} 筆）...")
        X_test = get_embeddings(prompts, self.emb_model, self.emb_batch_size)
        _, indices = self._nn.kneighbors(X_test)

        predicted = np.zeros((len(prompts), self._Y_train.shape[1]), dtype=np.float32)
        for i, nbr_idx in enumerate(indices):
            predicted[i] = np.mean(self._Y_train[nbr_idx], axis=0)
        return predicted
