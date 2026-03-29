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
        emb_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        emb_batch_size: int = 16,
    ) -> None:
        self.k = k
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self._nn = None
        self._X_train: Optional[np.ndarray] = None
        self._Y_train: Optional[np.ndarray] = None

    def _fit(self, data: RouterData) -> None:
        from sklearn.neighbors import NearestNeighbors

        if data.train_embed is not None:
            print(f"[KNN] 使用預存訓練集嵌入（{len(data.train_prompt)} 筆）...")
            self._X_train = data.train_embed
        else:
            print(f"[KNN] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
            self._X_train = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)
        self._Y_train = data.train_score.astype(np.float32)

        k = min(self.k, len(data.train_prompt))
        self._nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        self._nn.fit(self._X_train)
        print(f"[KNN] 訓練完成（k={k}）。")

    def _predict_from_embed(self, X_test: np.ndarray) -> np.ndarray:
        _, indices = self._nn.kneighbors(X_test)
        predicted = np.zeros((len(X_test), self._Y_train.shape[1]), dtype=np.float32)
        for i, nbr_idx in enumerate(indices):
            predicted[i] = np.mean(self._Y_train[nbr_idx], axis=0)
        return predicted

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("請先呼叫 fit()")
        print(f"[KNN] 計算測試集嵌入（{len(prompts)} 筆）...")
        X_test = get_embeddings(prompts, self.emb_model, self.emb_batch_size)
        return self._predict_from_embed(X_test)

    def _predict_for_eval(self, data: RouterData) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("請先呼叫 fit()")
        if data.test_embed is not None:
            return self._predict_from_embed(data.test_embed)
        return self.predict_probs(data.test_prompt)
