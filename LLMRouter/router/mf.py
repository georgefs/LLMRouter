from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData
from ._embeddings import get_embeddings


class MFRouter(BaseRouter):
    """
    Matrix Factorization Router（改編自 RouterEval MF）。

    使用 PyTorch 訓練 MF 模型，將 prompt embedding 投影到隱空間，
    再與可學習的模型向量做點積，預測各模型的分數。

    Args:
        latent_dim: 隱空間維度（預設 64）
        epochs: 訓練 epochs（預設 50）
        batch_size: 訓練 batch size（預設 32）
        lr: 學習率（預設 0.001）
        dropout: Dropout 比例（預設 0.1）
        emb_model: 嵌入模型名稱
        emb_batch_size: 嵌入計算的 batch size
        seed: 隨機種子
    """

    def __init__(
        self,
        latent_dim: int = 64,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        dropout: float = 0.1,
        emb_model: str = "mixedbread-ai/mxbai-embed-large-v1",
        emb_batch_size: int = 16,
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self.seed = seed
        self._model = None
        self._device = None

    def _build_net(self, input_dim: int, num_models: int):
        import torch
        import torch.nn as nn

        latent_dim = self.latent_dim
        dropout = self.dropout

        class _MFNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Sequential(
                    nn.Linear(input_dim, latent_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.model_embeddings = nn.Parameter(
                    torch.randn(num_models, latent_dim) * 0.01
                )
                self.model_bias = nn.Parameter(torch.zeros(num_models))

            def forward(self, x):
                q = self.projection(x)
                return torch.matmul(q, self.model_embeddings.T) + self.model_bias

        return _MFNet()

    def _fit(self, data: RouterData) -> None:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        print(f"[MF] Device: {device}")

        if data.train_embed is not None:
            print(f"[MF] 使用預存訓練集嵌入（{len(data.train_prompt)} 筆）...")
            X_train = data.train_embed
        else:
            print(f"[MF] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
            X_train = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)

        X_t = torch.FloatTensor(X_train).to(device)
        Y_t = torch.FloatTensor(data.train_score).to(device)
        n, input_dim = X_t.shape
        num_models = data.train_score.shape[1]

        model = self._build_net(input_dim, num_models).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        model.train()
        print(f"[MF] 開始訓練（{self.epochs} epochs）...")
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(model(X_t[idx]), Y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch [{epoch+1}/{self.epochs}] "
                    f"Loss: {epoch_loss / max(1, num_batches):.4f}"
                )

        self._model = model
        print("[MF] 訓練完成。")

    def _predict_from_embed(self, X: np.ndarray) -> np.ndarray:
        import torch
        X_t = torch.FloatTensor(X).to(self._device)
        self._model.eval()
        with torch.no_grad():
            scores = self._model(X_t).cpu().numpy()
        return scores.astype(np.float32)

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("請先呼叫 fit()")
        print(f"[MF] 計算嵌入（{len(prompts)} 筆）...")
        X = get_embeddings(prompts, self.emb_model, self.emb_batch_size)
        return self._predict_from_embed(X)

    def _predict_for_eval(self, data: RouterData) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("請先呼叫 fit()")
        if data.test_embed is not None:
            return self._predict_from_embed(data.test_embed)
        return self.predict_probs(data.test_prompt)
