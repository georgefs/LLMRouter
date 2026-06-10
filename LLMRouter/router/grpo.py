from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData
from ._embeddings import get_embeddings


class GRPORouter(BaseRouter):
    """
    GRPO Router（Group Relative Policy Optimization）。

    以強化學習訓練 routing policy，無需 value network。
    對每個 prompt 採樣 G 個 routing decision 組成一個 group，
    用 group mean 作為 baseline 計算 advantage，避免 reward hacking。

    演算法：
        對每個 prompt x 與其 score 向量 y：
        1. 從當前 policy 採樣 G 個 action：a_1,...,a_G
        2. 計算 reward：r_j = y[a_j]
        3. 計算 group-relative advantage：A_j = (r_j - mean(r)) / (std(r) + ε)
        4. PPO-clip 更新：
           ratio_j = π_new(a_j|x) / π_old(a_j|x)
           L = -E[min(ratio_j * A_j, clip(ratio_j, 1-ε, 1+ε) * A_j)]
        5. 加入 KL 正則化防止 policy 偏移

    Args:
        group_size: 每個 prompt 採樣的 routing decision 數量（預設 8）
        hidden_dim: policy network 的隱藏層維度（預設 256）
        epochs: 訓練 epochs（預設 30）
        batch_size: 訓練 batch size（預設 32）
        lr: 學習率（預設 3e-4）
        clip_eps: PPO clip ε（預設 0.2）
        kl_coef: KL 正則化係數（預設 0.01）
        entropy_coef: entropy bonus 係數，鼓勵探索（預設 0.01）
        update_steps: 每批 rollout 執行的 policy update 次數（預設 1）
        emb_model: 嵌入模型名稱
        emb_batch_size: 嵌入計算 batch size
        seed: 隨機種子
    """

    def __init__(
        self,
        group_size: int = 8,
        hidden_dim: int = 256,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        kl_coef: float = 0.01,
        entropy_coef: float = 0.01,
        update_steps: int = 1,
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.group_size = group_size
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef
        self.entropy_coef = entropy_coef
        self.update_steps = update_steps
        self.emb_model = emb_model
        self.emb_batch_size = emb_batch_size
        self.seed = seed
        self._policy = None
        self._device = None

    def _build_policy(self, input_dim: int, num_models: int):
        import torch.nn as nn

        hidden = self.hidden_dim

        class _PolicyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, num_models),
                )

            def forward(self, x):
                return self.net(x)

            def log_probs(self, x):
                import torch.nn.functional as F
                return F.log_softmax(self.forward(x), dim=-1)

        return _PolicyNet()

    def _fit(self, data: RouterData) -> None:
        import torch
        import torch.nn.functional as F
        import torch.optim as optim

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        print(f"[GRPO] Device: {device}")

        print(f"[GRPO] 計算訓練集嵌入（{len(data.train_prompt)} 筆）...")
        X_train = get_embeddings(data.train_prompt, self.emb_model, self.emb_batch_size)

        X_t = torch.FloatTensor(X_train).to(device)
        Y_t = torch.FloatTensor(data.train_score).to(device)  # (N, M)
        n, input_dim = X_t.shape
        num_models = data.train_score.shape[1]

        policy = self._build_policy(input_dim, num_models).to(device)
        optimizer = optim.AdamW(policy.parameters(), lr=self.lr, weight_decay=1e-4)

        print(
            f"[GRPO] 開始訓練（{self.epochs} epochs, group_size={self.group_size}, "
            f"clip_eps={self.clip_eps}）..."
        )

        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            num_batches = 0

            for i in range(0, n, self.batch_size):
                batch_idx = perm[i : i + self.batch_size]
                x_batch = X_t[batch_idx]          # (B, D)
                y_batch = Y_t[batch_idx]           # (B, M)
                B = x_batch.size(0)

                # ── Phase 1: 用目前 policy 採樣 rollout ─────────────────────
                with torch.no_grad():
                    logits_old = policy(x_batch)                  # (B, M)
                    log_probs_old = F.log_softmax(logits_old, dim=-1)  # (B, M)
                    probs_old = torch.exp(log_probs_old)

                    # 擴展成 (B, G, M) 後採樣
                    probs_expanded = probs_old.unsqueeze(1).expand(B, self.group_size, num_models)
                    # 對每個 (b, g) 採樣一個 action
                    flat_probs = probs_expanded.reshape(B * self.group_size, num_models)
                    actions = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # (B*G,)
                    actions = actions.view(B, self.group_size)  # (B, G)

                    # 取對應的 log_prob_old
                    # log_probs_old_expanded: (B, G, M)
                    lp_old_exp = log_probs_old.unsqueeze(1).expand(B, self.group_size, num_models)
                    # gather action 對應的 log prob
                    log_prob_old_action = lp_old_exp.gather(
                        2, actions.unsqueeze(-1)
                    ).squeeze(-1)  # (B, G)

                    # 計算 reward：y[b, a_{b,g}]
                    y_expanded = y_batch.unsqueeze(1).expand(B, self.group_size, num_models)
                    rewards = y_expanded.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (B, G)

                    # Group-relative advantage
                    r_mean = rewards.mean(dim=1, keepdim=True)   # (B, 1)
                    r_std = rewards.std(dim=1, keepdim=True) + 1e-8
                    advantages = (rewards - r_mean) / r_std       # (B, G)

                # ── Phase 2: policy 更新（可多步）──────────────────────────
                for _ in range(self.update_steps):
                    logits_new = policy(x_batch)
                    log_probs_new = F.log_softmax(logits_new, dim=-1)  # (B, M)

                    lp_new_exp = log_probs_new.unsqueeze(1).expand(B, self.group_size, num_models)
                    log_prob_new_action = lp_new_exp.gather(
                        2, actions.unsqueeze(-1)
                    ).squeeze(-1)  # (B, G)

                    # PPO-clip
                    log_ratio = log_prob_new_action - log_prob_old_action
                    ratio = torch.exp(log_ratio)
                    adv = advantages.detach()

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # KL 正則化：KL(π_old || π_new) = Σ π_old * (log π_old - log π_new)
                    kl = (probs_old * (log_probs_old - log_probs_new)).sum(dim=-1).mean()

                    # Entropy bonus（鼓勵探索）
                    probs_new = torch.exp(log_probs_new)
                    entropy = -(probs_new * log_probs_new).sum(dim=-1).mean()

                    loss = policy_loss + self.kl_coef * kl - self.entropy_coef * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

            if (epoch + 1) % 5 == 0:
                print(
                    f"  Epoch [{epoch+1}/{self.epochs}] "
                    f"Loss: {epoch_loss / max(1, num_batches):.4f}"
                )

        self._policy = policy
        print("[GRPO] 訓練完成。")

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        import torch

        if self._policy is None:
            raise RuntimeError("請先呼叫 fit()")

        print(f"[GRPO] 計算嵌入（{len(prompts)} 筆）...")
        X = get_embeddings(prompts, self.emb_model, self.emb_batch_size)
        X_t = torch.FloatTensor(X).to(self._device)

        self._policy.eval()
        with torch.no_grad():
            logits = self._policy(X_t)
            scores = torch.softmax(logits, dim=-1).cpu().numpy()

        return scores.astype(np.float32)

    def save(self, path: "str | Path") -> None:
        """
        保存訓練好的 router（包含 policy 權重 + model_names）。

        Args:
            path: 保存路徑 (.pkl 檔案)
        """
        import pickle
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._policy is None:
            raise RuntimeError("Router must be trained first (call fit())")

        checkpoint = {
            'model_names': self.model_names,
            'group_size': self.group_size,
            'hidden_dim': self.hidden_dim,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'clip_eps': self.clip_eps,
            'kl_coef': self.kl_coef,
            'entropy_coef': self.entropy_coef,
            'update_steps': self.update_steps,
            'emb_model': self.emb_model,
            'emb_batch_size': self.emb_batch_size,
            'seed': self.seed,
            'policy_state': self._policy.state_dict(),
        }

        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"[GRPO] Saved to {path}")

    @classmethod
    def load(cls, path: "str | Path") -> "GRPORouter":
        """
        載入訓練好的 router（恢復 policy 權重 + model_names）。

        Args:
            path: 載入路徑 (.pkl 檔案)

        Returns:
            GRPORouter 實例，包含已恢復的 model_names 和 policy
        """
        import pickle
        from pathlib import Path
        import torch

        path = Path(path)

        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        router = cls(
            group_size=checkpoint['group_size'],
            hidden_dim=checkpoint['hidden_dim'],
            epochs=checkpoint['epochs'],
            batch_size=checkpoint['batch_size'],
            lr=checkpoint['lr'],
            clip_eps=checkpoint['clip_eps'],
            kl_coef=checkpoint['kl_coef'],
            entropy_coef=checkpoint['entropy_coef'],
            update_steps=checkpoint['update_steps'],
            emb_model=checkpoint['emb_model'],
            emb_batch_size=checkpoint['emb_batch_size'],
            seed=checkpoint['seed'],
        )

        # 恢復 model_names
        router.model_names = checkpoint['model_names']

        # 恢復 policy
        if checkpoint['policy_state'] is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            router._device = device

            # 需要先建立 policy 結構
            num_models = len(router.model_names)
            # 推斷 input_dim（從狀態字典）
            fc_weight = checkpoint['policy_state'].get('net.0.weight')
            if fc_weight is not None:
                input_dim = fc_weight.shape[1]
                router._policy = router._build_policy(input_dim, num_models).to(device)
                router._policy.load_state_dict(checkpoint['policy_state'])
                router._policy.eval()

        print(f"[GRPO] Loaded from {path}")
        return router
