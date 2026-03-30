from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from ..manager import DatasetManager


@dataclass
class RouterData:
    """RouterEval 格式的訓練 / 驗證 / 測試資料集。"""

    train_prompt: List[str]
    val_prompt: List[str]
    test_prompt: List[str]
    train_score: np.ndarray           # (N_train, N_models)
    val_score: np.ndarray             # (N_val,   N_models)
    test_score: np.ndarray            # (N_test,  N_models)
    test_tokens: np.ndarray           # (N_test,  N_models) — 無資料時為零矩陣
    test_time: np.ndarray             # (N_test,  N_models) — 無資料時為零矩陣
    models: List[str]                 # 模型名稱清單（長度 = N_models）
    train_embed: Optional[np.ndarray] = None  # (N_train, D) — router prepare --emb-model 時填入
    val_embed: Optional[np.ndarray] = None    # (N_val,   D)
    test_embed: Optional[np.ndarray] = None   # (N_test,  D)

    def save(self, path: str | Path) -> None:
        """儲存為 .npz 檔案。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict = dict(
            train_prompt=np.array(self.train_prompt, dtype=object),
            val_prompt=np.array(self.val_prompt, dtype=object),
            test_prompt=np.array(self.test_prompt, dtype=object),
            train_score=self.train_score,
            val_score=self.val_score,
            test_score=self.test_score,
            test_tokens=self.test_tokens,
            test_time=self.test_time,
            model=np.array(self.models, dtype=object),
        )
        if self.train_embed is not None:
            arrays["train_embed"] = self.train_embed
        if self.val_embed is not None:
            arrays["val_embed"] = self.val_embed
        if self.test_embed is not None:
            arrays["test_embed"] = self.test_embed
        np.savez(path, **arrays)

    def subsample_train(self, size: "float | int", seed: int = 42) -> "RouterData":
        """
        回傳一個新的 RouterData，val / test 完全相同，
        但 training set 被隨機縮減。

        Args:
            size: float → 比例（0 < size ≤ 1.0）；int → 固定筆數（≥ 1）
            seed: 隨機種子（不同 seed 可取不同子集）
        """
        n = len(self.train_prompt)
        if isinstance(size, float):
            if not 0 < size <= 1.0:
                raise ValueError(f"float size 需在 (0, 1]，got {size}")
            k = max(1, round(n * size))
        else:
            if size < 1:
                raise ValueError(f"int size 需 ≥ 1，got {size}")
            k = min(size, n)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=k, replace=False))
        return self._select_train(idx)

    def filter_by_variance(self, min_var: float = 0.0) -> "RouterData":
        """
        過濾訓練集中「無鑑別度」的樣本：當一筆樣本的所有模型分數幾乎相同
        （所有模型都對、或所有模型都錯），對 router 學習沒有幫助。

        保留條件：np.var(scores[i, :]) > min_var

        Args:
            min_var: 變異數下限（預設 0.0，只去除完全相同的樣本）
                     二元分數常用 0.05；連續分數可用更小的值
        """
        variances = np.var(self.train_score, axis=1)   # (N_train,)
        keep_idx = np.where(variances > min_var)[0]
        n_removed = len(self.train_score) - len(keep_idx)
        print(f"[filter_by_variance] min_var={min_var}：保留 {len(keep_idx)} 筆，移除 {n_removed} 筆")
        return self._select_train(keep_idx)

    def deduplicate_train(
        self,
        eps: float = 0.15,
        min_samples: int = 2,
        metric: str = "cosine",
        cluster_sample_ratio: float = 0.3,
        seed: int = 42,
        embeddings: Optional[np.ndarray] = None,
        emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        emb_batch_size: int = 32,
    ) -> "RouterData":
        """
        以 DBSCAN 對訓練集的 prompt embedding 做近似去重。
        每個 cluster 隨機保留 cluster_sample_ratio 比例的樣本（至少 1 筆）；
        DBSCAN 認定的雜訊點全部保留。

        Args:
            eps: DBSCAN 半徑（預設 0.15）
            min_samples: 形成 cluster 的最少樣本數（預設 2）
            metric: 距離度量（預設 "cosine"）
            cluster_sample_ratio: 每個 cluster 保留的比例（預設 0.3）
            seed: 隨機種子
            embeddings: 若已有訓練集 embedding 可直接傳入，省略重新計算
            emb_model: 嵌入模型名稱（embeddings=None 時使用）
            emb_batch_size: 嵌入計算 batch size
        """
        from sklearn.cluster import DBSCAN
        from ._embeddings import get_embeddings

        if embeddings is None:
            print(f"[deduplicate_train] 計算 embedding（{len(self.train_prompt)} 筆）...")
            embeddings = get_embeddings(self.train_prompt, emb_model, emb_batch_size)

        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, algorithm="brute")
        labels = db.fit_predict(embeddings)

        rng = np.random.default_rng(seed)
        cluster_to_indices: dict = {}
        keep_idx = []

        for i, label in enumerate(labels):
            if label == -1:  # 雜訊點：直接保留
                keep_idx.append(i)
            else:
                cluster_to_indices.setdefault(label, []).append(i)

        total_in_clusters = 0
        total_retained = 0
        for cluster_indices in cluster_to_indices.values():
            n_cluster = len(cluster_indices)
            n_keep = max(1, int(n_cluster * cluster_sample_ratio))
            chosen = rng.choice(cluster_indices, size=n_keep, replace=False)
            keep_idx.extend(chosen.tolist())
            total_in_clusters += n_cluster
            total_retained += n_keep

        keep_idx = np.array(keep_idx)
        n_removed = len(self.train_prompt) - len(keep_idx)
        n_clusters = len(cluster_to_indices)
        n_noise = int(np.sum(labels == -1))
        print(
            f"[deduplicate_train] eps={eps}，cluster_sample_ratio={cluster_sample_ratio}："
            f"保留 {len(keep_idx)} 筆，移除 {n_removed} 筆重複"
            f"（{n_clusters} clusters 共 {total_in_clusters} 筆保留 {total_retained} 筆，{n_noise} 雜訊點）"
        )
        return self._select_train(keep_idx)

    def _select_train(self, idx: np.ndarray) -> "RouterData":
        """以 index 陣列選取訓練集子集，val / test 不變。"""
        return RouterData(
            train_prompt=[self.train_prompt[i] for i in idx],
            val_prompt=self.val_prompt,
            test_prompt=self.test_prompt,
            train_score=self.train_score[idx],
            val_score=self.val_score,
            test_score=self.test_score,
            test_tokens=self.test_tokens,
            test_time=self.test_time,
            models=self.models,
            train_embed=self.train_embed[idx] if self.train_embed is not None else None,
            val_embed=self.val_embed,
            test_embed=self.test_embed,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RouterData":
        """從 .npz 檔案載入。"""
        d = np.load(path, allow_pickle=True)
        return cls(
            train_prompt=d["train_prompt"].tolist(),
            val_prompt=d["val_prompt"].tolist(),
            test_prompt=d["test_prompt"].tolist(),
            train_score=d["train_score"],
            val_score=d["val_score"],
            test_score=d["test_score"],
            test_tokens=d["test_tokens"],
            test_time=d["test_time"],
            models=d["model"].tolist(),
            train_embed=d["train_embed"] if "train_embed" in d else None,
            val_embed=d["val_embed"] if "val_embed" in d else None,
            test_embed=d["test_embed"] if "test_embed" in d else None,
        )


class DataPreparer:
    """
    將 DatasetManager.extract() 的平坦記錄列表轉換為 RouterData。

    Usage::

        preparer = DataPreparer()

        # 從 DatasetManager 直接準備（含 token / time 資訊）
        data = preparer.from_manager(
            mgr, datasets=["squad2_dev"], models=["gpt-4o", "gpt-4o-mini"], strategies="llm"
        )
        data.save("train_data.npz")

        # 或從已有的 extract() 結果準備
        records = mgr.extract(["squad2_dev"], ["gpt-4o", "gpt-4o-mini"], strategies="llm")
        data = preparer.prepare(records)
    """

    def prepare(
        self,
        records: List[dict],
        *,
        tokens_by_key_model: Optional[Dict[Tuple[str, str], float]] = None,
        times_by_key_model: Optional[Dict[Tuple[str, str], float]] = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        seed: int = 42,
        emb_model: Optional[str] = None,
        emb_batch_size: int = 32,
    ) -> RouterData:
        """
        從平坦記錄列表（DatasetManager.extract() 輸出）建立 RouterData。

        每筆 record 需包含欄位：key, question, model, score。
        score 應由 DatasetManager.extract() 透過 scorer 預先計算，
        直接使用 r["score"]。

        Args:
            records: DatasetManager.extract() 的輸出
            tokens_by_key_model: (key, model) → token count 的 dict（可選）
            times_by_key_model: (key, model) → response time 的 dict（可選）
            train_ratio: 訓練集比例（預設 0.6）
            val_ratio: 驗證集比例（預設 0.2）；剩餘皆為測試集（預設 0.2）
            seed: 資料分割的隨機種子
        """
        if not records:
            raise ValueError("records 為空，無法建立 RouterData")

        key_to_question: Dict[str, str] = {}
        scores_map: Dict[Tuple[str, str], float] = {}
        models_set: set = set()

        for r in records:
            key = r["key"]
            key_to_question[key] = r["question"]
            models_set.add(r["model"])
            scores_map[(key, r["model"])] = float(r.get("score", 0.0))

        keys = sorted(key_to_question.keys())
        models = sorted(models_set)
        n, m = len(keys), len(models)

        scores = np.zeros((n, m), dtype=np.float32)
        tokens = np.zeros((n, m), dtype=np.float32)
        times = np.zeros((n, m), dtype=np.float32)

        for i, key in enumerate(keys):
            for j, model in enumerate(models):
                scores[i, j] = scores_map.get((key, model), 0.0)
                if tokens_by_key_model:
                    tokens[i, j] = tokens_by_key_model.get((key, model), 0.0)
                if times_by_key_model:
                    times[i, j] = times_by_key_model.get((key, model), 0.0)

        if train_ratio + val_ratio >= 1.0:
            raise ValueError(f"train_ratio + val_ratio = {train_ratio + val_ratio:.2f} ≥ 1.0，test set 會是空的")

        prompts = [key_to_question[k] for k in keys]
        idx = np.arange(n)

        # 先切出 train
        train_idx, temp_idx = train_test_split(idx, train_size=train_ratio, random_state=seed)
        # 再從剩餘切出 val（val 佔 total 的 val_ratio）
        val_frac_of_temp = val_ratio / (1.0 - train_ratio)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_frac_of_temp, random_state=seed)

        train_embed = val_embed = test_embed = None
        if emb_model:
            from ._embeddings import get_embeddings
            print(f"[DataPreparer] 計算嵌入（model={emb_model}，共 {n} 筆）...")
            all_embed = get_embeddings(prompts, emb_model, emb_batch_size)
            train_embed = all_embed[train_idx]
            val_embed   = all_embed[val_idx]
            test_embed  = all_embed[test_idx]

        return RouterData(
            train_prompt=[prompts[i] for i in train_idx],
            val_prompt=[prompts[i] for i in val_idx],
            test_prompt=[prompts[i] for i in test_idx],
            train_score=scores[train_idx],
            val_score=scores[val_idx],
            test_score=scores[test_idx],
            test_tokens=tokens[test_idx],
            test_time=times[test_idx],
            models=models,
            train_embed=train_embed,
            val_embed=val_embed,
            test_embed=test_embed,
        )

    def from_manager(
        self,
        mgr: "DatasetManager",
        datasets: List[str],
        models: List[str],
        strategies: "str | List[str]",
        scorer=None,
        emb_model: Optional[str] = None,
        emb_batch_size: int = 32,
        **kwargs,
    ) -> RouterData:
        """
        直接從 DatasetManager 準備資料（含 token / time 資訊）。

        Args:
            mgr: DatasetManager 實例
            datasets: dataset 名稱清單
            models: model 名稱清單
            strategies: annotation strategy 名稱或清單。
                        多個 strategy 時，第一個為主要（決定哪些 key 納入）。
            scorer: 評分策略（BaseScorer 實例，預設 FieldScorer("point")）
            **kwargs: 傳入 prepare() 的額外參數（train_ratio, val_ratio, seed）
        """
        records = mgr.extract(datasets, models, strategies, scorer=scorer)

        tokens_map: Dict[Tuple[str, str], float] = {}
        times_map: Dict[Tuple[str, str], float] = {}

        for dataset in datasets:
            for model in models:
                try:
                    for r in mgr.get_responses(dataset, model):
                        key = r["key"]
                        tokens_map[(key, model)] = float(
                            r.get("usage", {}).get("total_tokens", 0)
                        )
                        times_map[(key, model)] = float(r.get("response_time", 0))
                except FileNotFoundError:
                    pass

        return self.prepare(
            records,
            tokens_by_key_model=tokens_map,
            times_by_key_model=times_map,
            emb_model=emb_model,
            emb_batch_size=emb_batch_size,
            **kwargs,
        )
