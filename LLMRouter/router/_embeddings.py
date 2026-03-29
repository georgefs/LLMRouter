"""Mean-pooled sentence embeddings（共用輔助函數）。"""
from __future__ import annotations

from typing import List

import numpy as np

# ── 模型 cache ────────────────────────────────────────────────────────────────
# key: model_name → (tokenizer, model, device)
# 同一 process 內只載入一次，bench 多次 run 共用。
_model_cache: dict = {}

# ── embedding 結果 cache ──────────────────────────────────────────────────────
# key: (model_name, id(text_list), len(text_list))
# id(text_list) 僅在物件存活期間唯一；CPython 在 GC 後可能將同一記憶體位址
# 分配給新的 list，若 len 也相同則 cache key 相同 → 回傳錯誤結果。
# 因此 cache 只對「物件生命週期橫跨所有查詢」的情境安全，
# 例如 RouterBenchmark 持有的 test_prompt（始終存活）。
# Training subset 每次重建，不應命中此 cache — 不過因為
# id 可能被重用，實際上可能誤命中。這裡改用 object identity（以 id）加上
# 第一筆 + 最後一筆文字做 fingerprint，降低誤命中機率。
_emb_cache: dict = {}


def _load_model(model_name: str):
    """載入（或從 cache 取）tokenizer + model，回傳 (tokenizer, model, device)。"""
    if model_name not in _model_cache:
        import torch
        from transformers import AutoModel, AutoTokenizer

        print(f"[embeddings] 載入模型 {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        _model_cache[model_name] = (tokenizer, model, device)
        print(f"[embeddings] 模型已載入（device={device}）")

    return _model_cache[model_name]


def get_embeddings(
    text_list: List[str],
    model_name: str = "mixedbread-ai/mxbai-embed-large-v1",
    batch_size: int = 16,
) -> np.ndarray:
    """
    使用 HuggingFace transformer 模型計算 mean-pool 嵌入向量。

    模型與 tokenizer 在 process 內只載入一次（_model_cache）。
    相同 list 物件（相同 id）的計算結果也會被 cache，避免 bench 中
    反覆對相同 test_prompt 重算。

    Args:
        text_list: 待嵌入的文字列表
        model_name: HuggingFace 模型名稱（預設 mixedbread-ai/mxbai-embed-large-v1）
        batch_size: 每批處理的文字數量

    Returns:
        (N, D) float32 ndarray
    """
    import torch

    n = len(text_list)
    # fingerprint：id + len + 首尾文字，避免 CPython id 重用造成誤命中
    fp = (text_list[0] if n > 0 else "", text_list[-1] if n > 0 else "")
    cache_key = (model_name, id(text_list), n, fp)
    if cache_key in _emb_cache:
        return _emb_cache[cache_key]

    tokenizer, model, device = _load_model(model_name)

    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**inputs)
        tok_emb = out.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
        pooled = torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        embeddings.append(pooled.cpu().numpy())

    result = np.concatenate(embeddings, axis=0).astype(np.float32)
    _emb_cache[cache_key] = result
    return result


def clear_cache() -> None:
    """清除 embedding 結果 cache（模型本身仍保留在記憶體中）。"""
    _emb_cache.clear()
