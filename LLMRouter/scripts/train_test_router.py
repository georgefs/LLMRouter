#!/usr/bin/env python3
"""訓練並保存測試用的 Router

執行一次，生成固定的測試 router (.pkl 文件)
用於後續的集成測試
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from LLMRouter.router import RouterData, KNNRouter

def train_test_router():
    """訓練測試 router"""
    print("=" * 60)
    print("Training Test Router for Integration Tests")
    print("=" * 60)

    # 建立訓練資料
    n_samples = 100
    n_models = 3
    embedding_dim = 384  # 與 all-MiniLM-L6-v2 一致

    print(f"\n準備訓練資料:")
    print(f"  樣本數: {n_samples}")
    print(f"  模型數: {n_models}")
    print(f"  Embedding 維度: {embedding_dim}")

    prompts = [f"query_{i}" for i in range(n_samples)]
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    scores = np.random.rand(n_samples, n_models).astype(np.float32)

    data = RouterData(
        train_prompt=prompts[:80],
        val_prompt=prompts[80:90],
        test_prompt=prompts[90:],
        train_score=scores[:80],
        val_score=scores[80:90],
        test_score=scores[90:],
        test_tokens=np.zeros((10, n_models)),
        test_time=np.zeros((10, n_models)),
        models=["gpt-4", "gpt-3.5-turbo", "claude-3"],
        train_embed=embeddings[:80],
        val_embed=embeddings[80:90],
        test_embed=embeddings[90:],
    )

    # 訓練 router
    print(f"\n訓練 KNNRouter (k=5)...")
    router = KNNRouter(k=5)
    router.fit(data)

    print(f"✓ Router 訓練完成")
    print(f"  Model names: {router.model_names}")

    # 保存 router
    output_dir = Path(__file__).parent.parent / "test_data"
    output_dir.mkdir(exist_ok=True)

    router_path = output_dir / "test_router.pkl"
    print(f"\n保存 router 到: {router_path}")
    router.save(router_path)

    print(f"✓ Router 已保存")
    print(f"\n" + "=" * 60)
    print(f"測試 Router 位置: {router_path}")
    print(f"模型列表: {router.model_names}")
    print("=" * 60)

    return router_path


if __name__ == "__main__":
    train_test_router()
