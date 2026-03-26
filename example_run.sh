#!/usr/bin/env bash
# =============================================================================
# LLMRouter 完整流程：dataset → response → annotation → prepare → bench
# =============================================================================
set -euo pipefail

# ── 環境 ──────────────────────────────────────────────────────────────────────
source "$(dirname "$0")/routebench.env"
CONFIG="$(dirname "$0")/config.yaml"

# ── 參數設定 ──────────────────────────────────────────────────────────────────

# 評測的 dataset（逗號分隔可同時跑多個）
DATASETS="hellaswag_train"

# 要比較的模型（逗號分隔）
MODELS="gpt-oss-20b,Google-Gemma-3-27B"

# LLM-as-Judge 使用的 judge 模型
JUDGE="gpt-oss-120b"

# 輸出路徑
DATA_NPZ="data.npz"

# bench 參數
FRACTIONS="0.1,0.3,0.5,0.7,1.0"
REPEATS=3
ROUTERS="knn,oracle,random"

# 並發數
CONCURRENCY=16

# =============================================================================
# Step 1：檢查 dataset
# =============================================================================
echo "================================================================"
echo "Step 1 / 6  ─  查看 dataset"
echo "================================================================"
python3 -m LLMRouter --config "$CONFIG" dataset list

# =============================================================================
# Step 2：對每個 model 產生 response
# =============================================================================
echo ""
echo "================================================================"
echo "Step 2 / 6  ─  產生 model response（斷點續跑）"
echo "================================================================"
for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → response gen  ds=$DS  model=$MODEL"
        python3 -m LLMRouter --config "$CONFIG" \
            response gen "$DS" "$MODEL" \
            --concurrency "$CONCURRENCY"
    done
done

# =============================================================================
# Step 3：LLM-as-Judge annotation（strategy=llm）
# =============================================================================
echo ""
echo "================================================================"
echo "Step 3 / 6  ─  LLM-as-Judge annotation（斷點續跑）"
echo "================================================================"
for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → annotation gen  ds=$DS  model=$MODEL  judge=$JUDGE"
        python3 -m LLMRouter --config "$CONFIG" \
            annotation gen "$DS" "$MODEL" \
            --strategy llm \
            --judge "$JUDGE" \
            --concurrency "$CONCURRENCY"
    done
done

# =============================================================================
# Step 4：查看目前資料狀況
# =============================================================================
echo ""
echo "================================================================"
echo "Step 4 / 6  ─  列出 annotation 狀態"
echo "================================================================"
for DS in ${DATASETS//,/ }; do
    echo "  dataset: $DS"
    python3 -m LLMRouter --config "$CONFIG" annotation list "$DS"
done

# =============================================================================
# Step 5：準備 RouterData（三層 join → .npz）
# =============================================================================
echo ""
echo "================================================================"
echo "Step 5 / 6  ─  router prepare → $DATA_NPZ"
echo "================================================================"
python3 -m LLMRouter --config "$CONFIG" \
    router prepare \
    --datasets "$DATASETS" \
    --models   "$MODELS" \
    --strategy llm \
    --scorer   point \
    --train-ratio 0.6 \
    --val-ratio   0.1 \
    -o "$DATA_NPZ"

# （選用）訓練集預處理：去除無鑑別度樣本 + 語義去重
# python3 -m LLMRouter --config "$CONFIG" \
#     router prepare \
#     --datasets "$DATASETS" --models "$MODELS" \
#     --strategy llm --scorer point \
#     --min-var 0.05 --dedup-eps 0.3 \
#     -o data_clean.npz

# =============================================================================
# Step 6：Router bench（縮放實驗 + 橫向對比）
# =============================================================================
echo ""
echo "================================================================"
echo "Step 6 / 6  ─  router bench（routers=$ROUTERS）"
echo "================================================================"
python3 -m LLMRouter \
    router bench "$ROUTERS" \
    --data       "$DATA_NPZ" \
    --fractions  "$FRACTIONS" \
    --repeats    "$REPEATS" \
    --k          10

echo ""
echo "================================================================"
echo "完成。結果說明："
echo "  mu  = 選出模型的平均分數（越高越好）"
echo "  vb  = mu / oracle（越接近 1 越好）"
echo "  ep  = 預測分佈 entropy（bits，越高表示越分散）"
echo "================================================================"
