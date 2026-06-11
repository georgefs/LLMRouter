#!/usr/bin/env bash
# =============================================================================
# 01_full_workflow 設定
# 修改此檔案後執行 ./run.sh
# 所有變數均可在執行前以環境變數覆蓋，例如：
#   DATASETS=my_dataset bash run.sh
# =============================================================================

# LLMRouter config.yaml 路徑（相對於 repo 根目錄，或絕對路徑）
export CONFIG="${CONFIG:-config.yaml}"

# 評測資料集（逗號分隔）
export DATASETS="${DATASETS:-mmlu_pro_test}"

# 候選模型（逗號分隔）
export MODELS="${MODELS:-gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B}"

# LLM-as-Judge 模型
export JUDGE="${JUDGE:-gpt-oss-120b}"

# Inference / annotation 並發數
export CONCURRENCY="${CONCURRENCY:-16}"

# Embedding 模型（router prepare 時預存，加速後續 bench）
export EMB_MODEL="${EMB_MODEL:-mixedbread-ai/mxbai-embed-large-v1}"

# router bench 參數
export FRACTIONS="${FRACTIONS:-0.1,0.3,0.5,0.7,1.0}"
export REPEATS="${REPEATS:-3}"
export ROUTERS="${ROUTERS:-oracle,random,knn}"

# 輸出檔案（預設存在此情景資料夾內）
# 由 run.sh 在 source 後以 SCENARIO_DIR 解析為絕對路徑
export DATA_NPZ="${DATA_NPZ:-data.npz}"
export BEST_ROUTER_PKL="${BEST_ROUTER_PKL:-best_router.pkl}"
export ANALYSIS_CSV="${ANALYSIS_CSV:-dataset_analysis.csv}"
