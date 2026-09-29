#!/usr/bin/env bash
# =============================================================================
# 01_full_workflow 設定
# 修改此檔案後執行 ./run.sh
# 所有變數均可在執行前以環境變數覆蓋，例如：
#   DATASETS=my_dataset bash run.sh
# =============================================================================

# LLMRouter 資料根目錄
export DATA_PATH="${DATA_PATH:-/home/test123/work/tmp/LLMRouter2/datasets}"

# LLMRouter config.yaml 路徑（相對於 repo 根目錄，或絕對路徑）
export CONFIG="${CONFIG:-config.yaml}"

# 評測資料集（逗號分隔）
#export DATASETS="${DATASETS:-hellaswag_train}"
export DATASETS="${DATASETS:-mmlu_pro_test,hellaswag_train}"

# 候選模型（逗號分隔）
export MODELS="${MODELS:-Google-Gemma-3-27B,gpt-oss-20b,Llama-4-Maverick-17B-128E-Instruct-FP8,Llama-4-Scout-17B-16E-Instruct-FP8,Microsoft-Phi-4,Mistral-Small-3.1-24B-Instruct-2503}"

# LLM-as-Judge 模型
export JUDGE="${JUDGE:-gpt-oss-120b}"

# Inference / annotation 並發數
export CONCURRENCY="${CONCURRENCY:-32}"

# Embedding 模型（router prepare 時預存，加速後續 bench）
export EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# router bench 參數
export FRACTIONS="${FRACTIONS:-1.0}"
export REPEATS="${REPEATS:-1}"
export ROUTERS="${ROUTERS:-oracle,random,knn,sft_grpo}"

# 輸出檔案（預設存在此情景資料夾內）
# 由 run.sh 在 source 後以 SCENARIO_DIR 解析為絕對路徑
export DATA_NPZ="${DATA_NPZ:-data.npz}"
export BEST_ROUTER_PKL="${BEST_ROUTER_PKL:-best_router.pkl}"
export ANALYSIS_CSV="${ANALYSIS_CSV:-dataset_analysis.csv}"
