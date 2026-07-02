#!/usr/bin/env bash
# =============================================================================
# 02_dataset_analysis 設定
# 修改此檔案後執行 ./run.sh
# =============================================================================

# LLMRouter 資料根目錄
export DATA_PATH="${DATA_PATH:-/home/george/work2/delta/test/LLMRouter/datasets}"

# 要分析的資料集（逗號分隔）
export DATASETS="${DATASETS:-arc_challenge_train}"

# 候選模型（逗號分隔）
export MODELS="${MODELS:-gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B}"

# annotation strategy（決定如何讀取分數）
export STRATEGY="${STRATEGY:-llm}"

# Embedding 模型（用於計算 CH Score / Avg_Sim）
export EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# 輸出檔案（預設存在此情景資料夾內）
export OUTPUT_CSV="${OUTPUT_CSV:-analysis_results.csv}"
export OUTPUT_JSON="${OUTPUT_JSON:-analysis_results.json}"
