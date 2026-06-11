#!/usr/bin/env bash
# =============================================================================
# 04_semantic_api_router 設定
# 修改此檔案後執行 ./run.sh
# =============================================================================

# RouterData .npz（由 01_full_workflow 產生）
export DATA_NPZ="${DATA_NPZ:-../01_full_workflow/data.npz}"

# semantic router 的 API 位址
# 若未啟動，run.sh 會自動起一個 mock server 供驗證用
export SR_BASE_URL="${SR_BASE_URL:-http://localhost:8080}"
export SR_TIMEOUT="${SR_TIMEOUT:-10}"

# SemanticAPIRouter checkpoint 儲存路徑（預設在此情景資料夾內）
export SR_ROUTER_PKL="${SR_ROUTER_PKL:-semantic_api_router.pkl}"
