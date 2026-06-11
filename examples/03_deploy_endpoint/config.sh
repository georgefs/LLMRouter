#!/usr/bin/env bash
# =============================================================================
# 03_deploy_endpoint 設定
# 修改此檔案後執行 ./run.sh
# =============================================================================

# RouterData .npz（由 01_full_workflow 產生）
export DATA_NPZ="${DATA_NPZ:-../01_full_workflow/data.npz}"

# 訓練好的 router 儲存路徑（預設在此情景資料夾內）
export ROUTER_PKL="${ROUTER_PKL:-router.pkl}"

# LLMRouter Endpoint 設定
export ENDPOINT_PORT="${ENDPOINT_PORT:-8888}"
export ENDPOINT_HOST="${ENDPOINT_HOST:-0.0.0.0}"

# Docker 宿主機 IP（semantic router 容器連回宿主機用）
# 留空則由 run.sh 自動偵測（docker0 介面 → 預設路由 gateway → host.docker.internal）
export DOCKER_HOST_IP="${DOCKER_HOST_IP:-}"
