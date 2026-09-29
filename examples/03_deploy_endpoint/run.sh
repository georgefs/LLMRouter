#!/usr/bin/env bash
# =============================================================================
# Example 3 — 訓練 Router → 部署 HTTP Endpoint → 串接 semantic router
#
# 前提：已有 RouterData .npz（執行 01_full_workflow 後取得）
#
# 用法：
#   cd examples/03_deploy_endpoint
#   vim config.sh          # 確認 DATA_NPZ 路徑、ENDPOINT_PORT
#   bash run.sh
#
# 產出：
#   router.pkl             — 已訓練的 router（此情景資料夾內）
#   sr_resolved.yaml       — 展開變數後的 semantic router 設定檔
# =============================================================================
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCENARIO_DIR/config.sh"

# 解析相對路徑
[[ "$DATA_NPZ"    != /* ]] && DATA_NPZ="$SCENARIO_DIR/$DATA_NPZ"
[[ "$ROUTER_PKL"  != /* ]] && ROUTER_PKL="$SCENARIO_DIR/$ROUTER_PKL"

SR_TEMPLATE="$SCENARIO_DIR/semantic_router.yaml"
SR_RESOLVED="$SCENARIO_DIR/sr_resolved.yaml"

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }
die()  { echo "  ✗ $1" >&2; exit 1; }

# ── 自動偵測 Docker 宿主機 IP ─────────────────────────────────────────────
if [ -z "$DOCKER_HOST_IP" ]; then
    DOCKER_HOST_IP=$(ip addr show docker0 2>/dev/null \
                     | grep -oP 'inet \K[\d.]+' | head -1)
fi
if [ -z "$DOCKER_HOST_IP" ]; then
    DOCKER_HOST_IP=$(ip route show default 2>/dev/null \
                     | awk '/default/ {print $3}' | head -1)
fi
if [ -z "$DOCKER_HOST_IP" ]; then
    DOCKER_HOST_IP="host.docker.internal"
fi

ENDPOINT_PID=""
cleanup() {
    [ -n "$ENDPOINT_PID" ] && kill -0 "$ENDPOINT_PID" 2>/dev/null && kill "$ENDPOINT_PID"
}
trap cleanup EXIT

echo "設定："
echo "  DATA_NPZ        = $DATA_NPZ"
echo "  ROUTER_PKL      = $ROUTER_PKL"
echo "  ENDPOINT_PORT   = $ENDPOINT_PORT"
echo "  DOCKER_HOST_IP  = $DOCKER_HOST_IP"
echo "  SR_TEMPLATE     = $SR_TEMPLATE"

# =============================================================================
step "Step 1 / 6  確認 RouterData 並訓練 router"

[ -f "$DATA_NPZ" ] || die "RouterData 不存在：$DATA_NPZ\n先執行 examples/01_full_workflow/run.sh"

python3 -m LLMRouter router train knn \
    --data "$DATA_NPZ" --k 10 -o "$ROUTER_PKL"

[ -f "$ROUTER_PKL" ] || die "router train 失敗"
ok "router → $ROUTER_PKL"

# =============================================================================
step "Step 2 / 6  驗證 checkpoint 並產生 semantic_router.yaml"

# 讀取 checkpoint 取得模型清單
MODEL_NAMES=$(python3 - <<PYEOF
import pickle
ck = pickle.load(open("$ROUTER_PKL", "rb"))
print(",".join(ck["model_names"]))
PYEOF
)

echo "  router_type  = $(python3 -c "import pickle; print(pickle.load(open('$ROUTER_PKL','rb'))['router_type'])")"
echo "  model_names  = $MODEL_NAMES"

# 產生 YAML 片段（供 envsubst 填入）
export DEFAULT_MODEL="${MODEL_NAMES%%,*}"
export DOCKER_HOST_IP
export ENDPOINT_PORT

# 建立 model blocks / cards / refs
export MODEL_BLOCKS MODEL_CARDS MODEL_REFS
eval "$(python3 - <<PYEOF
import sys
names = "$MODEL_NAMES".split(",")

blocks = "\n".join(f"""
    - name: {n}
      provider_model_id: {n}
      api_format: openai
      backend_refs:
        - name: backend-{n}
          base_url: \${{BACKEND_BASE_URL:-https://api.openai.com/v1}}
          provider: openai
          api_key_env: OPENAI_API_KEY
          weight: 100""" for n in names)

cards = "\n".join(f"    - name: {n}" for n in names)
refs  = "\n".join(f"        - name: {n}" for n in names)

import shlex
print(f"MODEL_BLOCKS={shlex.quote(blocks)}")
print(f"MODEL_CARDS={shlex.quote(cards)}")
print(f"MODEL_REFS={shlex.quote(refs)}")
PYEOF
)"

envsubst < "$SR_TEMPLATE" > "$SR_RESOLVED"
ok "semantic router 設定 → $SR_RESOLVED"
echo ""
echo "  ── sr_resolved.yaml rl_driven 區塊 ──"
grep -A5 "rl_driven:" "$SR_RESOLVED"

# =============================================================================
step "Step 3 / 6  啟動 LLMRouter Endpoint（背景）"

lsof -i ":$ENDPOINT_PORT" -t >/dev/null 2>&1 \
    && die "Port $ENDPOINT_PORT 已被佔用，請更改 ENDPOINT_PORT"

python3 -m LLMRouter.scripts.start_endpoint \
    --router "$ROUTER_PKL" --port "$ENDPOINT_PORT" --host "$ENDPOINT_HOST" &
ENDPOINT_PID=$!
echo "  PID: $ENDPOINT_PID"

for i in $(seq 1 15); do
    sleep 1
    curl -sf "http://localhost:$ENDPOINT_PORT/health" >/dev/null 2>&1 && break
    kill -0 "$ENDPOINT_PID" 2>/dev/null || die "endpoint 啟動失敗"
done
curl -sf "http://localhost:$ENDPOINT_PORT/health" >/dev/null 2>&1 \
    || die "endpoint 未回應（逾時）"
ok "endpoint 已啟動"

# =============================================================================
step "Step 4 / 6  API 驗證"

BASE="http://localhost:$ENDPOINT_PORT"

echo "  ─ GET /health"
curl -sf "$BASE/health" | python3 -c "
import json,sys; d=json.load(sys.stdin)
assert d['status']=='healthy' and d.get('model_count',0)>0
print(f'  ✓ status={d[\"status\"]}  models={d[\"model_count\"]}')"

echo "  ─ GET /models"
curl -sf "$BASE/models" | python3 -c "
import json,sys; d=json.load(sys.stdin)
names=[m['name'] for m in d['models']]
assert names; print(f'  ✓ models={names}')"

echo "  ─ POST /route"
curl -sf -X POST "$BASE/route" \
    -H "Content-Type: application/json" \
    -d '{"query":"What is the capital of France?"}' \
  | python3 -c "
import json,sys; d=json.load(sys.stdin)
assert d.get('selected_model'); print(f'  ✓ selected_model={d[\"selected_model\"]}')"

echo "  ─ POST /route（空 query → 預期 400）"
CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/route" -H "Content-Type: application/json" -d '{"query":""}')
[ "$CODE" = "400" ] && ok "空 query 回傳 400" || warn "空 query 回傳 $CODE（預期 400）"

ok "API 驗證通過"

# =============================================================================
step "Step 5 / 6  部署說明"

cat <<INFO

  ─────────────────────────────────────────────────────────
  LLMRouter Endpoint 已在 localhost:$ENDPOINT_PORT 運行
  （此 endpoint 會在 script 結束時停止；持續運行請見下方指令）

  部署 semantic router（需另開 terminal）：
    semantic-router start --config $SR_RESOLVED

  sr_resolved.yaml 關鍵設定：
    router_r1_server_url: http://$DOCKER_HOST_IP:$ENDPOINT_PORT
    （Docker 容器透過宿主機 IP 連回 LLMRouter）

  若 DOCKER_HOST_IP 不正確，更新 config.sh 後重新執行：
    export DOCKER_HOST_IP=<your-host-ip>
    bash run.sh

  驗證串接：
    curl http://localhost:8899/health
    curl -X POST http://localhost:8899/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"messages":[{"role":"user","content":"What is 2+2?"}]}'

  持續運行 endpoint（不受 script 結束影響）：
    nohup python3 -m LLMRouter.scripts.start_endpoint \\
        --router $ROUTER_PKL --port $ENDPOINT_PORT \\
        > endpoint.log 2>&1 &

  自訂 Docker network 時確認宿主機 IP：
    docker network inspect <network-name> | grep Gateway
  ─────────────────────────────────────────────────────────

INFO

# =============================================================================
step "Step 6 / 6  清理"

kill "$ENDPOINT_PID" 2>/dev/null; wait "$ENDPOINT_PID" 2>/dev/null || true
ENDPOINT_PID=""
ok "endpoint 已停止"

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "完成！"
echo "  $ROUTER_PKL"
echo "  $SR_RESOLVED   ← 交給 semantic-router start 使用"
echo "════════════════════════════════════════════════════"
