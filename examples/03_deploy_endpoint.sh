#!/usr/bin/env bash
# =============================================================================
# Example 3 — 訓練 Router → 部署 HTTP Endpoint → 串接 semantic router
#
# 前提：已有 RouterData .npz（執行 01_full_workflow.sh Step 1-6 後取得）
#
# 流程：
#   Step 1  訓練 router 並儲存 .pkl
#   Step 2  驗證 checkpoint 內容
#   Step 3  啟動 LLMRouter Endpoint（背景執行）
#   Step 4  健康檢查 + API 功能驗證（curl + Python）
#   Step 5  產生 semantic router config（最小化）
#   Step 6  說明如何啟動 semantic router 並串接
#   Step 7  清理背景 endpoint
#
# 輸出：
#   $ROUTER_PKL              — 已訓練的 router
#   semantic_router.yaml     — 可直接用於 semantic router 的設定檔
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 參數設定
# ─────────────────────────────────────────────────────────────────────────────

DATA_NPZ="${DATA_NPZ:-data.npz}"
ROUTER_PKL="${ROUTER_PKL:-deployed_router.pkl}"
ENDPOINT_PORT="${ENDPOINT_PORT:-8888}"
ENDPOINT_HOST="${ENDPOINT_HOST:-0.0.0.0}"
SEMANTIC_CONFIG_OUT="semantic_router.yaml"

# ── 偵測 Docker 容器可到達的宿主機 IP ──────────────────────────────────────
# semantic router 預設以 Docker 部署，容器內無法用 localhost 連到宿主機 port。
# 優先序：環境變數 DOCKER_HOST_IP > docker0 介面 > 預設路由 gateway
if [ -z "${DOCKER_HOST_IP:-}" ]; then
    # Linux Docker bridge（docker0）
    DOCKER_HOST_IP=$(ip addr show docker0 2>/dev/null \
                     | grep -oP 'inet \K[\d.]+' | head -1)
fi
if [ -z "${DOCKER_HOST_IP:-}" ]; then
    # 若使用自訂 network，fallback 到預設路由的 gateway
    DOCKER_HOST_IP=$(ip route show default 2>/dev/null \
                     | awk '/default/ {print $3}' | head -1)
fi
if [ -z "${DOCKER_HOST_IP:-}" ]; then
    # Docker Desktop (Mac/Windows) 的特殊 DNS
    DOCKER_HOST_IP="host.docker.internal"
fi

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }
die()  { echo "  ✗ $1" >&2; exit 1; }

# endpoint server PID（清理用）
ENDPOINT_PID=""

cleanup() {
    if [ -n "$ENDPOINT_PID" ] && kill -0 "$ENDPOINT_PID" 2>/dev/null; then
        echo ""
        echo "  [cleanup] 停止 endpoint (PID $ENDPOINT_PID)..."
        kill "$ENDPOINT_PID"
    fi
}
trap cleanup EXIT

# =============================================================================
# Step 1：確認 RouterData 並訓練 router
# =============================================================================
step "Step 1 / 7  訓練 KNN router（100% 訓練資料）"

[ -f "$DATA_NPZ" ] || die "RouterData 不存在：$DATA_NPZ\n請先執行 01_full_workflow.sh。"

python3 -m LLMRouter \
    router train knn \
    --data "$DATA_NPZ" \
    --k    10 \
    -o     "$ROUTER_PKL"

[ -f "$ROUTER_PKL" ] || die "router train 失敗：$ROUTER_PKL 不存在"
ok "router 訓練完成 → $ROUTER_PKL"

# =============================================================================
# Step 2：驗證 checkpoint 內容
# =============================================================================
step "Step 2 / 7  驗證 checkpoint 內容"

python3 - <<PYEOF
import pickle, sys, numpy as np

with open("$ROUTER_PKL", "rb") as f:
    ck = pickle.load(f)

# 必要欄位
for key in ("router_type", "model_names"):
    if key not in ck:
        print(f"  [ERR] checkpoint 缺少欄位：{key}", file=sys.stderr)
        sys.exit(1)

rtype = ck["router_type"]
models = ck["model_names"]
print(f"  router_type  : {rtype}")
print(f"  model_names  : {models}")

# 確認 KNN 有訓練集 embedding
if rtype == "knn":
    has_X = "X_train" in ck or "_X_train" in ck
    print(f"  embedding    : {'已預存' if has_X else '推論時計算'}")

print("  checkpoint 驗證通過")
PYEOF

ok "checkpoint 格式正確"

# =============================================================================
# Step 3：啟動 LLMRouter Endpoint（背景執行）
# =============================================================================
step "Step 3 / 7  啟動 LLMRouter Endpoint（port $ENDPOINT_PORT）"

# 確認 port 未被佔用
if lsof -i ":$ENDPOINT_PORT" -t >/dev/null 2>&1; then
    die "Port $ENDPOINT_PORT 已被佔用。請更改 ENDPOINT_PORT 後重試。"
fi

python3 -m LLMRouter.scripts.start_endpoint \
    --router "$ROUTER_PKL" \
    --port   "$ENDPOINT_PORT" \
    --host   "$ENDPOINT_HOST" &
ENDPOINT_PID=$!

echo "  PID: $ENDPOINT_PID"
echo "  等待 server 啟動（最多 15 秒）..."

# 等待 /health 回應
for i in $(seq 1 15); do
    sleep 1
    if curl -sf "http://localhost:$ENDPOINT_PORT/health" >/dev/null 2>&1; then
        ok "server 已啟動（${i}s）"
        break
    fi
    if ! kill -0 "$ENDPOINT_PID" 2>/dev/null; then
        die "endpoint 啟動失敗（process 已結束）"
    fi
done

if ! curl -sf "http://localhost:$ENDPOINT_PORT/health" >/dev/null 2>&1; then
    die "server 未回應，啟動逾時"
fi

# =============================================================================
# Step 4：API 功能驗證
# =============================================================================
step "Step 4 / 7  API 功能驗證"

BASE="http://localhost:$ENDPOINT_PORT"

# 4-1. GET /health
echo "  ─ GET /health"
HEALTH=$(curl -sf "$BASE/health")
echo "  $HEALTH"
echo "$HEALTH" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert d['status'] == 'healthy', f'status 不是 healthy: {d}'
assert 'router_type' in d, 'health 回應缺少 router_type'
assert d.get('model_count', 0) > 0, 'model_count = 0'
print('  ✓ /health 驗證通過')
"

# 4-2. GET /models
echo ""
echo "  ─ GET /models"
MODELS_RESP=$(curl -sf "$BASE/models")
echo "  $MODELS_RESP"
echo "$MODELS_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'models' in d, '回應缺少 models 欄位'
names = [m['name'] for m in d['models']]
assert len(names) > 0, 'models 清單為空'
print(f'  ✓ /models 驗證通過：{names}')
"

# 4-3. POST /route — 正常查詢
echo ""
echo "  ─ POST /route（正常查詢）"
ROUTE_RESP=$(curl -sf -X POST "$BASE/route" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is the capital of France?"}')
echo "  $ROUTE_RESP"
echo "$ROUTE_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
assert 'selected_model' in d, '回應缺少 selected_model'
assert d['selected_model'], 'selected_model 為空'
print(f'  ✓ /route 驗證通過：selected_model={d[\"selected_model\"]}')
"

# 4-4. POST /route — 一致性：同一 query 多次應回傳同一結果
echo ""
echo "  ─ POST /route（一致性，5 次）"
python3 - <<PYEOF
import json, urllib.request

results = []
for _ in range(5):
    req = urllib.request.Request(
        "http://localhost:$ENDPOINT_PORT/route",
        data=json.dumps({"query": "Solve: 2 + 2 = ?"}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        results.append(json.loads(r.read())["selected_model"])

if len(set(results)) == 1:
    print(f"  ✓ 一致性驗證通過：5 次均回傳 {results[0]}")
else:
    print(f"  ⚠ 結果不一致（正常，因為部分 router 有隨機性）：{set(results)}")
PYEOF

# 4-5. POST /route — 邊界：空 query 應回 400
echo ""
echo "  ─ POST /route（空 query → 預期 400）"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$BASE/route" \
    -H "Content-Type: application/json" \
    -d '{"query": ""}')
if [ "$HTTP_CODE" = "400" ]; then
    ok "空 query 正確回傳 400"
else
    warn "空 query 回傳 $HTTP_CODE（預期 400）"
fi

ok "所有 API 驗證通過"

# =============================================================================
# Step 5：產生 semantic router config（最小化）
# =============================================================================
step "Step 5 / 7  產生 semantic router config"

echo "  Docker 宿主機 IP：$DOCKER_HOST_IP"
echo "  （semantic router 以 Docker 部署，需用此 IP 連回宿主機的 endpoint）"
echo "  如需覆蓋，請設定環境變數：export DOCKER_HOST_IP=<your-host-ip>"

# 從 /health 取得 model 清單
MODEL_NAMES=$(python3 - <<PYEOF
import json, urllib.request
with urllib.request.urlopen("http://localhost:$ENDPOINT_PORT/models") as r:
    d = json.load(r)
print(",".join(m["name"] for m in d["models"]))
PYEOF
)

echo "  model 清單：$MODEL_NAMES"

python3 - <<PYEOF
import sys

endpoint_port   = $ENDPOINT_PORT
docker_host_ip  = "$DOCKER_HOST_IP"
model_names     = "$MODEL_NAMES".split(",")
router_url      = f"http://{docker_host_ip}:{endpoint_port}"

# 產生 YAML model block
def model_block(name):
    return f"""\
    - name: {name}
      provider_model_id: {name}
      api_format: openai
      backend_refs:
        - name: backend-{name}
          base_url: \${{BACKEND_BASE_URL:-https://api.openai.com/v1}}
          provider: openai
          api_key_env: OPENAI_API_KEY
          weight: 100"""

model_blocks = "\n".join(model_block(n) for n in model_names)

# routing modelCards
model_cards = "\n".join(f"    - name: {n}" for n in model_names)

yaml = f"""# =============================================================================
# semantic router 最小化設定（自動由 03_deploy_endpoint.sh 生成）
#
# 用途：串接 LLMRouter Endpoint 作為 r1 router 進行智慧路由
#
# 啟動方式：
#   semantic-router start --config semantic_router.yaml
#
# 注意：router_r1_server_url 使用宿主機 IP（{docker_host_ip}），
#   而非 localhost。原因：semantic router 以 Docker 部署，容器內的
#   localhost 指向容器自身，無法連到宿主機上的 LLMRouter Endpoint。
#
#   如需修改，請設定環境變數後重新產生：
#     export DOCKER_HOST_IP=<your-host-ip>
#     bash examples/03_deploy_endpoint.sh
#
# 前提：
#   LLMRouter Endpoint 已在宿主機 port {endpoint_port} 執行
# =============================================================================

version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

# ── 模型定義 ──────────────────────────────────────────────────────────────────
# 請將 BACKEND_BASE_URL / OPENAI_API_KEY 替換為實際值，
# 或設定同名環境變數。

providers:
  defaults:
    default_model: {model_names[0]}

  models:
{model_blocks}

# ── 路由設定 ──────────────────────────────────────────────────────────────────

routing:
  modelCards:
{model_cards}

  decisions:
    - name: default
      description: "Route all requests via LLMRouter"
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
{chr(10).join(f"        - name: {n}" for n in model_names)}

  rl_driven:
    enabled: true
    # 宿主機 IP（Docker 容器內需用此位址連回宿主機）
    # 若使用自訂 Docker network，可能需要調整此 IP
    router_r1_server_url: {router_url}
    # 若 LLMRouter 不可用，降級為 Thompson Sampling
    llm_routing_fallback: thompson
    # 每次請求 LLMRouter 的 timeout（秒）
    router_r1_timeout: 5
"""

with open("semantic_router.yaml", "w") as f:
    f.write(yaml)

print("  已產生 semantic_router.yaml")
print(f"  模型數：{len(model_names)}")
print(f"  router_r1_server_url：{router_url}")
PYEOF

ok "semantic_router.yaml 已產生"

# =============================================================================
# Step 6：說明串接流程
# =============================================================================
step "Step 6 / 7  串接 semantic router（說明）"

cat <<INFO
  ─────────────────────────────────────────────────────────
  完整串接流程（兩個 terminal）：

  Terminal 1 — LLMRouter Endpoint（已在背景執行，監聽宿主機 port）：
    python3 -m LLMRouter.scripts.start_endpoint \\
        --router deployed_router.pkl \\
        --port   $ENDPOINT_PORT

  Terminal 2 — semantic router（Docker 部署）：
    semantic-router start --config semantic_router.yaml
    # semantic_router.yaml 中 router_r1_server_url 已設為
    # http://$DOCKER_HOST_IP:$ENDPOINT_PORT
    # Docker 容器透過此 IP 連回宿主機的 LLMRouter Endpoint

  驗證串接（Terminal 3）：
    # 健康檢查
    curl http://localhost:8899/health

    # 送出查詢（semantic router 會自動呼叫 LLMRouter → 選擇模型）
    curl -X POST http://localhost:8899/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "auto",
           "messages": [{"role": "user", "content": "What is 2+2?"}]}'

  錯誤降級：
    若 LLMRouter Endpoint 不可用，semantic router 會自動
    降級為 Thompson Sampling（llm_routing_fallback: thompson）。
  ─────────────────────────────────────────────────────────

  注意事項：
    1. router_r1_server_url 必須是宿主機 IP，不能用 localhost
       目前偵測到：$DOCKER_HOST_IP
       如需手動指定：export DOCKER_HOST_IP=<ip> 後重新執行

    2. 若 semantic router 使用自訂 Docker network（非預設 bridge），
       宿主機 IP 可能不同，請用以下指令確認：
         docker network inspect <network-name> | grep Gateway

    3. BACKEND_BASE_URL / OPENAI_API_KEY 請替換為實際值

    4. 持續運行 endpoint（不受 script 結束影響）：
         nohup python3 -m LLMRouter.scripts.start_endpoint \\
             --router deployed_router.pkl --port $ENDPOINT_PORT \\
             > endpoint.log 2>&1 &
         echo "PID: \$!"
INFO

# =============================================================================
# Step 7：清理
# =============================================================================
step "Step 7 / 7  清理（停止背景 endpoint）"

if [ -n "$ENDPOINT_PID" ] && kill -0 "$ENDPOINT_PID" 2>/dev/null; then
    kill "$ENDPOINT_PID"
    wait "$ENDPOINT_PID" 2>/dev/null || true
    ENDPOINT_PID=""
    ok "endpoint 已停止"
else
    ok "endpoint 已停止（或未啟動）"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "  所有步驟完成！"
echo ""
echo "  產出物："
echo "    $ROUTER_PKL          — 已訓練 router"
echo "    semantic_router.yaml — semantic router 設定檔"
echo ""
echo "  部署指令（需在背景持續執行）："
echo "    python3 -m LLMRouter.scripts.start_endpoint \\"
echo "        --router $ROUTER_PKL --port $ENDPOINT_PORT"
echo ""
echo "  下一步："
echo "    用 SemanticAPIRouter 把 semantic router 加入評估："
echo "    → 參考 examples/04_semantic_api_router.sh"
echo "════════════════════════════════════════════════════"
