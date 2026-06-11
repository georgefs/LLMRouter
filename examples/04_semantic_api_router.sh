#!/usr/bin/env bash
# =============================================================================
# Example 4 — SemanticAPIRouter 整合 semantic router 進入評估
#
# 將正在運行的 semantic router（HTTP API）包裝成 BaseRouter，
# 與 oracle / random / knn 放在同一個 bench 下橫向對比。
#
# 前提：
#   1. 已有 RouterData（data.npz）
#   2. semantic router 正在 http://localhost:8080 執行
#      （若沒有，Step 0 會啟動一個 mock server 供測試）
#
# 流程：
#   Step 0  （可選）啟動 mock semantic router（無須真實 semantic router）
#   Step 1  確認 semantic router 可連線
#   Step 2  用 CLI 訓練 SemanticAPIRouter（fit = 驗證連線）
#   Step 3  CLI bench：semantic_api vs oracle / random / knn
#   Step 4  Python API 方式：直接操作 SemanticAPIRouter + RouterBenchmark
#   Step 5  驗證 predict_probs 格式（one-hot、index 有效）
#   Step 6  清理
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 參數設定
# ─────────────────────────────────────────────────────────────────────────────

DATA_NPZ="${DATA_NPZ:-data.npz}"
SR_BASE_URL="${SR_BASE_URL:-http://localhost:8080}"
SR_TIMEOUT="${SR_TIMEOUT:-10}"
ENDPOINT_PORT="${ENDPOINT_PORT:-8888}"

# bench 參數
MODELS="${MODELS:-gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B}"
DATASETS="${DATASETS:-mmlu_pro_test}"
STRATEGY="${STRATEGY:-llm}"

MOCK_PID=""

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }
die()  { echo "  ✗ $1" >&2; exit 1; }

cleanup() {
    if [ -n "$MOCK_PID" ] && kill -0 "$MOCK_PID" 2>/dev/null; then
        echo "  [cleanup] 停止 mock server (PID $MOCK_PID)..."
        kill "$MOCK_PID"
    fi
}
trap cleanup EXIT

# =============================================================================
# Step 0：啟動 mock semantic router（若真實 semantic router 未運行）
# =============================================================================
step "Step 0 / 6  確認 semantic router 連線（若無則啟動 mock）"

SR_LIVE=false
if curl -sf "$SR_BASE_URL/api/v1/classify/intent" \
        -X POST -H "Content-Type: application/json" \
        -d '{"query":"test"}' >/dev/null 2>&1; then
    ok "semantic router 已運行：$SR_BASE_URL"
    SR_LIVE=true
else
    warn "semantic router 未在 $SR_BASE_URL 運行，啟動 mock server..."

    # 啟動簡易 mock，模擬 POST /api/v1/classify/intent
    python3 - <<'PYEOF' &
import json, random
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_NAMES = ["gpt-oss-20b", "Microsoft-Phi-4", "Google-Gemma-3-27B"]

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 靜音

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = json.loads(self.rfile.read(length) or b"{}")
        # 模擬隨機選模型（實際 semantic router 由 RL/Thompson 決定）
        selected = random.choice(MODEL_NAMES)
        resp = json.dumps({"recommended_model": selected}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        if self.path == "/health":
            resp = b'{"status":"healthy"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)

HTTPServer(("0.0.0.0", 8080), MockHandler).serve_forever()
PYEOF
    MOCK_PID=$!

    echo "  mock server PID: $MOCK_PID"
    sleep 2

    if curl -sf "http://localhost:8080/api/v1/classify/intent" \
            -X POST -H "Content-Type: application/json" \
            -d '{"query":"test"}' >/dev/null 2>&1; then
        SR_BASE_URL="http://localhost:8080"
        ok "mock server 已啟動（$SR_BASE_URL）"
        warn "注意：mock server 回傳隨機結果，SemanticAPIRouter 的 HR 為隨機值"
    else
        die "mock server 啟動失敗"
    fi
fi

# =============================================================================
# Step 1：確認連線 + 回應格式
# =============================================================================
step "Step 1 / 6  驗證 semantic router API 格式"

python3 - <<PYEOF
import json, urllib.request, sys

url = "$SR_BASE_URL/api/v1/classify/intent"
payload = json.dumps({"query": "What is the speed of light?"}).encode()
req = urllib.request.Request(url, data=payload,
                              headers={"Content-Type": "application/json"})

try:
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())
except Exception as e:
    print(f"  [ERR] 無法連線 {url}：{e}", file=sys.stderr)
    sys.exit(1)

print(f"  回應：{d}")
if "recommended_model" not in d:
    print("  [ERR] 回應缺少 recommended_model 欄位", file=sys.stderr)
    sys.exit(1)

print(f"  ✓ API 格式驗證通過  recommended_model={d['recommended_model']!r}")
PYEOF

ok "semantic router API 連線正常"

# =============================================================================
# Step 2：CLI — 訓練 SemanticAPIRouter（fit = 驗證連線）並儲存
# =============================================================================
step "Step 2 / 6  CLI 訓練 SemanticAPIRouter"

[ -f "$DATA_NPZ" ] || die "RouterData 不存在：$DATA_NPZ\n請先執行 01_full_workflow.sh。"

python3 -m LLMRouter \
    router train semantic_api \
    --data "$DATA_NPZ" \
    --semantic-api-url     "$SR_BASE_URL" \
    --semantic-api-timeout "$SR_TIMEOUT" \
    -o semantic_api_router.pkl

[ -f semantic_api_router.pkl ] || die "semantic_api_router.pkl 未產生"

# 驗證 checkpoint
python3 - <<PYEOF
import pickle
with open("semantic_api_router.pkl", "rb") as f:
    ck = pickle.load(f)
assert ck.get("router_type") == "semantic_api", f"type={ck.get('router_type')}"
assert "model_names"  in ck
assert "base_url"     in ck
print(f"  router_type : {ck['router_type']}")
print(f"  base_url    : {ck['base_url']}")
print(f"  model_names : {ck['model_names']}")
print("  ✓ checkpoint 驗證通過")
PYEOF

ok "SemanticAPIRouter 已儲存 → semantic_api_router.pkl"

# =============================================================================
# Step 3：CLI bench — semantic_api vs oracle / random / knn + §4.3 metrics
# =============================================================================
step "Step 3 / 6  CLI bench：semantic_api vs oracle / random / knn（--show-cost）"

python3 -m LLMRouter \
    router bench "oracle,random,knn,semantic_api:semantic_api_router.pkl" \
    --data                   "$DATA_NPZ" \
    --fractions              1.0 \
    --repeats                1 \
    --k                      10 \
    --semantic-api-url       "$SR_BASE_URL" \
    --semantic-api-timeout   "$SR_TIMEOUT" \
    --show-cost

echo ""
echo "  結果說明："
echo "    • SemanticAPIRouter 的 HR 反映 semantic router 的路由品質"
echo "    • 若使用 mock server，HR 接近 random（隨機選模型）"
echo "    • 若連線真實 semantic router，HR 反映其 RL/Thompson 決策品質"

# =============================================================================
# Step 4：Python API — 直接操作 SemanticAPIRouter + RouterBenchmark
# =============================================================================
step "Step 4 / 6  Python API 完整評估流程"

python3 - <<PYEOF
import numpy as np
from LLMRouter.router import (
    RouterData, RouterBenchmark,
    OracleRouter, RandomRouter, KNNRouter,
    SemanticAPIRouter,
    model_unit_costs,
)

# ── 載入資料 ──────────────────────────────────────────────────────────────
data = RouterData.load("$DATA_NPZ")
print(f"\n  RouterData: train={len(data.train_prompt)}  test={len(data.test_prompt)}")
print(f"  模型池: {data.model_names}")

# ── 準備 benchmark ────────────────────────────────────────────────────────
bench = RouterBenchmark(data)

# Oracle（上界）
bench.run(OracleRouter, label="oracle")

# Random（下界）
bench.run(RandomRouter, label="random")

# KNN
bench.run(KNNRouter, {"k": 10}, label="knn")

# SemanticAPIRouter（從儲存的 pkl 載入）
sr = SemanticAPIRouter.load("semantic_api_router.pkl")
probs = sr.predict_probs(data.test_prompt[:50])  # 先用 50 筆測試速度
print(f"\n  SemanticAPIRouter predict_probs shape: {probs.shape}")
print(f"  （前 3 筆）：\n{probs[:3]}")

# 完整 test set 評估
bench.run(SemanticAPIRouter,
          {"base_url": "$SR_BASE_URL", "timeout": $SR_TIMEOUT},
          label="semantic_api")

# ── 印表（含 §4.3 metrics）────────────────────────────────────────────────
print("\n  ─── RouterBenchmark 結果 ───")
bench.print_table(show_cost_metrics=True)

# ── 找出最佳 router ────────────────────────────────────────────────────────
baseline = bench.strongest_baseline()
if baseline:
    print(f"\n  最強 baseline：{baseline.label}  HR={baseline.hr:.4f}")
PYEOF

ok "Python API 評估完成"

# =============================================================================
# Step 5：驗證 predict_probs 格式
# =============================================================================
step "Step 5 / 6  驗證 predict_probs 格式"

python3 - <<PYEOF
import numpy as np
from LLMRouter.router import RouterData, SemanticAPIRouter

data = RouterData.load("$DATA_NPZ")
sr   = SemanticAPIRouter.load("semantic_api_router.pkl")

# 取 10 筆測試
probs = sr.predict_probs(data.test_prompt[:10])
probs = np.asarray(probs)
n, m  = probs.shape

# 驗證 one-hot 格式
assert probs.shape == (10, len(data.model_names)), \
    f"shape 錯誤：{probs.shape}，預期 (10, {len(data.model_names)})"

for i, row in enumerate(probs):
    idx = np.argmax(row)
    assert row[idx] == 1.0,    f"row {i}: max 不是 1.0 ({row})"
    assert row.sum() == 1.0,   f"row {i}: sum 不是 1.0 ({row.sum()})"
    assert 0 <= idx < m,       f"row {i}: index {idx} 超出範圍 [0, {m})"

print(f"  ✓ predict_probs 格式驗證通過")
print(f"    shape={probs.shape}  (one-hot, 每行 sum=1, 每個 index ∈ [0, {m}))")
print(f"    各模型被選次數：{ {data.model_names[i]: int((np.argmax(probs,axis=1)==i).sum()) for i in range(m)} }")
PYEOF

ok "predict_probs 格式正確"

# =============================================================================
# Step 6：清理
# =============================================================================
step "Step 6 / 6  清理"

if [ -n "$MOCK_PID" ] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID"
    wait "$MOCK_PID" 2>/dev/null || true
    MOCK_PID=""
    ok "mock server 已停止"
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "  所有步驟完成！"
echo ""
echo "  產出物："
echo "    semantic_api_router.pkl — 可重複載入的 SemanticAPIRouter checkpoint"
echo ""
echo "  關鍵結論："
echo "    • SemanticAPIRouter.fit() 只驗證連線，不做本地訓練"
echo "    • predict_probs() 對每個 prompt 發一次 HTTP 請求"
echo "    • HR 反映 semantic router 的路由品質（RL 訓練越好 HR 越高）"
echo "    • 可直接參與 RouterBenchmark，與其他 router 等值比較"
echo "════════════════════════════════════════════════════"
