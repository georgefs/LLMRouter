#!/usr/bin/env bash
# =============================================================================
# Example 4 — SemanticAPIRouter：把 semantic router 納入橫向評估
#
# 將正在運行的 semantic router 包裝成 BaseRouter，
# 與 oracle / random / knn 放進同一個 RouterBenchmark 對比。
#
# 用法：
#   cd examples/04_semantic_api_router
#   vim config.sh      # 設定 SR_BASE_URL（或留空讓 script 啟動 mock）
#   bash run.sh
#
# 若 semantic router 未在 SR_BASE_URL 運行，Step 0 會自動啟動 mock server，
# mock 回傳隨機結果，SemanticAPIRouter 的 HR 會接近 random（符合預期）。
# =============================================================================
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCENARIO_DIR/config.sh"

[[ "$DATA_NPZ"       != /* ]] && DATA_NPZ="$SCENARIO_DIR/$DATA_NPZ"
[[ "$SR_ROUTER_PKL"  != /* ]] && SR_ROUTER_PKL="$SCENARIO_DIR/$SR_ROUTER_PKL"

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }
die()  { echo "  ✗ $1" >&2; exit 1; }

MOCK_PID=""
cleanup() {
    [ -n "$MOCK_PID" ] && kill -0 "$MOCK_PID" 2>/dev/null && kill "$MOCK_PID"
}
trap cleanup EXIT

echo "設定："
echo "  DATA_NPZ      = $DATA_NPZ"
echo "  SR_BASE_URL   = $SR_BASE_URL"
echo "  SR_ROUTER_PKL = $SR_ROUTER_PKL"

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

    # 從 SR_BASE_URL 解析 port（預設 8080）
    MOCK_PORT=$(python3 -c "from urllib.parse import urlparse; print(urlparse('$SR_BASE_URL').port or 8080)")

    # 若 port 已佔用則找下一個可用 port
    while ss -tlnp "sport = :$MOCK_PORT" 2>/dev/null | grep -q LISTEN; do
        MOCK_PORT=$((MOCK_PORT + 1))
    done

    # 取得模型清單供 mock 使用
    MOCK_MODELS=$(python3 - <<PYEOF
import numpy as np
d = np.load("$DATA_NPZ", allow_pickle=True)
print(",".join(d["model"]))
PYEOF
)

    MOCK_PORT_VAL="$MOCK_PORT" python3 - <<PYEOF &
import json, random, os
from http.server import HTTPServer, BaseHTTPRequestHandler

MODEL_NAMES = "$MOCK_MODELS".split(",")
PORT = int(os.environ["MOCK_PORT_VAL"])

class MockHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        resp = json.dumps({"recommended_model": random.choice(MODEL_NAMES)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(resp))
        self.end_headers()
        self.wfile.write(resp)

    def do_GET(self):
        resp = b'{"status":"healthy"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp)

HTTPServer(("0.0.0.0", PORT), MockHandler).serve_forever()
PYEOF
    MOCK_PID=$!
    SR_BASE_URL="http://localhost:$MOCK_PORT"
    sleep 2

    curl -sf "$SR_BASE_URL/api/v1/classify/intent" \
         -X POST -H "Content-Type: application/json" \
         -d '{"query":"test"}' >/dev/null 2>&1 \
        || die "mock server 啟動失敗"

    ok "mock server 已啟動（PID $MOCK_PID）"
    warn "mock 回傳隨機結果；SemanticAPIRouter HR 預期接近 random"
fi

# =============================================================================
step "Step 1 / 6  驗證 semantic router API 格式"

python3 - <<PYEOF
import json, urllib.request, sys

url = "$SR_BASE_URL/api/v1/classify/intent"
req = urllib.request.Request(url,
        data=json.dumps({"query": "What is the speed of light?"}).encode(),
        headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=10) as r:
    d = json.loads(r.read())

assert "recommended_model" in d, f"回應缺少 recommended_model：{d}"
print(f"  ✓ recommended_model={d['recommended_model']!r}")
PYEOF

ok "API 格式驗證通過"

# =============================================================================
step "Step 2 / 6  訓練 SemanticAPIRouter（fit = 驗證連線）"

[ -f "$DATA_NPZ" ] || die "RouterData 不存在：$DATA_NPZ"

python3 -m LLMRouter router train semantic_api \
    --data                   "$DATA_NPZ" \
    --semantic-api-url       "$SR_BASE_URL" \
    --semantic-api-timeout   "$SR_TIMEOUT" \
    -o "$SR_ROUTER_PKL"

python3 - <<PYEOF
import pickle
ck = pickle.load(open("$SR_ROUTER_PKL","rb"))
assert ck.get("router_type") == "semantic_api"
assert "model_names" in ck and "base_url" in ck
print(f"  router_type={ck['router_type']}  base_url={ck['base_url']}")
print(f"  model_names={ck['model_names']}")
PYEOF

ok "SemanticAPIRouter → $SR_ROUTER_PKL"

# =============================================================================
step "Step 3 / 6  CLI bench：semantic_api vs oracle / random / knn"

python3 -m LLMRouter router bench \
    "oracle,random,knn,semantic_api:$SR_ROUTER_PKL" \
    --data                 "$DATA_NPZ" \
    --fractions            1.0 \
    --repeats              1 \
    --k                    10 \
    --semantic-api-url     "$SR_BASE_URL" \
    --semantic-api-timeout "$SR_TIMEOUT" \
    --show-cost

echo ""
echo "  結果說明："
if [ "$SR_LIVE" = "true" ]; then
    echo "    HR 高於 random → semantic router RL 訓練有效"
    echo "    HR 接近 oracle → 無需額外訓練本地 router"
else
    echo "    ⚠ 使用 mock server，HR 接近 random 是正常的"
    echo "    接上真實 semantic router 才能得到有意義的比較"
fi

# =============================================================================
step "Step 4 / 6  Python API 完整評估"

python3 - <<PYEOF
import numpy as np
from LLMRouter.router import (
    RouterData, RouterBenchmark,
    OracleRouter, RandomRouter, KNNRouter, SemanticAPIRouter,
)

data  = RouterData.load("$DATA_NPZ")
bench = RouterBenchmark(data)
print(f"\n  RouterData: train={len(data.train_prompt)}  test={len(data.test_prompt)}")
print(f"  模型池: {data.models}")

bench.run(OracleRouter,    label="oracle")
bench.run(RandomRouter,    label="random")
bench.run(KNNRouter, {"k": 10}, label="knn")

sr = SemanticAPIRouter.load("$SR_ROUTER_PKL")
bench.run(SemanticAPIRouter,
          {"base_url": "$SR_BASE_URL", "timeout": $SR_TIMEOUT},
          label="semantic_api")

print("\n  ─── RouterBenchmark 結果 ───")
bench.print_table(show_cost_metrics=True)

baseline = bench.strongest_baseline()
if baseline:
    print(f"\n  最強 baseline：{baseline.label}  HR={baseline.hr:.4f}")
PYEOF

ok "Python API 評估完成"

# =============================================================================
step "Step 5 / 6  驗證 predict_probs 格式（one-hot）"

python3 - <<PYEOF
import numpy as np
from LLMRouter.router import RouterData, SemanticAPIRouter

data  = RouterData.load("$DATA_NPZ")
sr    = SemanticAPIRouter.load("$SR_ROUTER_PKL")
probs = np.asarray(sr.predict_probs(data.test_prompt[:10]))
n, m  = probs.shape

assert probs.shape == (10, len(data.models)), \
    f"shape 錯誤：{probs.shape}"
for i, row in enumerate(probs):
    assert row.sum() == 1.0 and row.max() == 1.0, f"row {i} 不是 one-hot"

print(f"  ✓ shape={probs.shape}  one-hot 格式正確")
counts = {data.models[i]: int((np.argmax(probs,1)==i).sum()) for i in range(m)}
print(f"  各模型被選次數（10 筆）：{counts}")
PYEOF

ok "predict_probs 格式正確"

# =============================================================================
step "Step 6 / 6  清理"

if [ -n "$MOCK_PID" ] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID"; wait "$MOCK_PID" 2>/dev/null || true
    MOCK_PID=""
    ok "mock server 已停止"
fi

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "完成！"
echo "  $SR_ROUTER_PKL"
echo ""
echo "重點："
echo "  • fit() 只驗證連線，不做本地訓練"
echo "  • predict_probs() 每筆 prompt 發一次 HTTP 請求"
echo "  • 可直接放入 RouterBenchmark 與其他 router 等值比較"
echo "════════════════════════════════════════════════════"
