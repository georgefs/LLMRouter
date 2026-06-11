#!/usr/bin/env bash
# =============================================================================
# Example 1 — 完整端到端 workflow
#
# 涵蓋範圍：
#   Step 1  查看 dataset / model 清單
#   Step 2  對每個 model 產生 response（斷點續跑）
#   Step 3  LLM-as-Judge annotation
#   Step 4  確認三層資料齊全
#   Step 5  §7 四維資料集分析（先確認 routing 價值再繼續）
#   Step 6  prepare → RouterData .npz
#   Step 7  router bench（多 router 橫向對比 + §4.3 metrics）
#   Step 8  訓練最佳 router 並儲存
#
# 驗證方式：
#   每個 Step 完成後以 echo + exit code 確認；
#   Step 5 解析 analyze 結果，CH Score < 2.0 時發出警告並詢問是否繼續。
#
# 使用前請先設定下方「參數設定」區段
# =============================================================================
set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 參數設定
# ─────────────────────────────────────────────────────────────────────────────

CONFIG="${CONFIG:-$(dirname "$0")/../config.yaml}"

# 要評測的 dataset（逗號分隔可同時跑多個）
DATASETS="${DATASETS:-mmlu_pro_test}"

# 候選模型（逗號分隔）
MODELS="${MODELS:-gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B}"

# LLM-as-Judge 使用的 judge 模型
JUDGE="${JUDGE:-gpt-oss-120b}"

# 輸出路徑
DATA_NPZ="${DATA_NPZ:-data.npz}"
BEST_ROUTER_PKL="${BEST_ROUTER_PKL:-best_router.pkl}"

# bench 參數
FRACTIONS="${FRACTIONS:-0.1,0.3,0.5,0.7,1.0}"
REPEATS="${REPEATS:-3}"
ROUTERS="${ROUTERS:-oracle,random,knn}"

# 並發數
CONCURRENCY="${CONCURRENCY:-16}"

# embedding 模型（router prepare 預存，加速後續 bench）
EMB_MODEL="${EMB_MODEL:-mixedbread-ai/mxbai-embed-large-v1}"

# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

STEP=0
step() {
    STEP=$((STEP + 1))
    echo ""
    echo "================================================================"
    printf "Step %d  —  %s\n" "$STEP" "$1"
    echo "================================================================"
}

ok()   { echo "  [OK] $1"; }
warn() { echo "  [!!] $1"; }
die()  { echo "  [ERR] $1" >&2; exit 1; }

confirm() {
    # 詢問是否繼續（非互動環境直接繼續）
    if [ -t 0 ]; then
        read -r -p "  繼續執行？[Y/n] " ans
        case "$ans" in [Nn]*) die "已停止。";; esac
    fi
}

LLM_CMD="python3 -m LLMRouter --config $CONFIG"

# =============================================================================
# Step 1：確認 dataset / model 清單
# =============================================================================
step "查看 dataset / model 清單"

echo "─── Datasets ───"
$LLM_CMD dataset list

echo ""
echo "─── 確認目標 datasets 存在 ───"
for DS in ${DATASETS//,/ }; do
    if $LLM_CMD dataset list 2>/dev/null | grep -q "$DS"; then
        ok "dataset 存在：$DS"
    else
        die "dataset 不存在：$DS\n請先確認 DATA_PATH 並準備 dataset。"
    fi
done

# =============================================================================
# Step 2：產生 model response
# =============================================================================
step "產生 model response（斷點續跑，已有的自動跳過）"

for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → ds=$DS  model=$MODEL"
        $LLM_CMD response gen "$DS" "$MODEL" --concurrency "$CONCURRENCY"
        ok "response gen 完成：$DS / $MODEL"
    done
done

# =============================================================================
# Step 3：LLM-as-Judge annotation
# =============================================================================
step "LLM-as-Judge annotation（斷點續跑，已標注的自動跳過）"

for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → ds=$DS  model=$MODEL  judge=$JUDGE"
        $LLM_CMD annotation gen "$DS" "$MODEL" \
            --strategy llm \
            --judge "$JUDGE" \
            --concurrency "$CONCURRENCY"
        ok "annotation 完成：$DS / $MODEL"
    done
done

# =============================================================================
# Step 4：確認三層資料齊全
# =============================================================================
step "確認 response + annotation 都已完成"

for DS in ${DATASETS//,/ }; do
    echo "  dataset: $DS"
    $LLM_CMD annotation list "$DS"

    # 驗證：每個 model 都有 llm annotation
    for MODEL in ${MODELS//,/ }; do
        if $LLM_CMD annotation list "$DS" 2>/dev/null | grep -q "$MODEL"; then
            ok "annotation 存在：$DS / $MODEL"
        else
            warn "找不到 annotation：$DS / $MODEL  （請確認 Step 3 是否成功）"
        fi
    done
done

# =============================================================================
# Step 5：§7 四維資料集分析
# =============================================================================
step "§7 四維資料集分析（評估 routing 價值，決定是否繼續訓練）"

python3 -m LLMRouter.scripts.analyze_datasets \
    --datasets  "$DATASETS" \
    --models    "$MODELS" \
    --strategy  llm \
    --emb-model "$EMB_MODEL" \
    --detail \
    --output    "dataset_analysis.csv"

ok "分析結果已儲存至 dataset_analysis.csv"

# 解析 POOR grade 警告
if python3 - <<'PYEOF'
import csv, sys
poor = []
with open("dataset_analysis.csv") as f:
    for r in csv.DictReader(f):
        if r.get("grade") == "POOR":
            poor.append(r["dataset"])
if poor:
    print(f"  [!!] 以下 dataset 評級為 POOR，routing 訓練預期效果不佳：{poor}")
    sys.exit(1)
PYEOF
then
    ok "所有 dataset 評級通過（GOOD 或 MARGINAL），繼續訓練"
else
    warn "偵測到 POOR 評級的 dataset。建議改善資料集（增加樣本、調整模型池）後再訓練。"
    confirm
fi

# =============================================================================
# Step 6：準備 RouterData（三層 join → .npz，預存 embedding）
# =============================================================================
step "router prepare → $DATA_NPZ（含預存 embedding 加速後續 bench）"

python3 -m LLMRouter --config "$CONFIG" \
    router prepare \
    --datasets     "$DATASETS" \
    --models       "$MODELS" \
    --strategy     llm \
    --scorer       point \
    --train-ratio  0.6 \
    --val-ratio    0.1 \
    --emb-model    "$EMB_MODEL" \
    --emb-batch-size 32 \
    -o "$DATA_NPZ"

[ -f "$DATA_NPZ" ] || die "RouterData 輸出失敗：$DATA_NPZ 不存在"

# 驗證：npz 包含必要的 key
python3 - <<PYEOF
import numpy as np, sys
d = np.load("$DATA_NPZ", allow_pickle=True)
required = ["train_prompt", "val_prompt", "test_prompt",
            "train_score", "test_score", "model_names"]
missing = [k for k in required if k not in d]
if missing:
    print(f"  [ERR] RouterData 缺少欄位：{missing}", file=sys.stderr)
    sys.exit(1)
n_train = len(d["train_prompt"])
n_test  = len(d["test_prompt"])
models  = list(d["model_names"])
print(f"  [OK] RouterData 驗證通過")
print(f"       train={n_train}  test={n_test}  models={models}")
has_emb = "train_embed" in d and d["train_embed"] is not None
print(f"       embedding 預存：{'是' if has_emb else '否'}")
PYEOF

ok "RouterData 準備完成：$DATA_NPZ"

# =============================================================================
# Step 7：Router bench（多 router 橫向對比 + §4.3 metrics）
# =============================================================================
step "router bench — 多 router 橫向對比（含 §4.3 HR / Cost / TER / NBS）"

python3 -m LLMRouter \
    router bench "$ROUTERS" \
    --data       "$DATA_NPZ" \
    --fractions  "$FRACTIONS" \
    --repeats    "$REPEATS" \
    --k          10 \
    --show-cost

echo ""
echo "  指標說明："
echo "    HR   = Hit Rate：選出模型達到最高分的比例（↑）"
echo "    Cost = 平均 token 數加權成本（↓）"
echo "    TER  = Cost_Savings% / HR_Sacrifice%（↑，Dominant=雙優）"
echo "    NBS  = 3×ΔHR% + ΔCost_Savings%（↑，正值=淨收益）"
echo "    mu   = 選出模型的平均分數（↑）"
echo "    vb   = mu / oracle（↑，越接近 1 越好）"

# =============================================================================
# Step 8：訓練最佳 router 並儲存
# =============================================================================
step "訓練 KNN router（100% 訓練資料）並儲存為 $BEST_ROUTER_PKL"

python3 -m LLMRouter \
    router train knn \
    --data "$DATA_NPZ" \
    --k    10 \
    -o     "$BEST_ROUTER_PKL"

[ -f "$BEST_ROUTER_PKL" ] || die "router train 輸出失敗：$BEST_ROUTER_PKL 不存在"

# 驗證：載入 pkl 並做一次推論
python3 - <<PYEOF
import pickle, numpy as np, sys
with open("$BEST_ROUTER_PKL", "rb") as f:
    ck = pickle.load(f)
rtype = ck.get("router_type", "unknown")
model_names = ck.get("model_names", [])
print(f"  [OK] checkpoint 驗證")
print(f"       router_type={rtype}  model_names={model_names}")
PYEOF

# 快速 eval 驗證
python3 -m LLMRouter \
    router eval knn \
    --data  "$DATA_NPZ" \
    --model "$BEST_ROUTER_PKL"

ok "router 儲存完成：$BEST_ROUTER_PKL"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "================================================================"
echo "所有步驟完成！"
echo ""
echo "產出物："
echo "  $DATA_NPZ          — RouterData（可重複用於其他 bench）"
echo "  $BEST_ROUTER_PKL   — 已訓練的 KNN router"
echo "  dataset_analysis.csv — §7 四維分析報告"
echo ""
echo "下一步建議："
echo "  • 部署 router 為 HTTP endpoint → 參考 examples/03_deploy_endpoint.sh"
echo "  • 與 semantic router 串接     → 參考 examples/03_deploy_endpoint.sh"
echo "================================================================"
