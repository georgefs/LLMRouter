#!/usr/bin/env bash
# =============================================================================
# Example 1 — 完整端到端 workflow
#
# inference → annotation → §7 資料集品質評估 → RouterData → bench → 訓練儲存
#
# 用法：
#   cd examples/01_full_workflow
#   cp config.sh config.local.sh   # 可選：建立本地覆蓋
#   vim config.sh                  # 修改設定
#   bash run.sh
# =============================================================================
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCENARIO_DIR/config.sh"

# 解析輸出路徑：若使用者設定的是相對路徑，錨定到情景資料夾
[[ "$DATA_NPZ"       != /* ]] && DATA_NPZ="$SCENARIO_DIR/$DATA_NPZ"
[[ "$BEST_ROUTER_PKL" != /* ]] && BEST_ROUTER_PKL="$SCENARIO_DIR/$BEST_ROUTER_PKL"
[[ "$ANALYSIS_CSV"   != /* ]] && ANALYSIS_CSV="$SCENARIO_DIR/$ANALYSIS_CSV"

LLM_CMD="python3 -m LLMRouter --config $CONFIG"

STEP=0
step() { STEP=$((STEP+1)); echo ""; echo "================================================================"; printf "Step %d  —  %s\n" "$STEP" "$1"; echo "================================================================"; }
ok()   { echo "  [OK] $1"; }
warn() { echo "  [!!] $1"; }
die()  { echo "  [ERR] $1" >&2; exit 1; }
confirm() {
    if [ -t 0 ]; then
        read -r -p "  繼續執行？[Y/n] " ans
        case "$ans" in [Nn]*) die "已停止。";; esac
    fi
}

echo "設定："
echo "  DATASETS  = $DATASETS"
echo "  MODELS    = $MODELS"
echo "  DATA_NPZ  = $DATA_NPZ"
echo "  ROUTER    = $BEST_ROUTER_PKL"

# =============================================================================
step "確認 dataset / model 清單"

echo "─── Datasets ───"
$LLM_CMD dataset list

for DS in ${DATASETS//,/ }; do
    $LLM_CMD dataset list 2>/dev/null | grep -q "$DS" \
        && ok "dataset 存在：$DS" \
        || die "dataset 不存在：$DS"
done

# =============================================================================
step "產生 model response（斷點續跑，已有的自動跳過）"

for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → ds=$DS  model=$MODEL"
        $LLM_CMD response gen "$DS" "$MODEL" --concurrency "$CONCURRENCY"
        ok "response gen：$DS / $MODEL"
    done
done

# =============================================================================
step "LLM-as-Judge annotation（斷點續跑）"

for MODEL in ${MODELS//,/ }; do
    for DS in ${DATASETS//,/ }; do
        echo "  → ds=$DS  model=$MODEL  judge=$JUDGE"
        $LLM_CMD annotation gen "$DS" "$MODEL" \
            --strategy llm --judge "$JUDGE" --concurrency "$CONCURRENCY"
        ok "annotation：$DS / $MODEL"
    done
done

# =============================================================================
step "確認三層資料齊全"

for DS in ${DATASETS//,/ }; do
    $LLM_CMD annotation list "$DS"
    for MODEL in ${MODELS//,/ }; do
        $LLM_CMD annotation list "$DS" 2>/dev/null | grep -q "$MODEL" \
            && ok "annotation 存在：$DS / $MODEL" \
            || warn "找不到 annotation：$DS / $MODEL"
    done
done

# =============================================================================
step "§7 四維資料集分析"

python3 -m LLMRouter.scripts.analyze_datasets \
    --datasets  "$DATASETS" \
    --models    "$MODELS" \
    --strategy  llm \
    --emb-model "$EMB_MODEL" \
    --detail \
    --output    "$ANALYSIS_CSV"

ok "分析結果 → $ANALYSIS_CSV"

if ANALYSIS_CSV="$ANALYSIS_CSV" python3 - <<'PYEOF'
import csv, sys, os
poor = [r["dataset"] for r in csv.DictReader(open(os.environ["ANALYSIS_CSV"]))
        if r.get("grade") == "POOR"]
if poor:
    print(f"  [!!] POOR 評級資料集（routing 訓練效果不佳）：{poor}")
    sys.exit(1)
PYEOF
then
    ok "所有資料集評級通過（GOOD 或 MARGINAL）"
else
    warn "偵測到 POOR 評級。建議先改善資料集再訓練。"
    confirm
fi

# =============================================================================
step "router prepare → RouterData .npz（含預存 embedding）"

python3 -m LLMRouter router prepare \
    --datasets     "$DATASETS" \
    --models       "$MODELS" \
    --strategy     llm \
    --scorer       point \
    --train-ratio  0.6 \
    --val-ratio    0.1 \
    --emb-model    "$EMB_MODEL" \
    --emb-batch-size 32 \
    -o "$DATA_NPZ"

[ -f "$DATA_NPZ" ] || die "RouterData 未產生：$DATA_NPZ"

python3 - <<PYEOF
import numpy as np, sys
d = np.load("$DATA_NPZ", allow_pickle=True)
missing = [k for k in ("train_prompt","val_prompt","test_prompt",
                        "train_score","test_score","model")
           if k not in d]
if missing:
    print(f"  [ERR] RouterData 缺欄位：{missing}", file=sys.stderr); sys.exit(1)
print(f"  train={len(d['train_prompt'])}  test={len(d['test_prompt'])}"
      f"  models={list(d['model'])}")
print(f"  embedding 預存：{'是' if 'train_embed' in d else '否'}")
PYEOF

ok "RouterData → $DATA_NPZ"

# =============================================================================
step "router bench — 多 router 橫向對比（§4.3 metrics）"

python3 -m LLMRouter router bench "$ROUTERS" \
    --data      "$DATA_NPZ" \
    --fractions "$FRACTIONS" \
    --repeats   "$REPEATS" \
    --k         10 \
    --show-cost

# =============================================================================
step "訓練 KNN router 並儲存"

python3 -m LLMRouter router train knn \
    --data "$DATA_NPZ" --k 10 -o "$BEST_ROUTER_PKL"

[ -f "$BEST_ROUTER_PKL" ] || die "router.pkl 未產生"

python3 - <<PYEOF
import pickle
ck = pickle.load(open("$BEST_ROUTER_PKL","rb"))
print(f"  router_type={ck['router_type']}  models={ck['model_names']}")
PYEOF

python3 -m LLMRouter router eval knn \
    --data "$DATA_NPZ" --model "$BEST_ROUTER_PKL"

ok "router → $BEST_ROUTER_PKL"

# =============================================================================
echo ""
echo "================================================================"
echo "完成！"
echo "  $DATA_NPZ"
echo "  $BEST_ROUTER_PKL"
echo "  $ANALYSIS_CSV"
echo ""
echo "下一步："
echo "  部署 endpoint → examples/03_deploy_endpoint/"
echo "================================================================"
