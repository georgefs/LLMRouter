#!/usr/bin/env bash
# =============================================================================
# Example 2 — §7 四維資料集分析
#
# 批次評估多個資料集的 routing 價值（CH Score / Avg_Sim / Dec_Var / N），
# 決定哪些值得投入 router 訓練資源。
#
# 用法：
#   cd examples/02_dataset_analysis
#   vim config.sh      # 設定 DATASETS / MODELS
#   bash run.sh
# =============================================================================
set -euo pipefail

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCENARIO_DIR/config.sh"

[[ "$OUTPUT_CSV"  != /* ]] && OUTPUT_CSV="$SCENARIO_DIR/$OUTPUT_CSV"
[[ "$OUTPUT_JSON" != /* ]] && OUTPUT_JSON="$SCENARIO_DIR/$OUTPUT_JSON"

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }

echo "設定："
echo "  DATASETS = $DATASETS"
echo "  MODELS   = $MODELS"
echo "  OUTPUT   = $OUTPUT_CSV"

# =============================================================================
step "Step 1 / 4  批次四維分析（對比表）"

python3 -m LLMRouter.scripts.analyze_datasets \
    --datasets  "$DATASETS" \
    --models    "$MODELS" \
    --strategy  "$STRATEGY" \
    --emb-model "$EMB_MODEL" \
    --output    "$OUTPUT_CSV"

ok "CSV → $OUTPUT_CSV"

# =============================================================================
step "Step 2 / 4  分類 GOOD / MARGINAL / POOR"

python3 - <<PYEOF
import csv, json

results = {}
with open("$OUTPUT_CSV") as f:
    for row in csv.DictReader(f):
        grade = row["grade"]
        results.setdefault(grade, []).append({
            "dataset":   row["dataset"],
            "ch_score":  float(row["ch_score"] or 0),
            "avg_sim":   float(row["avg_sim"]  or 0),
            "dec_var":   float(row["dec_var"]  or 0),
            "n_samples": int(row["n_samples"]  or 0),
        })

print("")
for grade, items in sorted(results.items()):
    symbol = {"GOOD": "✓", "MARGINAL": "~", "POOR": "✗"}.get(grade, "?")
    print(f"  {symbol} {grade} ({len(items)}):")
    for r in items:
        print(f"      {r['dataset']:<30}  CH={r['ch_score']:.3f}"
              f"  Sim={r['avg_sim']:.3f}  Var={r['dec_var']:.3f}"
              f"  N={r['n_samples']:,}")

print("")
good     = [r["dataset"] for r in results.get("GOOD",     [])]
marginal = [r["dataset"] for r in results.get("MARGINAL", [])]
poor     = [r["dataset"] for r in results.get("POOR",     [])]
if good:     print(f"  推薦訓練（GOOD）：{good}")
if marginal: print(f"  可嘗試訓練（MARGINAL）：{marginal}")
if poor:     print(f"  不建議訓練（POOR）：{poor}")

with open("$OUTPUT_JSON", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n  JSON → $OUTPUT_JSON")
PYEOF

ok "分類完成"

# =============================================================================
step "Step 3 / 4  完整診斷報告（GOOD + MARGINAL）"

DETAIL_DATASETS=$(OUTPUT_JSON="$OUTPUT_JSON" python3 - <<'PYEOF'
import json, os
with open(os.environ["OUTPUT_JSON"]) as f:
    results = json.load(f)
keep = results.get("GOOD", []) + results.get("MARGINAL", [])
print(",".join(r["dataset"] for r in keep))
PYEOF
)

if [ -n "$DETAIL_DATASETS" ]; then
    echo "  → 詳細診斷：$DETAIL_DATASETS"
    python3 -m LLMRouter.scripts.analyze_datasets \
        --datasets  "$DETAIL_DATASETS" \
        --models    "$MODELS" \
        --strategy  "$STRATEGY" \
        --emb-model "$EMB_MODEL" \
        --detail
else
    warn "無 GOOD/MARGINAL 資料集，對全部印診斷報告"
    python3 -m LLMRouter.scripts.analyze_datasets \
        --datasets  "$DATASETS" \
        --models    "$MODELS" \
        --strategy  "$STRATEGY" \
        --emb-model "$EMB_MODEL" \
        --detail
fi

ok "診斷報告完成"

# =============================================================================
step "Step 4 / 4  驗證指標數值合理性"

OUTPUT_CSV="$OUTPUT_CSV" python3 - <<'PYEOF'
import csv, sys, os

THRESHOLDS = {"ch_score": 2.0, "avg_sim": 0.025, "dec_var": 0.015}
csv_path = os.environ["OUTPUT_CSV"]
errors = []

with open(csv_path) as f:
    for row in csv.DictReader(f):
        ds  = row["dataset"]
        ch  = float(row["ch_score"] or 0)
        sim = float(row["avg_sim"]  or 0)
        var = float(row["dec_var"]  or 0)
        n   = int(row["n_samples"]  or 0)

        if not (ch  >= 0):         errors.append(f"{ds}: CH Score 異常（{ch}）")
        if not (0 <= sim <= 1):    errors.append(f"{ds}: Avg_Sim 超出範圍（{sim:.4f}）")
        if not (0 <= var <= 0.25): errors.append(f"{ds}: Dec_Var 超出範圍（{var:.4f}）")
        if n <= 0:                  errors.append(f"{ds}: 樣本數為 0")

        expected = (
            "POOR"     if ch  < THRESHOLDS["ch_score"] else
            "MARGINAL" if sim < THRESHOLDS["avg_sim"] or var < THRESHOLDS["dec_var"] else
            "GOOD"
        )
        if row["grade"] != expected:
            errors.append(f"{ds}: grade={row['grade']} 但數值應為 {expected}")

if errors:
    print("\n  [FAIL]")
    for e in errors: print(f"    • {e}")
    sys.exit(1)

total = sum(1 for _ in open(csv_path)) - 1
print(f"\n  ✓ 所有指標驗證通過（{total} 筆）")
PYEOF

ok "驗證通過"

# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "完成！"
echo "  $OUTPUT_CSV"
echo "  $OUTPUT_JSON"
echo ""
echo "後續建議："
echo "  GOOD     → bash ../01_full_workflow/run.sh"
echo "  MARGINAL → 可試訓，效果可能不穩定"
echo "  POOR     → CH 低：換 embedding 模型；Var 低：縮小模型池"
echo "════════════════════════════════════════════════════"
