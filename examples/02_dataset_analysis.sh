#!/usr/bin/env bash
# =============================================================================
# Example 2 — §7 四維資料集分析
#
# 從既有的 datasets 直接做四維指標分析，無需先 prepare .npz。
# 適合在決定要訓練哪些 router 之前，快速篩選資料集品質。
#
# 四個指標（Technical Report §7）：
#   CH Score   > 2.0   (P1) 標籤-特徵對齊度，最關鍵
#   Avg_Sim    > 0.025 (P2) 語意一致性
#   Dec_Var σ² > 0.015 (P3) 模型鑑別增益
#   N          ≥ 3000      樣本數基線
#
# 評級：GOOD / MARGINAL / POOR
#
# Step 1  批次分析，印對比表
# Step 2  輸出 CSV，用 Python 解析並分類
# Step 3  對 GOOD/MARGINAL dataset 印完整診斷報告
# Step 4  驗證指標數值合理性（單元驗證）
# =============================================================================
set -euo pipefail

CONFIG="${CONFIG:-$(dirname "$0")/../config.yaml}"

# 要分析的 datasets（逗號分隔）
DATASETS="${DATASETS:-mmlu_pro_test,arc_challenge,gpqa_diamond}"

# 候選模型（逗號分隔）
MODELS="${MODELS:-gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B}"

STRATEGY="${STRATEGY:-llm}"
EMB_MODEL="${EMB_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
OUTPUT_CSV="analysis_results.csv"
OUTPUT_JSON="analysis_results.json"

step() { echo ""; echo "════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════"; }
ok()   { echo "  ✓ $1"; }
warn() { echo "  ⚠ $1"; }

# =============================================================================
# Step 1：批次分析所有 datasets，印橫向對比表
# =============================================================================
step "Step 1 / 4  批次四維分析（對比表）"

python3 -m LLMRouter.scripts.analyze_datasets \
    --datasets  "$DATASETS" \
    --models    "$MODELS" \
    --strategy  "$STRATEGY" \
    --emb-model "$EMB_MODEL" \
    --output    "$OUTPUT_CSV"

ok "對比表已輸出，CSV 儲存至 $OUTPUT_CSV"

# =============================================================================
# Step 2：解析 CSV，分類 GOOD / MARGINAL / POOR
# =============================================================================
step "Step 2 / 4  解析結果，分類各 dataset"

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
            "win_rates": row["win_rates"],
        })

print("")
for grade, items in sorted(results.items()):
    symbol = {"GOOD": "✓", "MARGINAL": "~", "POOR": "✗"}.get(grade, "?")
    print(f"  {symbol} {grade} ({len(items)} datasets):")
    for r in items:
        print(f"      {r['dataset']:<30}  CH={r['ch_score']:.3f}  "
              f"Sim={r['avg_sim']:.3f}  Var={r['dec_var']:.3f}  N={r['n_samples']:,}")

# 建議
print("")
good     = results.get("GOOD", [])
marginal = results.get("MARGINAL", [])
poor     = results.get("POOR", [])

if good:
    names = [r["dataset"] for r in good]
    print(f"  推薦訓練 router（GOOD）：{names}")
if marginal:
    names = [r["dataset"] for r in marginal]
    print(f"  可嘗試訓練（MARGINAL）：{names}（效果可能不穩定）")
if poor:
    names = [r["dataset"] for r in poor]
    print(f"  不建議訓練（POOR）：{names}  → CH Score 太低，請參閱診斷報告")

# 儲存 JSON
with open("$OUTPUT_JSON", "w") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n  結果已儲存至 $OUTPUT_JSON")
PYEOF

ok "分類完成"

# =============================================================================
# Step 3：對 GOOD / MARGINAL datasets 印完整診斷報告
# =============================================================================
step "Step 3 / 4  印完整診斷報告（GOOD + MARGINAL）"

# 從 JSON 取出值得深看的 datasets
DETAIL_DATASETS=$(python3 - <<'PYEOF'
import json
with open("analysis_results.json") as f:
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
    ok "診斷報告完成"
else
    warn "沒有 GOOD 或 MARGINAL 的 dataset，請檢視 POOR 診斷報告"
    python3 -m LLMRouter.scripts.analyze_datasets \
        --datasets  "$DATASETS" \
        --models    "$MODELS" \
        --strategy  "$STRATEGY" \
        --emb-model "$EMB_MODEL" \
        --detail
fi

# =============================================================================
# Step 4：數值合理性驗證
# =============================================================================
step "Step 4 / 4  驗證指標數值合理性"

python3 - <<'PYEOF'
import csv, sys

THRESHOLDS = {"ch_score": 2.0, "avg_sim": 0.025, "dec_var": 0.015}

errors = []
warnings = []

with open("analysis_results.csv") as f:
    for row in csv.DictReader(f):
        ds = row["dataset"]
        ch  = float(row["ch_score"] or 0)
        sim = float(row["avg_sim"]  or 0)
        var = float(row["dec_var"]  or 0)
        n   = int(row["n_samples"]  or 0)

        # 基本範圍檢查
        if not (0 <= ch):
            errors.append(f"{ds}: CH Score 異常（{ch}）")
        if not (0 <= sim <= 1):
            errors.append(f"{ds}: Avg_Sim 超出 [0,1] 範圍（{sim:.4f}）")
        if not (0 <= var <= 0.25):
            errors.append(f"{ds}: Dec_Var 超出合理範圍（{var:.4f}）")
        if n <= 0:
            errors.append(f"{ds}: 樣本數為 0")

        # grade 一致性
        expected_grade = (
            "POOR"     if ch  < THRESHOLDS["ch_score"] else
            "MARGINAL" if sim < THRESHOLDS["avg_sim"]  or var < THRESHOLDS["dec_var"] else
            "GOOD"
        )
        if row["grade"] != expected_grade:
            errors.append(f"{ds}: grade={row['grade']} 但數值應為 {expected_grade}")

if errors:
    print("\n  [FAIL] 驗證失敗：")
    for e in errors:
        print(f"    • {e}")
    sys.exit(1)

print(f"\n  所有指標數值通過合理性驗證（共 {sum(1 for _ in open('analysis_results.csv')) - 1} 筆）")
if warnings:
    for w in warnings:
        print(f"  ⚠ {w}")
PYEOF

ok "驗證通過"

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "════════════════════════════════════════════════════"
echo "  所有步驟完成！"
echo ""
echo "  產出物："
echo "    $OUTPUT_CSV  — 完整指標數值"
echo "    $OUTPUT_JSON — 按評級分類的 JSON"
echo ""
echo "  解讀建議："
echo "    • GOOD     → 推薦訓練 router（可用 01_full_workflow.sh）"
echo "    • MARGINAL → 可試訓，效果可能不穩定，考慮擴大資料量"
echo "    • POOR     → 不建議，先改善資料集"
echo ""
echo "  常見修復方向（POOR）："
echo "    1. CH Score 低   → 換更敏感的 embedding 模型重新計算"
echo "    2. Avg_Sim 低    → 縮小 domain 範圍，增加 prompt density"
echo "    3. Dec_Var 低    → 剔除能力重疊的模型，保留能力差距最大的子集"
echo "    4. N 樣本不足    → 收集更多資料（建議 ≥ 3,000）"
echo "════════════════════════════════════════════════════"
