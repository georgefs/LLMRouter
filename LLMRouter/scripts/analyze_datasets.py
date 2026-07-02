#!/usr/bin/env python3
"""
analyze_datasets.py — 批次執行 §7 四維指標，評估多個 dataset 的 routing 價值

四個指標（Technical Report §7）：
  N         — 樣本數
  Avg_Sim   — 語意一致性（平均成對 cosine 相似度）；閾值 > 0.025
  Dec_Var   — 鑑別增益（模型 win-rate 的變異數 σ²）；閾值 > 0.015
  CH Score  — 標籤-特徵對齊度（Calinski-Harabasz）；閾值 > 2.0

用法
----
python -m LLMRouter.scripts.analyze_datasets \\
    --datasets mmlu_pro_test,arc_challenge,gpqa_diamond \\
    --models gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B \\
    --strategy llm \\
    [--detail]        # 每個 dataset 印完整診斷報告
    [--output out.csv]
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# 輸出工具
# ─────────────────────────────────────────────────────────────────────────────

_THRESHOLDS = {
    "ch_score": 2.0,
    "avg_sim":  0.025,
    "dec_var":  0.015,
    "n_samples": 3000,
}

GRADE_EMOJI = {"GOOD": "✓", "MARGINAL": "~", "POOR": "✗"}


def _cell(v, decimals=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _pass(field: str, value) -> str:
    thr = _THRESHOLDS.get(field)
    if thr is None or value is None:
        return ""
    return "✓" if value >= thr else "✗"


def _print_comparison(rows: List[dict]) -> None:
    """
    橫向對比表：每個 dataset 一行，顯示四維指標 + 綜合評分。
    """
    header = ["dataset", "N", "CH Score", "Avg_Sim", "Dec_Var σ²", "grade", "n_models"]
    col_w  = [max(len(h), max((len(str(_row_val(r, h))) for r in rows), default=0))
              for h in header]

    def fmt_row(r):
        return [
            r["dataset"],
            f"{r['n_samples']:,}",
            f"{_cell(r['ch_score'])} {_pass('ch_score', r['ch_score'])}",
            f"{_cell(r['avg_sim'])} {_pass('avg_sim',  r['avg_sim'])}",
            f"{_cell(r['dec_var'])} {_pass('dec_var',  r['dec_var'])}",
            f"{GRADE_EMOJI.get(r['grade'], '?')} {r['grade']}",
            str(r["n_models"]),
        ]

    # 重新算寬（含實際值）
    all_rows = [header] + [fmt_row(r) for r in rows]
    col_w = [max(len(str(row[i])) for row in all_rows) for i in range(len(header))]
    sep = "-+-".join("-" * w for w in col_w)
    fmt = " | ".join(f"{{:<{w}}}" for w in col_w)

    total = sum(col_w) + 3 * (len(col_w) - 1)
    print("\n" + "=" * total)
    print(" §7 Dataset Routing Value — 四維指標對比")
    print("=" * total)
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(*fmt_row(r)))
    print(sep)
    print(f"  閾值：CH Score > 2.0  |  Avg_Sim > 0.025  |  Dec_Var > 0.015  |  ✓=Pass  ✗=Fail")
    print("=" * total)


def _row_val(r, h):
    mapping = {
        "dataset": r["dataset"],
        "N": f"{r['n_samples']:,}",
        "CH Score": _cell(r["ch_score"]),
        "Avg_Sim":  _cell(r["avg_sim"]),
        "Dec_Var σ²": _cell(r["dec_var"]),
        "grade":    r["grade"],
        "n_models": r["n_models"],
    }
    return mapping.get(h, "")


# ─────────────────────────────────────────────────────────────────────────────
# 單一 dataset 分析
# ─────────────────────────────────────────────────────────────────────────────

def analyze_one(
    dataset_name: str,
    models: List[str],
    strategy: str,
    *,
    scorer: str = "point",
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    seed: int = 42,
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    emb_batch_size: int = 32,
    config_path: Optional[str] = None,
    base_path: Optional[str] = None,
    detail: bool = False,
) -> dict:
    """
    回傳 summary dict：
      dataset, n_samples, avg_sim, dec_var, ch_score, grade, n_models,
      win_rates (list), model_names (list)
    """
    from LLMRouter.manager import DatasetManager
    from LLMRouter.router.data import DataPreparer
    from LLMRouter.router.dataset_eval import analyze, format_report

    # ── DatasetManager ────────────────────────────────────────────────────────
    import os
    from LLMRouter.scorer import FieldScorer
    mgr = DatasetManager(base_path or os.environ.get("DATA_PATH"))
    scorer_obj = FieldScorer(scorer) if isinstance(scorer, str) else scorer

    # ── 資料 ──────────────────────────────────────────────────────────────────
    print(f"\n[{dataset_name}] 準備資料…", flush=True)
    data = DataPreparer().from_manager(
        mgr,
        datasets=[dataset_name],
        models=models,
        strategies=[strategy],
        scorer=scorer_obj,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )
    model_names = list(data.models)
    print(f"[{dataset_name}] train={len(data.train_prompt)}  "
          f"val={len(data.val_prompt)}  test={len(data.test_prompt)}")

    # ── 分析 ──────────────────────────────────────────────────────────────────
    print(f"[{dataset_name}] 計算四維指標…", flush=True)
    result = analyze(data, emb_model=emb_model, emb_batch_size=emb_batch_size)

    if detail:
        print(format_report(result, title=dataset_name))

    return {
        "dataset":     dataset_name,
        "n_samples":   result.n_samples,
        "avg_sim":     result.avg_sim,
        "dec_var":     result.dec_var,
        "ch_score":    result.ch_score,
        "grade":       result.overall_grade,
        "n_models":    len(model_names),
        "model_names": model_names,
        "win_rates":   result.win_rates.tolist(),
        "n_train":     result.n_train,
        "n_val":       result.n_val,
        "n_test":      result.n_test,
        "emb_source":  result.emb_source,
        "warnings":    result.warnings,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV 輸出
# ─────────────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "dataset", "grade", "n_samples", "ch_score", "avg_sim", "dec_var",
    "n_models", "n_train", "n_val", "n_test", "emb_source",
]


def write_csv(path: str, rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS + ["win_rates", "warnings"])
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in _CSV_FIELDS}
            row["win_rates"] = ";".join(
                f"{m}={v:.4f}"
                for m, v in zip(r.get("model_names", []), r.get("win_rates", []))
            )
            row["warnings"] = " | ".join(r.get("warnings", []))
            w.writerow(row)
    print(f"\n[output] CSV 已寫入 → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="批次執行 §7 四維指標，評估多個 dataset 的 routing 價值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--datasets",  required=True,
                   help="逗號分隔的 dataset 名稱")
    p.add_argument("--models",    required=True,
                   help="逗號分隔的 model 名稱")
    p.add_argument("--strategy",  required=True,
                   help="annotation strategy（e.g. llm）")
    p.add_argument("--scorer",    default="point",
                   help="scorer 名稱（預設 point）")
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--emb-model",
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="embedding 模型（預設 all-MiniLM-L6-v2）")
    p.add_argument("--emb-batch-size", type=int, default=32,
                   help="embedding batch size（預設 32）")
    p.add_argument("--detail", action="store_true",
                   help="印每個 dataset 的完整診斷報告")
    p.add_argument("--config",  default=None, help="config.yaml 路徑")
    p.add_argument("--base",    default=None, help="data 根目錄")
    p.add_argument("--output",  default=None, help="輸出 CSV 路徑（可選）")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models   = [m.strip() for m in args.models.split(",")   if m.strip()]

    all_rows: List[dict] = []

    for ds in datasets:
        try:
            row = analyze_one(
                ds, models, args.strategy,
                scorer=args.scorer,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.seed,
                emb_model=args.emb_model,
                emb_batch_size=args.emb_batch_size,
                config_path=args.config,
                base_path=args.base,
                detail=args.detail,
            )
            all_rows.append(row)
        except Exception as exc:
            print(f"\n[ERROR] {ds}: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()

    if all_rows:
        _print_comparison(all_rows)

    if args.output and all_rows:
        write_csv(args.output, all_rows)


if __name__ == "__main__":
    main()
