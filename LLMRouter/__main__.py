"""
CLI entry point: python -m LLMRouter <subcommand> ...

Subcommands:
  dataset list                                        列出所有 dataset
  model list <dataset>                                列出某 dataset 的 response model
  annotation list <dataset>                           列出某 dataset 的 annotation strategy/model
  response gen <dataset> <model>                      用 litellm 對 dataset 做 inference
  extract --datasets d1,d2 --models m1,m2 --strategy s [--output out.jsonl]
  annotation gen <dataset> <model> --judge <judge_model> [--strategy llm]
  router prepare --datasets d1,d2 --models m1,m2 --strategy s --output data.npz
  router train <type> --data data.npz [--output router.pkl] [router options]
  router eval <type> --data data.npz [--model router.pkl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .manager import DatasetManager, _read_jsonl


# ── helpers ──────────────────────────────────────────────────────────────────


def _print_jsonl(data: list) -> None:
    for item in data:
        print(json.dumps(item, ensure_ascii=False))


# ── subcommand handlers ───────────────────────────────────────────────────────


def cmd_scorer_list() -> None:
    from .scorer import list_scorers
    names = list_scorers()
    if not names:
        print("（無已註冊的 scorer）")
        return
    for name in names:
        print(name)


def cmd_dataset_list(mgr: DatasetManager, args: argparse.Namespace) -> None:
    datasets = mgr.list_datasets()
    if not datasets:
        print("（無資料）")
        return
    for name in datasets:
        path = mgr.datasets_path / f"{name}.jsonl"
        size = sum(1 for _ in path.open(encoding="utf-8") if _.strip())
        print(f"{name}  ({size} 筆)")


def cmd_model_list(mgr: DatasetManager, args: argparse.Namespace) -> None:
    models = mgr.list_models(args.dataset)
    if not models:
        print(f"dataset '{args.dataset}' 目前沒有 response。")
        return
    for m in models:
        print(m)


def cmd_annotation_list(mgr: DatasetManager, args: argparse.Namespace) -> None:
    strategies = mgr.list_annotation_strategies(args.dataset)
    if not strategies:
        print(f"dataset '{args.dataset}' 目前沒有 annotation。")
        return
    for strategy, models in strategies.items():
        print(f"[{strategy}]")
        for m in models:
            print(f"  {m}")


def cmd_response_gen(args: argparse.Namespace) -> None:
    from .inferencer import Inferencer

    config_path = args.config or (Path(__file__).parent.parent / "config.yaml")
    inf = Inferencer(config_path=config_path, concurrency=args.concurrency)
    inf.generate(args.dataset, args.model, overwrite=args.overwrite)


def cmd_annotation_gen(args: argparse.Namespace) -> None:
    from .annotator import AnnotationRunner
    from .inferencer import load_config

    config_path = args.config or (Path(__file__).parent.parent / "config.yaml")
    config = load_config(config_path)

    from .manager import DatasetManager
    mgr = DatasetManager(base_path=config.get("data_path"))

    # 依 strategy 建立對應的 annotator
    annotator = _build_annotator(args, config)

    runner = AnnotationRunner(mgr=mgr, concurrency=args.concurrency)
    runner.run(annotator, args.dataset, args.model, args.strategy, overwrite=args.overwrite)


def _build_annotator(args: argparse.Namespace, config: dict):
    """依 --strategy 建立對應的 annotator 實例。"""
    strategy = args.strategy

    if strategy == "llm":
        from .annotator import LLMJudgeAnnotator
        from litellm import Router
        if not args.judge:
            print("錯誤：--strategy llm 需要指定 --judge <model_name>", file=sys.stderr)
            sys.exit(1)
        router = Router(model_list=config["model_list"])
        return LLMJudgeAnnotator(router=router, judge=args.judge)

    # 其他 strategy 在此擴充，例如：
    # if strategy == "cost":
    #     from .annotator import CostAnnotator
    #     return CostAnnotator(price_per_token=args.price_per_token)

    print(f"錯誤：未知 strategy '{strategy}'。目前支援：llm", file=sys.stderr)
    sys.exit(1)


def cmd_router_prepare(mgr: DatasetManager, args: argparse.Namespace) -> None:
    from .router import DataPreparer
    from .scorer import get as get_scorer

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]

    try:
        scorer = get_scorer(args.scorer)
    except KeyError as e:
        print(f"錯誤：{e}", file=sys.stderr)
        sys.exit(1)

    preparer = DataPreparer()
    data = preparer.from_manager(
        mgr,
        datasets=datasets,
        models=models,
        strategies=strategies,
        scorer=scorer,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    data = _preprocess_data(data, args)

    out = Path(args.output)
    data.save(out)
    print(
        f"已儲存 RouterData → {out}\n"
        f"  strategy: {strategies}\n"
        f"  模型: {data.models}\n"
        f"  train={len(data.train_prompt)}, val={len(data.val_prompt)}, test={len(data.test_prompt)}"
    )


_ROUTER_TYPES = {
    "oracle": "OracleRouter",
    "random": "RandomRouter",
    "knn": "KNNRouter",
    "mf": "MFRouter",
    "sw": "SWRankingRouter",
    "roberta": "RoBERTaMLCRouter",
}


def _add_preprocess_args(p: argparse.ArgumentParser) -> None:
    """共用預處理參數：去重 + 鑑別度篩選。"""
    p.add_argument(
        "--min-var",
        dest="min_var",
        type=float,
        default=None,
        help="過濾 np.var(scores) ≤ min_var 的訓練樣本（不指定則不套用）",
    )
    p.add_argument(
        "--dedup-eps",
        dest="dedup_eps",
        type=float,
        default=None,
        help="DBSCAN eps，去除語義重複的訓練樣本，預設 0.15（不指定則不套用）",
    )
    p.add_argument(
        "--dedup-emb-model",
        dest="dedup_emb_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="去重用的嵌入模型（預設 all-MiniLM-L6-v2）",
    )


def _preprocess_data(data, args: argparse.Namespace):
    """依 args 套用 filter_by_variance 和 deduplicate_train。"""
    if getattr(args, "min_var", None) is not None:
        data = data.filter_by_variance(min_var=args.min_var)
    if getattr(args, "dedup_eps", None) is not None:
        data = data.deduplicate_train(
            eps=args.dedup_eps,
            emb_model=args.dedup_emb_model,
        )  # metric 固定 "cosine"，min_samples 固定 2（與預設一致）
    return data


def _router_cls_kwargs(router_type: str, args: argparse.Namespace):
    """回傳 (RouterClass, kwargs_dict)，不實例化。"""
    from . import router as router_mod

    cls = getattr(router_mod, _ROUTER_TYPES[router_type])
    kwargs: dict = {}
    if router_type == "knn":
        kwargs = {"k": args.k, "emb_model": args.emb_model}
    elif router_type == "mf":
        kwargs = {
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "emb_model": args.emb_model,
        }
    elif router_type == "sw":
        kwargs = {"k": args.k, "temperature": args.temperature, "emb_model": args.emb_model}
    elif router_type == "roberta":
        kwargs = {"model_name": args.roberta_model, "epochs": args.epochs}
    return cls, kwargs


def _build_router(router_type: str, args: argparse.Namespace):
    cls, kwargs = _router_cls_kwargs(router_type, args)
    return cls(**kwargs)


def cmd_router_train(args: argparse.Namespace) -> None:
    import pickle
    from .router import RouterData

    data = RouterData.load(args.data)
    data = _preprocess_data(data, args)
    r = _build_router(args.router_type, args)
    r.fit(data)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            pickle.dump(r, f)
        print(f"Router 已儲存 → {out}")


def cmd_router_bench(args: argparse.Namespace) -> None:
    """
    固定 test set，對一或多個 router 跑多種訓練資料大小的對比實驗。
    每個 (router, size) 組合重複多個隨機種子（repeats），結果取平均。
    """
    from .router import RouterData, RouterBenchmark

    data = RouterData.load(args.data)
    data = _preprocess_data(data, args)  # 預處理只做一次，之後 subsample 都從乾淨資料來

    router_types = [t.strip() for t in args.router_types.split(",") if t.strip()]
    for rtype in router_types:
        if rtype not in _ROUTER_TYPES:
            print(
                f"錯誤：未知 router 類型 '{rtype}'。可用：{', '.join(_ROUTER_TYPES)}",
                file=sys.stderr,
            )
            sys.exit(1)

    seeds = list(range(args.repeats))

    if args.sizes:
        sizes: list = [int(s) for s in args.sizes.split(",") if s.strip()]
    else:
        sizes = [float(f) for f in args.fractions.split(",") if f.strip()]

    bench = RouterBenchmark(data)
    for rtype in router_types:
        cls, kwargs = _router_cls_kwargs(rtype, args)
        bench.run(cls, kwargs, sizes=sizes, seeds=seeds, label=rtype)

    print(
        f"\nBenchmark  |  routers={','.join(router_types)}"
        f"  |  test={len(data.test_prompt)}"
        f"  |  full_train={len(data.train_prompt)}"
        f"  |  repeats={args.repeats}"
    )
    bench.print_table()


def cmd_router_eval(args: argparse.Namespace) -> None:
    import pickle
    from .router import RouterData, OracleRouter, RandomRouter

    data = RouterData.load(args.data)
    router_type = args.router_type

    if router_type in ("oracle", "random"):
        r = _build_router(router_type, args)
        r.fit(data)
    elif args.model:
        with open(args.model, "rb") as f:
            r = pickle.load(f)
    else:
        print("錯誤: 非 oracle/random router 需要 --model 指定已訓練的 router 檔案。", file=sys.stderr)
        sys.exit(1)

    metrics = r.evaluate(data)
    print(f"Router type : {router_type}")
    print(f"METRIC_MU   : {metrics['mu']:.4f}")
    print(f"METRIC_VB   : {metrics['vb']:.4f}")
    print(f"METRIC_EP   : {metrics['ep']:.4f}")
    print(f"METRIC_TOKEN: {metrics['avg_tokens']:.2f}")
    print(f"METRIC_LAT  : {metrics['avg_latency']:.4f}")


def cmd_extract(mgr: DatasetManager, args: argparse.Namespace) -> None:
    from .scorer import get as get_scorer

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]

    try:
        scorer = get_scorer(args.scorer)
    except KeyError as e:
        print(f"錯誤：{e}", file=sys.stderr)
        sys.exit(1)

    data = mgr.extract(
        datasets=datasets,
        models=models,
        strategies=strategies,
        scorer=scorer,
        include_response=not args.no_response,
    )

    if not data:
        print("沒有符合條件的資料（請確認 dataset/model/strategy 是否有交集）。", file=sys.stderr)
        sys.exit(1)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"已輸出 {len(data)} 筆 → {out}")
    else:
        _print_jsonl(data)
        print(f"\n# 共 {len(data)} 筆", file=sys.stderr)


# ── parser setup ─────────────────────────────────────────────────────────────


def _add_router_args(p: argparse.ArgumentParser) -> None:
    """共用 router 超參數（各 router 類型只使用其中相關的）。"""
    p.add_argument("--k", type=int, default=10, help="KNN / SW: 近鄰數量")
    p.add_argument("--temperature", type=float, default=0.1, help="SW: Softmax 溫度")
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=64, help="MF: 隱空間維度")
    p.add_argument("--epochs", type=int, default=50, help="MF / RoBERTa: 訓練 epochs")
    p.add_argument("--lr", type=float, default=0.001, help="MF: 學習率")
    p.add_argument(
        "--emb-model",
        dest="emb_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="KNN / MF / SW: 嵌入模型名稱",
    )
    p.add_argument("--roberta-model", dest="roberta_model", default="roberta-base", help="RoBERTa: 模型名稱")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m LLMRouter",
        description="LLMRouter dataset 管理工具",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.yaml 路徑（預設自動偵測）",
    )
    parser.add_argument(
        "--base",
        default=None,
        help="data 根目錄，優先於 config.yaml 的 data_path",
    )
    sub = parser.add_subparsers(dest="command", metavar="<subcommand>")
    sub.required = True

    # ── scorer list ───────────────────────────────────────────────────────
    p_sc = sub.add_parser("scorer", help="scorer registry 操作")
    sc_sub = p_sc.add_subparsers(dest="sc_action", metavar="<action>")
    sc_sub.required = True
    sc_sub.add_parser("list", help="列出所有已註冊的 scorer 名稱")

    # ── dataset list ──────────────────────────────────────────────────────
    p_dl = sub.add_parser("dataset", help="dataset 相關操作")
    ds_sub = p_dl.add_subparsers(dest="ds_action", metavar="<action>")
    ds_sub.required = True
    ds_sub.add_parser("list", help="列出所有 dataset")

    # ── model list ────────────────────────────────────────────────────────
    p_ml = sub.add_parser("model", help="model response 相關操作")
    ml_sub = p_ml.add_subparsers(dest="ml_action", metavar="<action>")
    ml_sub.required = True
    p_ml_list = ml_sub.add_parser("list", help="列出某 dataset 的 response model")
    p_ml_list.add_argument("dataset", help="dataset 名稱")

    # ── annotation list ───────────────────────────────────────────────────
    p_al = sub.add_parser("annotation", help="annotation 相關操作")
    al_sub = p_al.add_subparsers(dest="al_action", metavar="<action>")
    al_sub.required = True

    p_al_list = al_sub.add_parser("list", help="列出某 dataset 的 annotation")
    p_al_list.add_argument("dataset", help="dataset 名稱")

    p_al_gen = al_sub.add_parser("gen", help="對 model response 執行 annotation 並儲存")
    p_al_gen.add_argument("dataset", help="dataset 名稱")
    p_al_gen.add_argument("model", help="被評估的 model 名稱")
    p_al_gen.add_argument("--strategy", required=True,
                          help="annotator 類型：llm（LLM-as-Judge）、cost、…")
    # llm strategy 專用
    p_al_gen.add_argument("--judge", default=None,
                          help="[strategy=llm] judge LLM 的 model_name（需在 config.yaml 中）")
    p_al_gen.add_argument("--concurrency", type=int, default=8, help="同時進行的請求數（預設 8）")
    p_al_gen.add_argument("--overwrite", action="store_true", help="重新跑全部，不做斷點續跑")

    # ── response gen ──────────────────────────────────────────────────────
    p_ra = sub.add_parser("response", help="response 相關操作")
    ra_sub = p_ra.add_subparsers(dest="ra_action", metavar="<action>")
    ra_sub.required = True

    p_ra_list = ra_sub.add_parser("list", help="列出某 dataset 的 response model")
    p_ra_list.add_argument("dataset", help="dataset 名稱")

    p_ra_gen = ra_sub.add_parser("gen", help="用 litellm 對 dataset 做 inference 並儲存 response")
    p_ra_gen.add_argument("dataset", help="dataset 名稱")
    p_ra_gen.add_argument("model", help="config.yaml 中的 model_name")
    p_ra_gen.add_argument("--concurrency", type=int, default=8, help="同時進行的請求數（預設 8）")
    p_ra_gen.add_argument("--overwrite", action="store_true", help="重新跑全部，不做斷點續跑")

    # ── extract ───────────────────────────────────────────────────────────
    p_ex = sub.add_parser("extract", help="抽取訓練資料")
    p_ex.add_argument("--datasets", required=True, help="逗號分隔的 dataset 名稱清單")
    p_ex.add_argument("--models", required=True, help="逗號分隔的 model 名稱清單")
    p_ex.add_argument("--strategy", required=True,
                      help="annotation strategy，可逗號分隔多個（第一個為主要）。e.g. llm 或 llm,cost")
    p_ex.add_argument("--scorer", default="point",
                      help="scorer 名稱（需已在 registry 中，預設 point）")
    p_ex.add_argument("--output", "-o", default=None, help="輸出 JSONL 路徑（不指定則印到 stdout）")
    p_ex.add_argument("--no-response", action="store_true", help="輸出中不包含 response 文字")

    # ── router ────────────────────────────────────────────────────────────
    p_rt = sub.add_parser("router", help="router 訓練與評估")
    rt_sub = p_rt.add_subparsers(dest="rt_action", metavar="<action>")
    rt_sub.required = True

    # router prepare
    p_rt_prep = rt_sub.add_parser("prepare", help="將資料準備成 RouterData (.npz)")
    p_rt_prep.add_argument("--datasets", required=True, help="逗號分隔的 dataset 名稱清單")
    p_rt_prep.add_argument("--models", required=True, help="逗號分隔的 model 名稱清單")
    p_rt_prep.add_argument("--strategy", required=True,
                           help="annotation strategy，可逗號分隔多個（第一個為主要）。e.g. llm 或 llm,cost")
    p_rt_prep.add_argument("--scorer", default="point",
                           help="scorer 名稱（需已在 registry 中，預設 point）")
    p_rt_prep.add_argument("--output", "-o", required=True, help="輸出 .npz 路徑")
    p_rt_prep.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.6, help="訓練集比例（預設 0.6）")
    p_rt_prep.add_argument("--val-ratio", dest="val_ratio", type=float, default=0.1, help="驗證集比例（預設 0.1）；剩餘為 test")
    p_rt_prep.add_argument("--seed", type=int, default=42, help="分割隨機種子")
    _add_preprocess_args(p_rt_prep)

    # router train
    p_rt_train = rt_sub.add_parser("train", help="訓練 router 並儲存")
    p_rt_train.add_argument(
        "router_type", choices=list(_ROUTER_TYPES), help="router 類型"
    )
    p_rt_train.add_argument("--data", required=True, help="RouterData .npz 路徑")
    p_rt_train.add_argument("--output", "-o", default=None, help="儲存 router 的 .pkl 路徑")
    _add_router_args(p_rt_train)
    _add_preprocess_args(p_rt_train)

    # router bench
    p_rt_bench = rt_sub.add_parser(
        "bench", help="固定 test set，對一或多個 router 跑訓練資料縮放實驗"
    )
    p_rt_bench.add_argument(
        "router_types",
        help=f"逗號分隔的 router 類型（e.g. knn 或 knn,mf,oracle）；可用：{','.join(_ROUTER_TYPES)}",
    )
    p_rt_bench.add_argument("--data", required=True, help="RouterData .npz 路徑")
    bench_group = p_rt_bench.add_mutually_exclusive_group()
    bench_group.add_argument(
        "--fractions",
        default="0.1,0.2,0.3,0.5,0.7,1.0",
        help="逗號分隔的訓練資料比例（預設 0.1,0.2,0.3,0.5,0.7,1.0）",
    )
    bench_group.add_argument(
        "--sizes",
        default=None,
        help="逗號分隔的固定訓練資料筆數（e.g. 50,100,200,500）",
    )
    p_rt_bench.add_argument(
        "--repeats", type=int, default=3, help="每個大小重複幾次（不同隨機種子，預設 3）"
    )
    _add_router_args(p_rt_bench)
    _add_preprocess_args(p_rt_bench)

    # router eval
    p_rt_eval = rt_sub.add_parser("eval", help="評估 router 效能")
    p_rt_eval.add_argument(
        "router_type", choices=list(_ROUTER_TYPES), help="router 類型"
    )
    p_rt_eval.add_argument("--data", required=True, help="RouterData .npz 路徑")
    p_rt_eval.add_argument("--model", default=None, help="已訓練的 router .pkl 路徑（oracle/random 不需要）")
    _add_router_args(p_rt_eval)

    return parser


# ── main ──────────────────────────────────────────────────────────────────────


def _resolve_dataset_path(args: argparse.Namespace) -> Optional[str]:
    """--base > config.yaml dataset_path > DatasetManager fallback"""
    if args.base:
        return args.base
    from .inferencer import load_config
    config_path = args.config or (Path(__file__).parent.parent / "config.yaml")
    try:
        config = load_config(config_path)
        return config.get("data_path")
    except Exception:
        return None


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    mgr = DatasetManager(base_path=_resolve_dataset_path(args))

    try:
        if args.command == "scorer":
            if args.sc_action == "list":
                cmd_scorer_list()

        elif args.command == "dataset":
            if args.ds_action == "list":
                cmd_dataset_list(mgr, args)

        elif args.command == "model":
            if args.ml_action == "list":
                cmd_model_list(mgr, args)

        elif args.command == "annotation":
            if args.al_action == "list":
                cmd_annotation_list(mgr, args)
            elif args.al_action == "gen":
                cmd_annotation_gen(args)

        elif args.command == "response":
            if args.ra_action == "list":
                cmd_model_list(mgr, args)
            elif args.ra_action == "gen":
                cmd_response_gen(args)

        elif args.command == "extract":
            cmd_extract(mgr, args)

        elif args.command == "router":
            if args.rt_action == "prepare":
                cmd_router_prepare(mgr, args)
            elif args.rt_action == "train":
                cmd_router_train(args)
            elif args.rt_action == "eval":
                cmd_router_eval(args)
            elif args.rt_action == "bench":
                cmd_router_bench(args)

    except (FileNotFoundError, FileExistsError) as e:
        print(f"錯誤: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
