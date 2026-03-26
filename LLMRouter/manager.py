from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .scorer import BaseScorer


def _read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


class DatasetManager:
    """
    管理 datasets / responses / annotations 三層資料的存取與抽取。

    目錄結構假設:
        <base>/
          datasets/   — 原始 JSONL (key, question, answer, ...)
          responses/  — <dataset>/<model>.jsonl (key, text, usage, ...)
          annotations/— <dataset>/<strategy>/<model>.jsonl (key, point, meta?)
    """

    def __init__(self, base_path: Optional[str] = None) -> None:
        if base_path is None:
            # 開發期 fallback：repo 根目錄下的 datasets/
            base_path = Path(__file__).parent.parent / "datasets"
        self.base = Path(base_path)
        self.datasets_path = self.base / "datasets"
        self.responses_path = self.base / "responses"
        self.annotations_path = self.base / "annotations"

    # ------------------------------------------------------------------
    # List operations
    # ------------------------------------------------------------------

    def list_datasets(self) -> List[str]:
        """列出所有可用的 dataset 名稱。"""
        return sorted(p.stem for p in self.datasets_path.glob("*.jsonl"))

    def list_models(self, dataset: str) -> List[str]:
        """列出某 dataset 已有 response 的 model 名稱。"""
        path = self.responses_path / dataset
        if not path.exists():
            return []
        return sorted(p.stem for p in path.glob("*.jsonl"))

    def list_annotation_strategies(self, dataset: str) -> Dict[str, List[str]]:
        """
        列出某 dataset 的所有 annotation strategy 及其對應 model 清單。
        回傳: {strategy: [model, ...]}
        """
        path = self.annotations_path / dataset
        if not path.exists():
            return {}
        result: Dict[str, List[str]] = {}
        for strategy_dir in sorted(path.iterdir()):
            if not strategy_dir.is_dir():
                continue
            models = sorted(p.stem for p in strategy_dir.glob("*.jsonl"))
            if models:
                result[strategy_dir.name] = models
        return result

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_dataset(self, dataset: str) -> List[dict]:
        """讀取原始 dataset。"""
        path = self.datasets_path / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return _read_jsonl(path)

    def get_responses(self, dataset: str, model: str) -> List[dict]:
        """讀取某 dataset / model 的所有 response。"""
        path = self.responses_path / dataset / f"{model}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Responses not found: {path}")
        return _read_jsonl(path)

    def get_annotations(self, dataset: str, model: str, strategy: str) -> List[dict]:
        """讀取某 dataset / strategy / model 的所有 annotation。"""
        path = self.annotations_path / dataset / strategy / f"{model}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Annotations not found: {path}")
        return _read_jsonl(path)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_responses(
        self,
        dataset: str,
        model: str,
        data: List[dict],
        overwrite: bool = False,
    ) -> tuple[Path, int, int]:
        """
        新增 model response。
        data 每筆需包含 key, text；可附帶 usage, response_time 等欄位。

        - overwrite=False（預設）：合併，已存在的 key 自動跳過（續傳）
        - overwrite=True：完整覆蓋

        回傳 (path, added_count, skipped_count)
        """
        path = self.responses_path / dataset / f"{model}.jsonl"
        if overwrite or not path.exists():
            _write_jsonl(path, data)
            return path, len(data), 0

        existing = {item["key"]: item for item in _read_jsonl(path)}
        new_items = [item for item in data if item["key"] not in existing]
        if new_items:
            _write_jsonl(path, list(existing.values()) + new_items)
        return path, len(new_items), len(existing)



    def add_annotations(
        self,
        dataset: str,
        model: str,
        strategy: str,
        data: List[dict],
        overwrite: bool = False,
    ) -> tuple[Path, int, int]:
        """
        新增 annotation。
        data 每筆需包含 key, point；可附帶 meta 等欄位。

        - overwrite=False（預設）：合併，已存在的 key 自動跳過（續傳）
        - overwrite=True：完整覆蓋

        回傳 (path, added_count, skipped_count)
        """
        path = self.annotations_path / dataset / strategy / f"{model}.jsonl"
        if overwrite or not path.exists():
            _write_jsonl(path, data)
            return path, len(data), 0

        existing = {item["key"]: item for item in _read_jsonl(path)}
        new_items = [item for item in data if item["key"] not in existing]
        if new_items:
            _write_jsonl(path, list(existing.values()) + new_items)
        return path, len(new_items), len(existing)

    # ------------------------------------------------------------------
    # Extract operation
    # ------------------------------------------------------------------

    def extract(
        self,
        datasets: List[str],
        models: List[str],
        strategies: "str | List[str]",
        scorer: "BaseScorer | None" = None,
        include_response: bool = True,
    ) -> List[dict]:
        """
        抽取訓練 router 所需的組合資料。

        對每個 (dataset, model) 組合，join dataset / responses / annotations，
        只保留在「第一個 strategy」中有 annotation 的 key。
        各 strategy 的資料以 strategy 名稱為 key 獨立保存於 annotations dict。

        回傳格式:
            {key, dataset, question, answer, model, score, annotations, response?}

        其中 `score` 由 scorer 計算，預設為 FieldScorer("point")（從 annotations 取 point 欄位）。
        scorer 可存取完整的 dataset_item、response、annotations，實現任意評分邏輯。

        Args:
            datasets: 要抽取的 dataset 名稱清單
            models: 要抽取的 model 名稱清單
            strategies: annotation strategy 名稱，或多個 strategy 的清單。
                        第一個 strategy 為主要 strategy（決定哪些 key 納入）。
            scorer: 評分策略實例（預設 FieldScorer("point")）
            include_response: 是否在結果中附上 model 的原始回應文字
        """
        from .scorer import FieldScorer
        if scorer is None:
            scorer = FieldScorer("point")
        if isinstance(strategies, str):
            strategies = [strategies]

        primary = strategies[0]
        extras = strategies[1:]

        result: List[dict] = []

        for dataset in datasets:
            try:
                ds_items: Dict[str, dict] = {
                    item["key"]: item for item in self.get_dataset(dataset)
                }
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Cannot extract: {e}") from e

            for model in models:
                resp_path = self.responses_path / dataset / f"{model}.jsonl"
                anno_path = self.annotations_path / dataset / primary / f"{model}.jsonl"

                if not resp_path.exists() or not anno_path.exists():
                    continue

                responses: Dict[str, dict] = {
                    item["key"]: item for item in _read_jsonl(resp_path)
                }
                primary_annos: Dict[str, dict] = {
                    item["key"]: item for item in _read_jsonl(anno_path)
                }

                # 載入額外 strategy 的 annotation（key 不存在時略過，不強制要求）
                extra_annos: List[Dict[str, dict]] = []
                for strat in extras:
                    p = self.annotations_path / dataset / strat / f"{model}.jsonl"
                    if p.exists():
                        extra_annos.append({
                            item["key"]: item for item in _read_jsonl(p)
                        })
                    else:
                        extra_annos.append({})

                for key, anno in primary_annos.items():
                    if key not in ds_items or key not in responses:
                        continue

                    # 各 strategy 的資料獨立保存（不 merge），以 strategy 名稱為 key
                    annotations_by_strategy: dict = {
                        primary: {k: v for k, v in anno.items() if k != "key"}
                    }
                    for strat, ea in zip(extras, extra_annos):
                        if key in ea:
                            annotations_by_strategy[strat] = {
                                k: v for k, v in ea[key].items() if k != "key"
                            }

                    score_val = scorer(ds_items[key], responses[key], annotations_by_strategy)

                    row: dict = {
                        "key": key,
                        "dataset": dataset,
                        "question": ds_items[key]["question"],
                        "answer": ds_items[key].get("answer", ""),
                        "model": model,
                        "score": score_val,
                        "annotations": annotations_by_strategy,
                    }
                    if include_response:
                        row["response"] = responses[key]["text"]
                    result.append(row)

        return result
