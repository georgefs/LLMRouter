from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from tqdm.asyncio import tqdm

from ..manager import DatasetManager


class BaseAnnotator(ABC):
    """
    Annotator 基底類別。

    子類別只需實作 annotate_one()，其餘 IO / 斷點續跑 / 進度條
    由 AnnotationRunner 統一處理。
    """

    @abstractmethod
    async def annotate_one(
        self,
        key: str,
        dataset_item: dict,
        response: str,
        sem: asyncio.Semaphore,
    ) -> Optional[dict]:
        """
        對單筆 response 產生補充資訊。

        Args:
            key: 樣本唯一識別碼
            dataset_item: 原始 dataset 記錄（包含 question, answer, instruction_id_list 等所有欄位）
            response: model 的回應文字
            sem: 由 AnnotationRunner 傳入的 concurrency 控制信號量

        Returns:
            包含至少 {key} 的 dict，其餘欄位由各子類別自行定義：
              - LLMJudgeAnnotator  → {key, point, verdict, brief_reason, judge, raw, judge_time}
              - CostAnnotator      → {key, cost, input_tokens, output_tokens}
              - 自訂 annotator     → 任意欄位均可
            失敗時回傳 None（該筆將被略過）。
        """


class AnnotationRunner:
    """
    Annotation 執行器：負責資料 IO、斷點續跑、非同步排程、進度條。

    與具體評分邏輯（BaseAnnotator 子類別）解耦，可自由替換評分策略。

    Usage::

        runner = AnnotationRunner(mgr, concurrency=8)
        annotator = LLMJudgeAnnotator(router, judge="gpt-4o")
        runner.run(annotator, dataset="arc_challenge_train",
                   model="Llama-3.1-70B", strategy="llm")
    """

    def __init__(self, mgr: DatasetManager, concurrency: int = 8) -> None:
        self.mgr = mgr
        self.concurrency = concurrency

    async def arun(
        self,
        annotator: BaseAnnotator,
        dataset: str,
        model: str,
        strategy: str,
        overwrite: bool = False,
    ) -> List[dict]:
        # ── 載入 dataset ──────────────────────────────────────────────
        ds_items = {item["key"]: item for item in self.mgr.get_dataset(dataset)}

        # ── 載入 responses ────────────────────────────────────────────
        try:
            responses = {r["key"]: r for r in self.mgr.get_responses(dataset, model)}
        except FileNotFoundError:
            raise FileNotFoundError(
                f"找不到 response：dataset={dataset}, model={model}\n"
                f"請先執行：response gen {dataset} {model}"
            )

        # ── 斷點續跑：略過已完成的 key ────────────────────────────────
        existing: dict[str, dict] = {}
        if not overwrite:
            try:
                for row in self.mgr.get_annotations(dataset, model, strategy):
                    existing[row["key"]] = row
                if existing:
                    print(f"已有 {len(existing)} 筆 annotation，跳過已完成的 key...")
            except FileNotFoundError:
                pass

        todo = [
            {
                "key": k,
                "ds_item": ds_items[k],
                "response": responses[k]["text"],
            }
            for k in responses
            if k in ds_items and k not in existing
        ]

        target = self.mgr.annotations_path / dataset / strategy / f"{model}.jsonl"
        print(f"目標路徑：{target}")

        if not todo:
            print("所有資料已完成，無需重新 annotation。")
            return []

        print(f"開始 annotation：dataset={dataset}  model={model}  strategy={strategy}  共 {len(todo)} 筆")

        # ── 非同步執行 ────────────────────────────────────────────────
        sem = asyncio.Semaphore(self.concurrency)

        async def _with_progress(item: dict, bar: tqdm) -> Optional[dict]:
            result = await annotator.annotate_one(
                item["key"], item["ds_item"], item["response"], sem
            )
            bar.update(1)
            return result

        with tqdm(total=len(todo), unit="筆") as bar:
            tasks = [_with_progress(item, bar) for item in todo]
            new_results: List[dict] = [r for r in await asyncio.gather(*tasks) if r is not None]

        # ── 合併並儲存 ────────────────────────────────────────────────
        key_order = {k: i for i, k in enumerate(responses.keys())}
        merged = list(existing.values()) + new_results
        merged.sort(key=lambda r: key_order.get(r["key"], 0))

        path, added, _ = self.mgr.add_annotations(dataset, model, strategy, merged, overwrite=True)
        print(f"已儲存 {len(merged)} 筆 annotation → {path}  (新增 {added} 筆)")

        return new_results

    def run(
        self,
        annotator: BaseAnnotator,
        dataset: str,
        model: str,
        strategy: str,
        overwrite: bool = False,
    ) -> List[dict]:
        """同步版本（封裝 asyncio.run）。"""
        async def _run_and_cleanup() -> List[dict]:
            result = await self.arun(annotator, dataset, model, strategy, overwrite=overwrite)
            # litellm 在每次 acompletion() 後用 create_task() 建立 logging 背景 task，
            # 這些 task 未被 await。asyncio.run() 結束時會嘗試取消並等待它們，
            # 但若它們正在等網路 I/O 則會卡住。在此主動取消並等待，讓事件迴圈乾淨退出。
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            return result

        return asyncio.run(_run_and_cleanup())
