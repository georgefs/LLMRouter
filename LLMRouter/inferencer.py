"""
Inferencer：讀取 config.yaml，透過 litellm Router 對 dataset 做批次 inference，
結果存入 datasets/responses/{dataset}/{model}.jsonl。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

import yaml

from .manager import DatasetManager, _read_jsonl, _write_jsonl


# ── config loading ────────────────────────────────────────────────────────────


def _env_constructor(loader: yaml.SafeLoader, node: yaml.ScalarNode) -> str:
    """展開 !ENV ${VAR} → os.environ['VAR']"""
    value: str = loader.construct_scalar(node)
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), f"${{{m.group(1)}}}"),
        value,
    )


yaml.add_constructor("!ENV", _env_constructor, Loader=yaml.SafeLoader)


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Inferencer ────────────────────────────────────────────────────────────────


class Inferencer:
    """
    Args:
        config_path: litellm config.yaml 路徑（預設 repo 根目錄下的 config.yaml）
        concurrency: 同時進行的非同步請求數上限
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        concurrency: int = 8,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        config = load_config(config_path)

        from litellm import Router  # lazy import

        self.router = Router(model_list=config["model_list"])
        dataset_path = config.get("data_path")
        self.mgr = DatasetManager(base_path=dataset_path)
        self.concurrency = concurrency

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    async def _call(
        self,
        model: str,
        key: str,
        question: str,
        sem: asyncio.Semaphore,
    ) -> Optional[dict]:
        async with sem:
            t0 = time.time()
            try:
                resp = await self.router.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                )
                elapsed = round(time.time() - t0, 6)
                usage = resp.usage
                return {
                    "key": key,
                    "text": resp.choices[0].message.content,
                    "usage": {
                        "input_tokens": usage.prompt_tokens,
                        "output_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                    "response_time": elapsed,
                }
            except Exception as exc:
                print(f"[error] {key}: {exc}")
                return None

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    async def agenerate(
        self,
        dataset: str,
        model: str,
        overwrite: bool = False,
    ) -> List[dict]:
        """
        對 dataset 中所有問題呼叫 model，回傳新產生的 response list。

        - 若 response 檔已存在且 overwrite=False，自動跳過已完成的 key（斷點續跑）。
        - 完成後自動呼叫 DatasetManager.add_responses 存檔。
        """
        items: List[dict] = self.mgr.get_dataset(dataset)

        # 計算需要跑的 key
        existing: dict[str, dict] = {}
        if not overwrite:
            try:
                for row in self.mgr.get_responses(dataset, model):
                    existing[row["key"]] = row
                if existing:
                    print(f"已有 {len(existing)} 筆 response，跳過已完成的 key...")
            except FileNotFoundError:
                pass

        todo = [item for item in items if item["key"] not in existing]

        if not todo:
            print("所有資料已完成，無需重新 inference。")
            return []

        target = self.mgr.responses_path / dataset / f"{model}.jsonl"
        print(f"目標路徑：{target}")
        print(f"開始 inference：dataset={dataset}  model={model}  共 {len(todo)} 筆")

        from tqdm.asyncio import tqdm

        sem = asyncio.Semaphore(self.concurrency)

        async def _call_with_progress(item: dict, bar: tqdm) -> Optional[dict]:
            result = await self._call(model, item["key"], item["question"], sem)
            bar.update(1)
            return result

        with tqdm(total=len(todo), unit="筆") as bar:
            tasks = [_call_with_progress(item, bar) for item in todo]
            new_results: List[dict] = [r for r in await asyncio.gather(*tasks) if r is not None]

        # 合併 existing + new，以 dataset 原始順序排列
        key_order = {item["key"]: i for i, item in enumerate(items)}
        merged = list(existing.values()) + new_results
        merged.sort(key=lambda r: key_order.get(r["key"], 0))

        path, added, skipped = self.mgr.add_responses(dataset, model, merged, overwrite=True)
        print(f"已儲存 {len(merged)} 筆 response → {path}")

        return new_results

    def generate(self, dataset: str, model: str, overwrite: bool = False) -> List[dict]:
        """同步版本（封裝 asyncio.run）。"""
        async def _run_and_cleanup() -> List[dict]:
            result = await self.agenerate(dataset, model, overwrite=overwrite)
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            return result

        return asyncio.run(_run_and_cleanup())
