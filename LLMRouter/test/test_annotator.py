"""
annotator 模組單元測試

涵蓋：
  - _parse_score / _parse_meta（llm_judge 工具函式）
  - _describe_instruction / _build_instructions_text
  - LLMJudgeAnnotator.annotate_one（GT-based 與 IFEval-style）
  - AnnotationRunner.arun（基本流程、斷點續跑、overwrite）

執行: pytest LLMRouter/test/test_annotator.py -v
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from LLMRouter.scorer import BaseScorer, FieldScorer, get, list_scorers, register
from LLMRouter.annotator.llm_judge import (
    LLMJudgeAnnotator,
    _build_instructions_text,
    _describe_instruction,
    _parse_meta,
    _parse_score,
)
from LLMRouter.annotator.base import AnnotationRunner
from LLMRouter.manager import DatasetManager


# ── helpers ───────────────────────────────────────────────────────────────────


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_router_mock(content: str) -> MagicMock:
    """建立回傳固定內容的 litellm Router mock。"""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    router = MagicMock()
    router.acompletion = AsyncMock(return_value=resp)
    return router


# ── _parse_score ───────────────────────────────────────────────────────────────


class TestScorerRegistry:
    def test_default_point_registered(self):
        scorer = get("point")
        assert scorer is not None
        result = scorer({}, {}, {"llm": {"point": 0.8}})
        assert result == pytest.approx(0.8)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="未註冊"):
            get("__nonexistent__")

    def test_register_and_get(self):
        custom = FieldScorer("myscore")
        register("__test_myscore__", custom)
        assert get("__test_myscore__") is custom

    def test_register_returns_scorer(self):
        scorer = FieldScorer("x")
        result = register("__test_ret__", scorer)
        assert result is scorer

    def test_list_scorers_contains_point(self):
        assert "point" in list_scorers()

    def test_list_scorers_sorted(self):
        names = list_scorers()
        assert names == sorted(names)


class TestParseScore:
    def test_json_score_key(self):
        assert _parse_score('{"score": 0.8}') == 0.8

    def test_json_point_key(self):
        assert _parse_score('{"point": 0.5}') == 0.5

    def test_json_rating_key(self):
        assert _parse_score('{"rating": 1.0}') == 1.0

    def test_json_with_extra_fields(self):
        raw = '{"verdict": "correct", "score": 0.9, "brief_reason": "good"}'
        assert _parse_score(raw) == 0.9

    def test_regex_fallback(self):
        assert _parse_score('the score is "score": 0.75 here') == 0.75

    def test_plain_float_fallback(self):
        assert _parse_score("Score: 0.6 out of 1") == 0.6

    def test_clamp_above_one(self):
        assert _parse_score('{"score": 1.5}') == 1.0

    def test_clamp_below_zero(self):
        assert _parse_score('{"score": -0.3}') == 0.0

    def test_invalid_json_no_float(self):
        assert _parse_score("totally wrong output") == 0.0

    def test_integer_score(self):
        assert _parse_score('{"score": 1}') == 1.0


# ── _parse_meta ────────────────────────────────────────────────────────────────


class TestParseMeta:
    def test_valid_json(self):
        raw = '{"verdict": "correct", "score": 1.0, "brief_reason": "ok"}'
        meta = _parse_meta(raw)
        assert meta["verdict"] == "correct"
        assert meta["brief_reason"] == "ok"

    def test_invalid_json_returns_empty_dict(self):
        assert _parse_meta("not json at all") == {}

    def test_partial_json_returns_empty_dict(self):
        assert _parse_meta('{"verdict": "correct"') == {}


# ── _describe_instruction ──────────────────────────────────────────────────────


class TestDescribeInstruction:
    def test_known_id_no_comma(self):
        desc = _describe_instruction("punctuation:no_comma", {})
        assert "comma" in desc.lower()

    def test_known_id_with_params(self):
        desc = _describe_instruction(
            "length_constraints:number_words",
            {"relation": "at least", "num_words": 100},
        )
        assert "100" in desc
        assert "word" in desc.lower()

    def test_keywords_existence(self):
        desc = _describe_instruction(
            "keywords:existence",
            {"keywords": ["python", "code"]},
        )
        assert "python" in desc

    def test_unknown_id_returns_fallback(self):
        desc = _describe_instruction("unknown:whatever", {"foo": "bar"})
        assert "unknown:whatever" in desc

    def test_build_instructions_text_numbering(self):
        ids = ["punctuation:no_comma", "detectable_format:json_format"]
        kwargs = [{}, {}]
        text = _build_instructions_text(ids, kwargs)
        lines = text.strip().splitlines()
        assert lines[0].startswith("1.")
        assert lines[1].startswith("2.")


# ── LLMJudgeAnnotator ─────────────────────────────────────────────────────────


class TestLLMJudgeAnnotator:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def test_gt_based_returns_required_fields(self):
        raw = '{"verdict": "correct", "score": 0.9, "brief_reason": "looks good"}'
        router = _make_router_mock(raw)
        ann = LLMJudgeAnnotator(router, judge="test-judge")
        sem = asyncio.Semaphore(1)

        ds_item = {"key": "k0", "question": "Q?", "answer": "A"}
        result = self._run(ann.annotate_one("k0", ds_item, "model response", sem))

        assert result is not None
        assert result["key"] == "k0"
        assert result["point"] == pytest.approx(0.9)
        assert result["verdict"] == "correct"
        assert result["brief_reason"] == "looks good"
        assert result["judge"] == "test-judge"
        assert result["raw"] == raw
        assert "judge_time" in result

    def test_gt_based_uses_answer_from_dataset_item(self):
        raw = '{"score": 0.5}'
        router = _make_router_mock(raw)
        ann = LLMJudgeAnnotator(router, judge="j")
        sem = asyncio.Semaphore(1)

        ds_item = {"key": "k0", "question": "Q?", "answer": "THE ANSWER"}
        self._run(ann.annotate_one("k0", ds_item, "resp", sem))

        call_kwargs = router.acompletion.call_args
        prompt = call_kwargs[1]["messages"][0]["content"]
        assert "THE ANSWER" in prompt

    def test_ifeval_uses_format_prompt(self):
        raw = '{"verdict": "incorrect", "score": 0.0, "failed_instructions": ["no_comma"]}'
        router = _make_router_mock(raw)
        ann = LLMJudgeAnnotator(router, judge="j")
        sem = asyncio.Semaphore(1)

        ds_item = {
            "key": "k0",
            "question": "Write something",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        }
        result = self._run(ann.annotate_one("k0", ds_item, "Hello, world!", sem))

        assert result is not None
        assert result["point"] == 0.0
        assert result["verdict"] == "incorrect"

        call_kwargs = router.acompletion.call_args
        prompt = call_kwargs[1]["messages"][0]["content"]
        # Format prompt 包含 "Instructions to Verify"，不包含 "Gold Answer"
        assert "Instructions to Verify" in prompt
        assert "Gold Answer" not in prompt

    def test_ifeval_missing_kwargs_uses_empty_dicts(self):
        """instruction_id_list 有值但沒有 kwargs 欄位，應不崩潰。"""
        raw = '{"score": 1.0}'
        router = _make_router_mock(raw)
        ann = LLMJudgeAnnotator(router, judge="j")
        sem = asyncio.Semaphore(1)

        ds_item = {
            "key": "k0",
            "question": "Q",
            "instruction_id_list": ["punctuation:no_comma"],
        }
        result = self._run(ann.annotate_one("k0", ds_item, "resp", sem))
        assert result is not None
        assert result["point"] == 1.0

    def test_exception_returns_none(self):
        router = MagicMock()
        router.acompletion = AsyncMock(side_effect=RuntimeError("api error"))
        ann = LLMJudgeAnnotator(router, judge="j")
        sem = asyncio.Semaphore(1)

        ds_item = {"key": "k0", "question": "Q", "answer": "A"}
        result = self._run(ann.annotate_one("k0", ds_item, "resp", sem))
        assert result is None

    def test_no_answer_field_still_works(self):
        """dataset item 沒有 answer 欄位（純問答類），不應崩潰。"""
        raw = '{"score": 0.7}'
        router = _make_router_mock(raw)
        ann = LLMJudgeAnnotator(router, judge="j")
        sem = asyncio.Semaphore(1)

        ds_item = {"key": "k0", "question": "Open-ended Q?"}
        result = self._run(ann.annotate_one("k0", ds_item, "resp", sem))
        assert result is not None
        assert result["point"] == pytest.approx(0.7)


# ── AnnotationRunner ──────────────────────────────────────────────────────────


@pytest.fixture()
def base(tmp_path: Path) -> Path:
    for d in ("datasets", "responses", "annotations"):
        (tmp_path / d).mkdir()
    return tmp_path


@pytest.fixture()
def mgr(base: Path) -> DatasetManager:
    return DatasetManager(base_path=str(base))


def _write_dataset(base, ds_name, items):
    write_jsonl(base / "datasets" / f"{ds_name}.jsonl", items)


def _write_responses(base, ds_name, model, items):
    write_jsonl(base / "responses" / ds_name / f"{model}.jsonl", items)


def _write_annotations(base, ds_name, strategy, model, items):
    write_jsonl(base / "annotations" / ds_name / strategy / f"{model}.jsonl", items)


def _stub_annotator(point: float = 0.8):
    """建立固定回傳 point 的 annotator stub（不呼叫真實 LLM）。"""
    from LLMRouter.annotator.base import BaseAnnotator

    class _Stub(BaseAnnotator):
        async def annotate_one(self, key, dataset_item, response, sem):
            return {
                "key": key,
                "point": point,
                "verdict": "correct",
                "brief_reason": "stub",
            }

    return _Stub()


class TestAnnotationRunner:
    def _run(self, coro):
        return asyncio.get_event_loop().run_until_complete(coro)

    def _setup_data(self, base, ds="gsm8k", model="ModelA", strategy="llm", n=3):
        _write_dataset(base, ds, [
            {"key": f"{ds}_{i}", "question": f"Q{i}", "answer": f"A{i}"}
            for i in range(n)
        ])
        _write_responses(base, ds, model, [
            {"key": f"{ds}_{i}", "text": f"R{i}"}
            for i in range(n)
        ])

    def test_basic_run(self, base, mgr):
        self._setup_data(base)
        runner = AnnotationRunner(mgr, concurrency=4)
        results = runner.run(_stub_annotator(), "gsm8k", "ModelA", "llm")

        assert len(results) == 3
        assert all(r["point"] == 0.8 for r in results)
        assert all("key" in r for r in results)

        # 確認寫入磁碟
        saved = mgr.get_annotations("gsm8k", "ModelA", "llm")
        assert len(saved) == 3

    def test_resume_skips_existing(self, base, mgr):
        self._setup_data(base)
        # 預先寫入 gsm8k_0 的 annotation
        _write_annotations(base, "gsm8k", "llm", "ModelA", [
            {"key": "gsm8k_0", "point": 0.5}
        ])

        runner = AnnotationRunner(mgr, concurrency=4)
        results = runner.run(_stub_annotator(1.0), "gsm8k", "ModelA", "llm")

        # 只新增 2 筆（gsm8k_1, gsm8k_2）
        assert len(results) == 2
        # 最終檔案共 3 筆
        saved = {r["key"]: r for r in mgr.get_annotations("gsm8k", "ModelA", "llm")}
        assert len(saved) == 3
        # 舊的 gsm8k_0 仍是 0.5（未被覆蓋）
        assert saved["gsm8k_0"]["point"] == 0.5
        # 新加的是 1.0
        assert saved["gsm8k_1"]["point"] == 1.0

    def test_overwrite_reruns_all(self, base, mgr):
        self._setup_data(base)
        _write_annotations(base, "gsm8k", "llm", "ModelA", [
            {"key": "gsm8k_0", "point": 0.5}
        ])

        runner = AnnotationRunner(mgr, concurrency=4)
        results = runner.run(_stub_annotator(1.0), "gsm8k", "ModelA", "llm", overwrite=True)

        assert len(results) == 3
        saved = mgr.get_annotations("gsm8k", "ModelA", "llm")
        assert all(r["point"] == 1.0 for r in saved)

    def test_missing_response_raises(self, base, mgr):
        _write_dataset(base, "gsm8k", [{"key": "k0", "question": "Q", "answer": "A"}])
        # 沒有 response 檔案

        runner = AnnotationRunner(mgr, concurrency=4)
        with pytest.raises(FileNotFoundError):
            runner.run(_stub_annotator(), "gsm8k", "ModelA", "llm")

    def test_all_done_returns_empty(self, base, mgr):
        self._setup_data(base, n=2)
        _write_annotations(base, "gsm8k", "llm", "ModelA", [
            {"key": "gsm8k_0", "point": 1.0},
            {"key": "gsm8k_1", "point": 0.5},
        ])

        runner = AnnotationRunner(mgr, concurrency=4)
        results = runner.run(_stub_annotator(), "gsm8k", "ModelA", "llm")
        assert results == []

    def test_dataset_item_passed_to_annotator(self, base, mgr):
        """確認 annotator 接收到完整 dataset item（含非標準欄位）。"""
        _write_dataset(base, "gsm8k", [
            {"key": "gsm8k_0", "question": "Q0", "answer": "A0",
             "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}
        ])
        _write_responses(base, "gsm8k", "ModelA", [
            {"key": "gsm8k_0", "text": "R0"}
        ])

        received_items = []

        from LLMRouter.annotator.base import BaseAnnotator

        class _CapturingAnnotator(BaseAnnotator):
            async def annotate_one(self, key, dataset_item, response, sem):
                received_items.append(dataset_item)
                return {"key": key, "point": 0.0}

        runner = AnnotationRunner(mgr, concurrency=1)
        runner.run(_CapturingAnnotator(), "gsm8k", "ModelA", "llm")

        assert len(received_items) == 1
        item = received_items[0]
        assert item["question"] == "Q0"
        assert item["answer"] == "A0"
        assert item["instruction_id_list"] == ["punctuation:no_comma"]

    def test_none_results_are_skipped(self, base, mgr):
        """annotate_one 回傳 None 時，該筆不寫入。"""
        self._setup_data(base, n=3)

        from LLMRouter.annotator.base import BaseAnnotator

        class _FailFirst(BaseAnnotator):
            def __init__(self):
                self.call_count = 0

            async def annotate_one(self, key, dataset_item, response, sem):
                self.call_count += 1
                if key == "gsm8k_0":
                    return None
                return {"key": key, "point": 0.5}

        annotator = _FailFirst()
        runner = AnnotationRunner(mgr, concurrency=4)
        results = runner.run(annotator, "gsm8k", "ModelA", "llm")

        assert len(results) == 2
        saved = mgr.get_annotations("gsm8k", "ModelA", "llm")
        keys = {r["key"] for r in saved}
        assert "gsm8k_0" not in keys
