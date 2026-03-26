"""
DatasetManager 單元測試

執行: pytest LLMRouter/test/test_manager.py -v
"""

import json
from pathlib import Path

import pytest

from LLMRouter.manager import DatasetManager
from LLMRouter.scorer import BaseScorer, FieldScorer


# ── fixtures ──────────────────────────────────────────────────────────────────


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture()
def base(tmp_path: Path) -> Path:
    """建立最小可用的 dataset 目錄結構。"""
    ds = tmp_path / "datasets"
    rs = tmp_path / "responses"
    an = tmp_path / "annotations"
    ds.mkdir(); rs.mkdir(); an.mkdir()
    return tmp_path


@pytest.fixture()
def mgr(base: Path) -> DatasetManager:
    return DatasetManager(base_path=str(base))


@pytest.fixture()
def populated(base: Path, mgr: DatasetManager) -> DatasetManager:
    """填入 gsm8k / ModelA / llm 的基本資料。"""
    write_jsonl(base / "datasets" / "gsm8k.jsonl", [
        {"key": "gsm8k_0", "question": "Q0", "answer": "A0"},
        {"key": "gsm8k_1", "question": "Q1", "answer": "A1"},
        {"key": "gsm8k_2", "question": "Q2", "answer": "A2"},
    ])
    write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
        {"key": "gsm8k_0", "text": "R0"},
        {"key": "gsm8k_1", "text": "R1"},
        {"key": "gsm8k_2", "text": "R2"},
    ])
    write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
        {"key": "gsm8k_0", "point": 1.0},
        {"key": "gsm8k_1", "point": 0.5},
        {"key": "gsm8k_2", "point": 0.0},
    ])
    return mgr


# ── list operations ───────────────────────────────────────────────────────────


class TestList:
    def test_list_datasets_empty(self, mgr):
        assert mgr.list_datasets() == []

    def test_list_datasets(self, base, mgr):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        write_jsonl(base / "datasets" / "squad2.jsonl", [])
        assert mgr.list_datasets() == ["gsm8k", "squad2"]

    def test_list_models_missing_dataset(self, mgr):
        assert mgr.list_models("nonexistent") == []

    def test_list_models(self, populated):
        assert populated.list_models("gsm8k") == ["ModelA"]

    def test_list_annotation_strategies_empty(self, mgr):
        assert mgr.list_annotation_strategies("nonexistent") == {}

    def test_list_annotation_strategies(self, populated):
        result = populated.list_annotation_strategies("gsm8k")
        assert result == {"llm": ["ModelA"]}

    def test_list_annotation_strategies_multiple(self, base, mgr):
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [])
        write_jsonl(base / "annotations" / "gsm8k" / "similar" / "ModelA.jsonl", [])
        write_jsonl(base / "annotations" / "gsm8k" / "similar" / "ModelB.jsonl", [])
        result = mgr.list_annotation_strategies("gsm8k")
        assert list(result.keys()) == ["llm", "similar"]
        assert result["similar"] == ["ModelA", "ModelB"]


# ── read operations ───────────────────────────────────────────────────────────


class TestRead:
    def test_get_dataset_not_found(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.get_dataset("nonexistent")

    def test_get_dataset(self, populated):
        rows = populated.get_dataset("gsm8k")
        assert len(rows) == 3
        assert rows[0]["key"] == "gsm8k_0"
        assert rows[0]["question"] == "Q0"

    def test_get_responses_not_found(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.get_responses("gsm8k", "ModelA")

    def test_get_responses(self, populated):
        rows = populated.get_responses("gsm8k", "ModelA")
        assert len(rows) == 3
        assert rows[1]["text"] == "R1"

    def test_get_annotations_not_found(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.get_annotations("gsm8k", "ModelA", "llm")

    def test_get_annotations(self, populated):
        rows = populated.get_annotations("gsm8k", "ModelA", "llm")
        assert len(rows) == 3
        assert rows[0]["point"] == 1.0


# ── add_responses ─────────────────────────────────────────────────────────────


class TestAddResponses:
    def test_fresh_write(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        data = [{"key": "k0", "text": "t0"}, {"key": "k1", "text": "t1"}]
        path, added, skipped = mgr.add_responses("gsm8k", "ModelA", data)
        assert added == 2
        assert skipped == 0
        assert path.exists()
        assert len(mgr.get_responses("gsm8k", "ModelA")) == 2

    def test_merge_skips_existing_keys(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        existing = [{"key": "k0", "text": "old"}]
        mgr.add_responses("gsm8k", "ModelA", existing)

        new_data = [{"key": "k0", "text": "new"}, {"key": "k1", "text": "t1"}]
        _, added, skipped = mgr.add_responses("gsm8k", "ModelA", new_data)

        assert added == 1
        assert skipped == 1
        rows = mgr.get_responses("gsm8k", "ModelA")
        assert len(rows) == 2
        # k0 保留舊資料
        by_key = {r["key"]: r for r in rows}
        assert by_key["k0"]["text"] == "old"
        assert by_key["k1"]["text"] == "t1"

    def test_overwrite_replaces_all(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        mgr.add_responses("gsm8k", "ModelA", [{"key": "k0", "text": "old"}])
        new_data = [{"key": "k0", "text": "new"}, {"key": "k1", "text": "t1"}]
        _, added, skipped = mgr.add_responses("gsm8k", "ModelA", new_data, overwrite=True)
        assert added == 2
        assert skipped == 0
        rows = mgr.get_responses("gsm8k", "ModelA")
        by_key = {r["key"]: r for r in rows}
        assert by_key["k0"]["text"] == "new"

    def test_merge_all_existing_no_write(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        mgr.add_responses("gsm8k", "ModelA", [{"key": "k0", "text": "t0"}])
        _, added, skipped = mgr.add_responses("gsm8k", "ModelA", [{"key": "k0", "text": "dup"}])
        assert added == 0
        assert skipped == 1


# ── add_annotations ───────────────────────────────────────────────────────────


class TestAddAnnotations:
    def test_fresh_write(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        data = [{"key": "k0", "point": 1.0}, {"key": "k1", "point": 0.5}]
        path, added, skipped = mgr.add_annotations("gsm8k", "ModelA", "llm", data)
        assert added == 2
        assert skipped == 0
        assert path.exists()

    def test_merge_skips_existing_keys(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        mgr.add_annotations("gsm8k", "ModelA", "llm", [{"key": "k0", "point": 1.0}])
        _, added, skipped = mgr.add_annotations(
            "gsm8k", "ModelA", "llm",
            [{"key": "k0", "point": 0.0}, {"key": "k1", "point": 0.5}]
        )
        assert added == 1
        assert skipped == 1
        rows = mgr.get_annotations("gsm8k", "ModelA", "llm")
        by_key = {r["key"]: r for r in rows}
        assert by_key["k0"]["point"] == 1.0   # 舊值保留
        assert by_key["k1"]["point"] == 0.5

    def test_overwrite_replaces_all(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        mgr.add_annotations("gsm8k", "ModelA", "llm", [{"key": "k0", "point": 1.0}])
        _, added, skipped = mgr.add_annotations(
            "gsm8k", "ModelA", "llm",
            [{"key": "k0", "point": 0.0}],
            overwrite=True,
        )
        assert added == 1
        assert skipped == 0
        rows = mgr.get_annotations("gsm8k", "ModelA", "llm")
        assert rows[0]["point"] == 0.0

    def test_creates_strategy_subdir(self, mgr, base):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [])
        mgr.add_annotations("gsm8k", "ModelA", "similar", [{"key": "k0", "point": 0.9}])
        assert (base / "annotations" / "gsm8k" / "similar" / "ModelA.jsonl").exists()


# ── extract ───────────────────────────────────────────────────────────────────


class TestExtract:
    def test_basic_extract(self, populated):
        rows = populated.extract(["gsm8k"], ["ModelA"], "llm")
        assert len(rows) == 3
        r = rows[0]
        assert r["dataset"] == "gsm8k"
        assert r["model"] == "ModelA"
        assert r["question"] == "Q0"
        assert r["answer"] == "A0"
        assert r["response"] == "R0"
        assert r["score"] == 1.0
        # annotations 以 strategy 為 key
        assert "annotations" in r
        assert "llm" in r["annotations"]
        assert r["annotations"]["llm"]["point"] == 1.0

    def test_extract_annotations_dict_excludes_key(self, populated):
        rows = populated.extract(["gsm8k"], ["ModelA"], "llm")
        for r in rows:
            assert "key" not in r["annotations"]["llm"]

    def test_extract_annotations_dict_includes_extra_fields(self, base, mgr):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.9, "verdict": "correct", "brief_reason": "ok"},
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], "llm")
        ann = rows[0]["annotations"]["llm"]
        assert ann["point"] == 0.9
        assert ann["verdict"] == "correct"
        assert ann["brief_reason"] == "ok"

    def test_extract_custom_scorer_field(self, base, mgr):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "cost" / "ModelA.jsonl", [
            {"key": "k0", "cost": 0.0012},
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], "cost", scorer=FieldScorer("cost"))
        assert rows[0]["score"] == pytest.approx(0.0012)

    def test_extract_missing_score_field_defaults_zero(self, base, mgr):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.8},  # 沒有 "myscore"
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], "llm", scorer=FieldScorer("myscore"))
        assert rows[0]["score"] == 0.0

    def test_extract_custom_scorer_uses_response(self, base, mgr):
        """自訂 scorer 可存取 response 欄位（e.g. 根據 token 數計算 cost）。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "hello world", "usage": {"total_tokens": 50}},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 1.0},
        ])

        class TokenCostScorer(BaseScorer):
            def __call__(self, dataset_item, response, annotations):
                tokens = response.get("usage", {}).get("total_tokens", 0)
                return float(tokens) * 0.001

        rows = mgr.extract(["gsm8k"], ["ModelA"], "llm", scorer=TokenCostScorer())
        assert rows[0]["score"] == pytest.approx(0.05)

    def test_extract_custom_scorer_uses_dataset_item(self, base, mgr):
        """自訂 scorer 可存取 dataset_item 欄位（e.g. 根據題目難度加權）。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A", "difficulty": 2.0},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.5},
        ])

        class WeightedScorer(BaseScorer):
            def __call__(self, dataset_item, response, annotations):
                point = annotations.get("llm", {}).get("point", 0.0)
                difficulty = dataset_item.get("difficulty", 1.0)
                return float(point * difficulty)

        rows = mgr.extract(["gsm8k"], ["ModelA"], "llm", scorer=WeightedScorer())
        assert rows[0]["score"] == pytest.approx(1.0)  # 0.5 * 2.0

    def test_extract_no_response_field(self, populated):
        rows = populated.extract(["gsm8k"], ["ModelA"], "llm", include_response=False)
        assert all("response" not in r for r in rows)

    def test_extract_missing_model_skipped(self, populated):
        rows = populated.extract(["gsm8k"], ["ModelA", "ModelB"], "llm")
        # ModelB 沒資料，只回傳 ModelA 的
        assert all(r["model"] == "ModelA" for r in rows)

    def test_extract_missing_strategy_skipped(self, populated):
        rows = populated.extract(["gsm8k"], ["ModelA"], "similar")
        assert rows == []

    def test_extract_multiple_models(self, base, mgr):
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        for model in ("ModelA", "ModelB"):
            write_jsonl(base / "responses" / "gsm8k" / f"{model}.jsonl", [
                {"key": "k0", "text": f"R-{model}"},
            ])
            write_jsonl(base / "annotations" / "gsm8k" / "llm" / f"{model}.jsonl", [
                {"key": "k0", "point": 1.0},
            ])
        rows = mgr.extract(["gsm8k"], ["ModelA", "ModelB"], "llm")
        assert len(rows) == 2
        models = {r["model"] for r in rows}
        assert models == {"ModelA", "ModelB"}

    def test_extract_multiple_datasets(self, base, mgr):
        for ds in ("ds1", "ds2"):
            write_jsonl(base / "datasets" / f"{ds}.jsonl", [
                {"key": f"{ds}_0", "question": "Q", "answer": "A"},
            ])
            write_jsonl(base / "responses" / ds / "ModelA.jsonl", [
                {"key": f"{ds}_0", "text": "R"},
            ])
            write_jsonl(base / "annotations" / ds / "llm" / "ModelA.jsonl", [
                {"key": f"{ds}_0", "point": 0.8},
            ])
        rows = mgr.extract(["ds1", "ds2"], ["ModelA"], "llm")
        assert len(rows) == 2
        datasets = {r["dataset"] for r in rows}
        assert datasets == {"ds1", "ds2"}

    def test_extract_partial_annotation_coverage(self, base, mgr):
        """annotation 只有部分 key 時，只回傳有 annotation 的那幾筆。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q0", "answer": "A0"},
            {"key": "k1", "question": "Q1", "answer": "A1"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R0"},
            {"key": "k1", "text": "R1"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 1.0},  # k1 沒有 annotation
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], "llm")
        assert len(rows) == 1
        assert rows[0]["key"] == "k0"

    def test_extract_dataset_not_found(self, mgr):
        with pytest.raises(FileNotFoundError):
            mgr.extract(["nonexistent"], ["ModelA"], "llm")

    def test_extract_multi_strategy_separate_dicts(self, base, mgr):
        """多個 strategy 各自保存在獨立的 dict，以 strategy 名稱為 key。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.9, "verdict": "correct"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "cost" / "ModelA.jsonl", [
            {"key": "k0", "cost": 0.0015, "tokens": 300},
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], ["llm", "cost"])
        assert len(rows) == 1
        ann = rows[0]["annotations"]
        # 各 strategy 獨立
        assert "llm" in ann
        assert "cost" in ann
        assert ann["llm"]["point"] == 0.9
        assert ann["llm"]["verdict"] == "correct"
        assert ann["cost"]["cost"] == pytest.approx(0.0015)
        assert ann["cost"]["tokens"] == 300
        # 不同 strategy 的欄位不互相混雜
        assert "cost" not in ann["llm"]
        assert "point" not in ann["cost"]

    def test_extract_multi_strategy_score_from_each(self, base, mgr):
        """score_field 可指向任一 strategy 的欄位。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.8},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "cost" / "ModelA.jsonl", [
            {"key": "k0", "cost": 0.002},
        ])
        rows_by_point = mgr.extract(["gsm8k"], ["ModelA"], ["llm", "cost"], scorer=FieldScorer("point"))
        assert rows_by_point[0]["score"] == pytest.approx(0.8)

        rows_by_cost = mgr.extract(["gsm8k"], ["ModelA"], ["llm", "cost"], scorer=FieldScorer("cost"))
        assert rows_by_cost[0]["score"] == pytest.approx(0.002)

    def test_extract_multi_strategy_extra_missing_file_ok(self, base, mgr):
        """額外 strategy 沒有對應檔案時，主要 strategy 的資料仍正常回傳。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q", "answer": "A"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 0.7},
        ])
        # cost strategy 檔案不存在，不應崩潰
        rows = mgr.extract(["gsm8k"], ["ModelA"], ["llm", "cost"])
        assert len(rows) == 1
        assert rows[0]["annotations"]["llm"]["point"] == 0.7
        assert "cost" not in rows[0]["annotations"]

    def test_extract_multi_strategy_extra_missing_key_ok(self, base, mgr):
        """額外 strategy 缺少某些 key 時，那些 key 的 annotations 不含該 strategy。"""
        write_jsonl(base / "datasets" / "gsm8k.jsonl", [
            {"key": "k0", "question": "Q0", "answer": "A0"},
            {"key": "k1", "question": "Q1", "answer": "A1"},
        ])
        write_jsonl(base / "responses" / "gsm8k" / "ModelA.jsonl", [
            {"key": "k0", "text": "R0"},
            {"key": "k1", "text": "R1"},
        ])
        write_jsonl(base / "annotations" / "gsm8k" / "llm" / "ModelA.jsonl", [
            {"key": "k0", "point": 1.0},
            {"key": "k1", "point": 0.5},
        ])
        # cost 只有 k0，沒有 k1
        write_jsonl(base / "annotations" / "gsm8k" / "cost" / "ModelA.jsonl", [
            {"key": "k0", "cost": 0.001},
        ])
        rows = mgr.extract(["gsm8k"], ["ModelA"], ["llm", "cost"])
        by_key = {r["key"]: r for r in rows}
        assert len(rows) == 2
        assert "cost" in by_key["k0"]["annotations"]
        assert by_key["k0"]["annotations"]["cost"]["cost"] == pytest.approx(0.001)
        assert "cost" not in by_key["k1"]["annotations"]
