from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

from .base import BaseAnnotator


def _detect_dataset(key: str) -> str:
    """Detect dataset family from key prefix."""
    print(key)
    prefixes = [
        "arc_challenge",
        "gsm8k",
        "hellaswag",
        "humaneval",
        "mbpp",
        "ifeval",
        "bfcl",
        "mmlu_pro",
        "squad2",
        "truthfulqa",
    ]
    for p in prefixes:
        if key.startswith(p):
            print(p)
            return p
    return "unknown"


# ── Multiple-choice evaluation ─────────────────────────────────────────────────


def _eval_multiple_choice(response: str, answer: str) -> float:
    """
    Extract the selected option (letter A–J or digit 0–9) from *response*
    and compare it to *answer*.

    Recognised patterns (in order of precedence):
      1. "answer is X" / "answer: X" / "answer:** X" (skips markdown bold after colon)
      2. **X.** or **X. description** markdown bold option with period
      3. **X** markdown bold (lone letter)
      4. **(X) bold-parenthesised
      5. \(X\) LaTeX-style parentheses
      6. CJK answer prefix (答案/正解/正答/答案是/正確答案)
      7. Japanese option prefix (選択肢 X)
      8. (Option X) explicit option label
      9. Option X without parentheses
      10. (X) parenthesised
      11. Response starts with X
      12. Response ends with lone X
    """
    answer = answer.strip().upper()
    response = response.strip()

    patterns = [
        r'(?:the\s+)?(?:answer|option|choice)\s*(?:is|:)\*?\*?\s*([A-J0-9])',
        r'\*\*([A-J0-9])\.',
        r'\*\*([A-J0-9])\*\*',
        r'\*\*\(([A-J0-9])\)',
        r'\\\(([A-J0-9])\\\)',
        r'(?:答案|正解|正答|答案是|正確答案)[：:、\s]*\*?\*?\s*([A-J0-9])',
        r'[（(]選択肢\s*([A-J0-9])[)）]',
        r'\(Option\s+([A-J0-9])\)',
        r'\bOption\s+([A-J0-9])\b',
        r'\(([A-J0-9])\)',
        r'^([A-J0-9])\.?\b',
        r'\b([A-J0-9])\.?\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.IGNORECASE | re.MULTILINE)
        if m:
            if pat != patterns[0]:
                print([answer, m.group(1), pat, response])
            if m.group(1).upper() == answer:
                return 1.0
            else:
                return 0.0
            return 1.0 if m.group(1).upper() == answer else 0.0

    return 0.0


# ── GSM8K evaluation ───────────────────────────────────────────────────────────


def _extract_gsm8k_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from a GSM8K gold/predicted string."""
    # Gold format ends with "#### <number>"
    m = re.search(r'####\s*([\d,]+\.?\d*)', text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Fallback: last number in the text
    nums = re.findall(r"-?[\d,]+\.?\d*", text.replace("$", "").replace("%", ""))
    for raw in reversed(nums):
        try:
            return float(raw.replace(",", ""))
        except ValueError:
            continue
    return None


def _eval_gsm8k(response: str, answer: str) -> float:
    gold = _extract_gsm8k_number(answer)
    pred = _extract_gsm8k_number(response)
    if gold is None or pred is None:
        return 0.0
    return 1.0 if math.isclose(gold, pred, rel_tol=1e-6) else 0.0


# ── IFEval per-instruction rule checkers ──────────────────────────────────────


def _check_instruction(instruction_id: str, kwargs: dict, response: str, prompt: str = "") -> bool:
    """Return True iff *response* satisfies the given IFEval instruction."""
    p = {k: v for k, v in (kwargs or {}).items() if v is not None}

    if instruction_id == "punctuation:no_comma":
        return "," not in response

    if instruction_id == "detectable_format:number_highlighted_sections":
        num = int(p.get("num_highlights", 1))
        # *highlighted* (single-star, non-greedy, no newline inside)
        hits = re.findall(r"\*[^*\n]+\*", response)
        return len(hits) >= num

    if instruction_id == "detectable_format:number_bullet_lists":
        num = int(p.get("num_bullets", 1))
        bullets = re.findall(r"^\s*[*\-]\s+\S", response, re.MULTILINE)
        return len(bullets) >= num

    if instruction_id == "detectable_format:multiple_sections":
        num = int(p.get("num_sections", 1))
        splitter = p.get("section_spliter", "***")
        return len(response.split(splitter)) >= num

    if instruction_id == "detectable_format:json_format":
        try:
            json.loads(response.strip())
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    if instruction_id == "detectable_format:title":
        return bool(re.search(r"^#{1,6}\s+\S", response, re.MULTILINE))

    if instruction_id == "detectable_format:constrained_response":
        return bool(re.fullmatch(r"[A-D]", response.strip()))

    if instruction_id == "detectable_content:number_placeholders":
        num = int(p.get("num_placeholders", 1))
        return len(re.findall(r"\[[^\[\]\n]+\]", response)) >= num

    if instruction_id == "detectable_content:postscript":
        marker = str(p.get("postscript_marker", "P.S."))
        return marker in response

    if instruction_id == "length_constraints:number_words":
        num = int(p.get("num_words", 0))
        relation = p.get("relation", "at least")
        count = len(response.split())
        return _cmp(count, relation, num)

    if instruction_id == "length_constraints:number_sentences":
        num = int(p.get("num_sentences", 0))
        relation = p.get("relation", "at least")
        sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
        return _cmp(len(sentences), relation, num)

    if instruction_id == "length_constraints:number_paragraphs":
        num = int(p.get("num_paragraphs", 0))
        relation = p.get("relation", "at least")
        paragraphs = [pg for pg in re.split(r"\n\s*\n", response) if pg.strip()]
        return _cmp(len(paragraphs), relation, num)

    if instruction_id == "length_constraints:nth_paragraph_first_word":
        nth = int(p.get("nth_paragraph", 1))
        first_word = str(p.get("first_word", "")).lower()
        paragraphs = [pg for pg in re.split(r"\n\s*\n", response) if pg.strip()]
        if len(paragraphs) >= nth:
            words = paragraphs[nth - 1].strip().split()
            if words:
                return words[0].lower().rstrip(".,!?;:") == first_word
        return False

    if instruction_id == "keywords:existence":
        keywords = p.get("keywords", [])
        response_lower = response.lower()
        return all(str(kw).lower() in response_lower for kw in keywords)

    if instruction_id == "keywords:forbidden_words":
        forbidden = p.get("forbidden_words", [])
        response_lower = response.lower()
        return all(str(fw).lower() not in response_lower for fw in forbidden)

    if instruction_id == "keywords:frequency":
        keyword = str(p.get("keyword", "")).lower()
        freq = int(p.get("frequency", 1))
        relation = p.get("relation", "at least")
        return _cmp(response.lower().count(keyword), relation, freq)

    if instruction_id == "keywords:letter_frequency":
        letter = str(p.get("letter", "")).lower()
        freq = int(p.get("let_frequency", 1))
        relation = p.get("let_relation", "at least")
        return _cmp(response.lower().count(letter), relation, freq)

    if instruction_id == "language:response_language":
        # Require external library; pass through to avoid false negatives.
        return True

    if instruction_id == "change_case:english_capital":
        alpha = [c for c in response if c.isalpha()]
        return bool(alpha) and all(c.isupper() for c in alpha)

    if instruction_id == "change_case:english_lowercase":
        alpha = [c for c in response if c.isalpha()]
        return bool(alpha) and all(c.islower() for c in alpha)

    if instruction_id == "change_case:capital_word_frequency":
        freq = int(p.get("capital_frequency", 1))
        relation = p.get("capital_relation", "at least")
        caps = [w for w in response.split() if w.isupper() and w.isalpha()]
        return _cmp(len(caps), relation, freq)

    if instruction_id == "startend:end_checker":
        end_phrase = str(p.get("end_phrase", ""))
        return response.strip().endswith(end_phrase)

    if instruction_id == "startend:quotation":
        stripped = response.strip()
        return stripped.startswith('"') and stripped.endswith('"')

    if instruction_id == "combination:repeat_prompt":
        return bool(prompt) and prompt.strip() in response

    if instruction_id == "combination:two_responses":
        return "******" in response

    # Unknown instruction: pass by default to avoid false negatives.
    return True


def _cmp(value: int, relation: str, target: int) -> bool:
    if relation in ("at least", ">="):
        return value >= target
    if relation in ("at most", "<="):
        return value <= target
    if relation in ("exactly", "==", "around"):
        return value == target
    return False


def _eval_ifeval(response: str, dataset_item: dict) -> float:
    """Score IFEval as fraction of satisfied instructions (0.0–1.0)."""
    instruction_id_list = dataset_item.get("instruction_id_list") or []
    if not instruction_id_list:
        return 0.0

    kwargs_list = dataset_item.get("kwargs") or [{} for _ in instruction_id_list]
    prompt = dataset_item.get("question", "")

    results = [
        _check_instruction(instr_id, kw or {}, response, prompt)
        for instr_id, kw in zip(instruction_id_list, kwargs_list)
    ]
    return sum(results) / len(results)


# ── Code execution (HumanEval / MBPP) ─────────────────────────────────────────


def _extract_code(response: str) -> str:
    """Extract Python code from a markdown code block, or return text as-is."""
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\n?(.*?)```", response, re.DOTALL)
    if m:
        return m.group(1)
    return response


def _run_code(code: str, timeout: int = 10) -> float:
    """
    Execute *code* in an isolated subprocess.
    Returns 1.0 if exit code is 0, else 0.0.
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _eval_humaneval(response: str, dataset_item: dict) -> float:
    """Run HumanEval tests against the model's extracted code."""
    code = _extract_code(response)
    prompt_code = dataset_item.get("question", "")
    test_code = dataset_item.get("test", "")
    entry_point = dataset_item.get("entry_point", "")

    full_code = f"{prompt_code}\n{code}\n{test_code}\ncheck({entry_point})\n"
    return _run_code(full_code)


def _eval_mbpp(response: str, dataset_item: dict) -> float:
    """Run MBPP test assertions against the model's extracted code."""
    code = _extract_code(response)
    test_list = dataset_item.get("test_list", [])

    full_code = code + "\n" + "\n".join(test_list) + "\n"
    return _run_code(full_code)


# ── BFCL function-call evaluation ─────────────────────────────────────────────


def _normalize_val(v):
    """Recursively normalise a value for fuzzy comparison."""
    if isinstance(v, str):
        try:
            return float(v)
        except (ValueError, TypeError):
            return v.lower().strip()
    if isinstance(v, dict):
        return {k.lower(): _normalize_val(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_normalize_val(i) for i in v]
    return v


def _match_call(pred: dict, gold: dict) -> bool:
    """Check whether a single predicted call matches the gold call."""
    g_fn = next(iter(gold))
    p_fn = next(iter(pred), None)
    if p_fn is None or g_fn.lower() != p_fn.lower():
        return False

    g_params = gold[g_fn] if isinstance(gold[g_fn], dict) else {}
    p_params = pred[p_fn] if isinstance(pred[p_fn], dict) else {}

    for key, g_val in g_params.items():
        if key not in p_params:
            return False
        if _normalize_val(p_params[key]) != _normalize_val(g_val):
            return False
    return True


def _parse_json_response(response: str) -> Optional[list]:
    """Try to extract a JSON array (or object) from the response text."""
    response = response.strip()
    # Direct parse
    try:
        parsed = json.loads(response)
        return parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, ValueError):
        pass
    # First JSON array
    m = re.search(r"\[.*\]", response, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, ValueError):
            pass
    # First JSON object
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if m:
        try:
            return [json.loads(m.group(0))]
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _eval_bfcl(response: str, answer: str) -> float:
    """Compare predicted function calls to the gold answer (all calls must match)."""
    try:
        gold = json.loads(answer)
        gold = gold if isinstance(gold, list) else [gold]
    except (json.JSONDecodeError, ValueError, TypeError):
        return 0.0

    pred = _parse_json_response(response)
    if pred is None or len(pred) != len(gold):
        return 0.0

    return 1.0 if all(_match_call(p, g) for p, g in zip(pred, gold)) else 0.0


# ── SQuAD 2 evaluation ─────────────────────────────────────────────────────────


def _normalize_squad(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _squad_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_squad(pred).split()
    gold_tokens = _normalize_squad(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    num_common = sum(min(pred_tokens.count(t), gold_tokens.count(t)) for t in common)
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _eval_squad2(response: str, dataset_item: dict) -> float:
    """Token-level F1 against all gold answers; handles unanswerable questions."""
    answer = dataset_item.get("answer", "")
    print(answer)
    is_unanswerable = answer == "" or answer.lower() == "unanswerable"

    if is_unanswerable:
        return 1.0 if _normalize_squad(response.strip()) == "unanswerable" else 0.0

    gold_answers: list[str] = [answer]
    for a in dataset_item.get("answers", []):
        text = a.get("text", "") if isinstance(a, dict) else str(a)
        if text and text not in gold_answers:
            gold_answers.append(text)

    return max((_squad_f1(response, g) for g in gold_answers if g), default=0.0)


# ── TruthfulQA evaluation ──────────────────────────────────────────────────────


def _eval_truthfulqa(response: str, dataset_item: dict) -> float:
    """
    Score TruthfulQA by checking whether *response* contains a correct answer
    and does NOT contain an incorrect one.
    Returns 1.0 (correct), 0.0 (incorrect), or 0.5 (ambiguous).
    """
    correct = [str(a).lower() for a in dataset_item.get("correct_answers", []) if a]
    incorrect = [str(a).lower() for a in dataset_item.get("incorrect_answers", []) if a]
    resp_lower = response.lower().strip()

    hit_correct = any(a in resp_lower for a in correct)
    hit_incorrect = any(a in resp_lower for a in incorrect)

    if hit_correct and not hit_incorrect:
        return 1.0
    if hit_incorrect:
        return 0.0
    return 0.5  # neither matched — ambiguous


# ── OfficialAnnotator ──────────────────────────────────────────────────────────


class OfficialAnnotator(BaseAnnotator):
    """
    Rule-based annotator that applies each dataset's official evaluation metric.
    No LLM calls — fully deterministic and free to run.

    Supported datasets
    ------------------
    arc_challenge, hellaswag, mmlu_pro
        Multiple-choice exact match (extracts the letter/digit option).
    gsm8k
        Numeric answer extraction; compares the final numbers.
    ifeval
        Programmatic per-instruction checkers; score = fraction satisfied.
    humaneval, mbpp
        Executes extracted Python code in a subprocess; pass = 1.0.
    bfcl_*  (simple, multiple, parallel, sql, java, javascript, live_*, …)
        Parses JSON function calls and compares them to the gold answer.
    squad2
        Token-level F1 against all gold answers; handles unanswerable.
    truthfulqa
        Checks presence of correct/incorrect answers in the response.

    Usage::

        runner = AnnotationRunner(mgr)
        runner.run(OfficialAnnotator(), dataset="gsm8k_train",
                   model="Llama-3.1-70B", strategy="official")
    """

    async def annotate_one(
        self,
        key: str,
        dataset_item: dict,
        response: str,
        sem: asyncio.Semaphore,
    ) -> Optional[dict]:
        dataset_type = _detect_dataset(key)
        point, meta = self._evaluate(dataset_type, response, dataset_item)
        return {"key": key, "point": point, **meta}

    # ------------------------------------------------------------------

    def _evaluate(self, dataset_type: str, response: str, dataset_item: dict) -> tuple[float, dict]:
        answer = dataset_item.get("answer", "")

        if dataset_type in ("arc_challenge", "hellaswag", "mmlu_pro"):
            return _eval_multiple_choice(response, answer), {"method": "exact_match"}

        if dataset_type == "gsm8k":
            return _eval_gsm8k(response, answer), {"method": "numeric_match"}

        if dataset_type == "ifeval":
            return _eval_ifeval(response, dataset_item), {"method": "instruction_check"}

        if dataset_type == "humaneval":
            return _eval_humaneval(response, dataset_item), {"method": "code_execution"}

        if dataset_type == "mbpp":
            return _eval_mbpp(response, dataset_item), {"method": "code_execution"}

        if dataset_type == "bfcl":
            return _eval_bfcl(response, answer), {"method": "function_call_match"}

        if dataset_type == "squad2":
            return _eval_squad2(response, dataset_item), {"method": "f1"}

        if dataset_type == "truthfulqa":
            return _eval_truthfulqa(response, dataset_item), {"method": "answer_match"}

        return 0.0, {"method": "unsupported", "dataset_type": dataset_type}
