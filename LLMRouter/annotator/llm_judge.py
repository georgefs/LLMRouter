from __future__ import annotations

import asyncio
import json
import re
import time
from typing import List, Optional

from .base import BaseAnnotator

# ── Standard GT-based evaluation prompt ───────────────────────────────────────

GT_PROMPT_TEMPLATE = """\
You are a neutral, stable, and reproducible automated evaluation system.

Your task is to compare the "gold answer" and the "response" and assess the correctness of the response.

### Evaluation Principles
1. Judge based on semantic equivalence rather than exact string matching.
2. Evaluate only what is explicitly stated in the response; do not infer or fill in missing information.
3. If the response covers the key facts and conclusions of the gold answer, it should receive a high correctness score.
4. If the response omits key points, contains incorrect information, or is logically inconsistent, the score should be reduced.
5. Differences in wording, language, or expression should not affect the judgment.

### Gold Answer:
{answer}

### Response:
{response}

### Output JSON only (no additional text):
{{
  "verdict": "correct | partially_correct | incorrect",
  "score": 0.0–1.0,
  "missing_key_points": [],
  "incorrect_statements": [],
  "brief_reason": "Evaluation summary in no more than 30 words"
}}"""

# ── IFEval-style format verification prompt ───────────────────────────────────

FORMAT_PROMPT_TEMPLATE = """\
You are a neutral, stable, and reproducible automated evaluation system.

Your task is to evaluate whether the "response" follows ALL of the specified format instructions.

### Format Instructions to Verify:
{instructions}

### Response:
{response}

### Evaluation Principles
1. Check each instruction independently and strictly.
2. Do not infer intent — only evaluate what is explicitly present in the response.
3. If any instruction is not satisfied, it must be listed in failed_instructions.

### Output JSON only (no additional text):
{{
  "verdict": "correct | partially_correct | incorrect",
  "score": 0.0–1.0,
  "failed_instructions": [],
  "brief_reason": "Evaluation summary in no more than 30 words"
}}"""


def _describe_instruction(instruction_id: str, kwargs: dict) -> str:
    """將 instruction_id 與其參數轉換為可讀描述。"""
    p = {k: v for k, v in kwargs.items() if v is not None}

    desc_map = {
        "punctuation:no_comma":
            "The response must not contain any commas.",
        "detectable_format:number_highlighted_sections":
            f"The response must contain at least {p.get('num_highlights', 'N')} highlighted sections"
            f" (text wrapped in *highlighted text*).",
        "detectable_format:number_bullet_lists":
            f"The response must contain at least {p.get('num_bullets', 'N')} bullet list items"
            f" (lines starting with * or -).",
        "detectable_format:multiple_sections":
            f"The response must contain exactly {p.get('num_sections', 'N')} sections"
            f" divided by the section splitter '{p.get('section_spliter', '***')}'.",
        "detectable_format:json_format":
            "The entire response must be valid JSON.",
        "detectable_format:title":
            "The response must contain a title in markdown format (e.g., # Title).",
        "detectable_format:constrained_response":
            "The response must be a single letter (A, B, C, or D), nothing else.",
        "detectable_content:number_placeholders":
            f"The response must contain at least {p.get('num_placeholders', 'N')} placeholder(s)"
            f" in the format [placeholder].",
        "detectable_content:postscript":
            f"The response must end with a postscript starting with '{p.get('postscript_marker', 'P.S.')}'.",
        "length_constraints:number_words":
            f"The response must have {p.get('relation', 'at least')} {p.get('num_words', 'N')} words.",
        "length_constraints:number_sentences":
            f"The response must have {p.get('relation', 'at least')} {p.get('num_sentences', 'N')} sentences.",
        "length_constraints:number_paragraphs":
            f"The response must have {p.get('relation', 'at least')} {p.get('num_paragraphs', 'N')} paragraphs.",
        "length_constraints:nth_paragraph_first_word":
            f"The {p.get('nth_paragraph', 'N')}th paragraph must start with the word '{p.get('first_word', 'WORD')}'.",
        "keywords:existence":
            f"The response must include all of the following keywords: {p.get('keywords', [])}.",
        "keywords:forbidden_words":
            f"The response must NOT include any of the following words: {p.get('forbidden_words', [])}.",
        "keywords:frequency":
            f"The keyword '{p.get('keyword', 'WORD')}' must appear"
            f" {p.get('relation', 'at least')} {p.get('frequency', 'N')} time(s) in the response.",
        "keywords:letter_frequency":
            f"The letter '{p.get('letter', 'X')}' must appear"
            f" {p.get('let_relation', 'at least')} {p.get('let_frequency', 'N')} time(s) in the response.",
        "language:response_language":
            f"The entire response must be written in {p.get('language', 'the specified language')}.",
        "change_case:english_capital":
            "The entire response must be written in ALL CAPS.",
        "change_case:english_lowercase":
            "The entire response must be written in all lowercase letters.",
        "change_case:capital_word_frequency":
            f"The response must contain {p.get('capital_relation', 'at least')}"
            f" {p.get('capital_frequency', 'N')} word(s) written in ALL CAPS.",
        "startend:end_checker":
            f"The response must end with the phrase: '{p.get('end_phrase', 'PHRASE')}'.",
        "startend:quotation":
            "The response must be wrapped in double quotation marks.",
        "combination:repeat_prompt":
            "The response must first repeat the prompt verbatim, then provide the answer.",
        "combination:two_responses":
            "The response must contain two separate answers separated by 6 asterisks (******).",
    }

    return desc_map.get(instruction_id, f"{instruction_id}: params={p}")


def _build_instructions_text(instruction_id_list: List[str], kwargs_list: List[dict]) -> str:
    lines = []
    for i, (instr_id, kw) in enumerate(zip(instruction_id_list, kwargs_list), start=1):
        lines.append(f"{i}. [{instr_id}] {_describe_instruction(instr_id, kw)}")
    return "\n".join(lines)


def _parse_score(text: str) -> float:
    """從 judge 回應中解析出 0~1 的分數。"""
    try:
        obj = json.loads(text.strip())
        val = obj.get("score") or obj.get("point") or obj.get("rating")
        if val is not None:
            return float(max(0.0, min(1.0, float(val))))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    match = re.search(r'"score":\s*(\d+(?:\.\d+)?)', text)
    if match:
        return float(max(0.0, min(1.0, float(match.group(1)))))

    match = re.search(r"\b([01](?:\.\d+)?|0\.\d+)\b", text)
    if match:
        return float(max(0.0, min(1.0, float(match.group(1)))))

    return 0.0


def _parse_meta(text: str) -> dict:
    """嘗試解析 judge 回應的完整 JSON，失敗則回傳空 dict。"""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return {}


class LLMJudgeAnnotator(BaseAnnotator):
    """
    LLM-as-Judge 評分器：呼叫 judge LLM 對 response 評分，回傳 0~1 的分數與補充資訊。

    支援兩種評估模式：
    - 標準 GT-based：提供 gold answer，評估回應的語意正確性
    - IFEval-style：提供 instruction_id_list（dataset item 欄位），評估格式指令的遵守程度

    Args:
        router: litellm Router 實例
        judge: judge LLM 的 model_name（需在 config.yaml 中）
        gt_prompt_template: 標準評估 prompt 模板，需含 {answer}, {response} 佔位符
        format_prompt_template: IFEval 格式評估 prompt 模板，需含 {instructions}, {response} 佔位符
    """

    def __init__(
        self,
        router,
        judge: str,
        gt_prompt_template: str = GT_PROMPT_TEMPLATE,
        format_prompt_template: str = FORMAT_PROMPT_TEMPLATE,
    ) -> None:
        self.router = router
        self.judge = judge
        self.gt_prompt_template = gt_prompt_template
        self.format_prompt_template = format_prompt_template

    async def annotate_one(
        self,
        key: str,
        dataset_item: dict,
        response: str,
        sem: asyncio.Semaphore,
    ) -> Optional[dict]:
        # 根據 dataset item 是否有 instruction_id_list 選擇 prompt
        instruction_id_list = dataset_item.get("instruction_id_list")
        if instruction_id_list:
            kwargs_list = dataset_item.get("kwargs") or [{} for _ in instruction_id_list]
            instructions_text = _build_instructions_text(instruction_id_list, kwargs_list)
            prompt = self.format_prompt_template.format(
                instructions=instructions_text,
                response=response,
            )
        else:
            prompt = self.gt_prompt_template.format(
                answer=dataset_item.get("answer", ""),
                response=response,
            )

        async with sem:
            t0 = time.time()
            try:
                resp = await self.router.acompletion(
                    model=self.judge,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content
                meta = _parse_meta(raw)
                return {
                    "key": key,
                    "point": _parse_score(raw),
                    "verdict": meta.get("verdict", ""),
                    "brief_reason": meta.get("brief_reason", ""),
                    "judge": self.judge,
                    "raw": raw,
                    "judge_time": round(time.time() - t0, 6),
                }
            except Exception as exc:
                print(f"[error] {key}: {exc}")
                return None
