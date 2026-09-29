"""
新 Annotator 實作範本
=====================
複製此檔案，全域搜尋 MyAnnotator / my_strategy 替換為實際名稱。

最少需要實作
------------
  annotate(prompt, response, dataset_item)
    回傳 (score: float, metadata: dict)
    score ∈ [0.0, 1.0]

新增後不需修改任何其他檔案
--------------------------
  register() 呼叫完成後，CLI 即可使用：
    python -m LLMRouter annotation gen <dataset> <model> --strategy my_strategy
"""
from __future__ import annotations

from .base import BaseAnnotator


class MyAnnotator(BaseAnnotator):
    """
    一句話說明評分策略。

    Args:
        threshold: 範例參數，說明用途
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def annotate(
        self,
        prompt: str,
        response: str,
        dataset_item: dict,
    ) -> tuple[float, dict]:
        """
        對單筆 (prompt, response) 計算品質分數。

        Args:
            prompt:       原始問題文字
            response:     模型回答文字
            dataset_item: 原始資料集條目（含 ground truth、metadata 等）

        Returns:
            (score, metadata)
            score    ∈ [0.0, 1.0]，越高越好
            metadata 任意 dict，存入 annotation 檔案供事後分析

        Raises:
            任何例外都會被 AnnotationRunner 攔截並記錄為 None（跳過該筆）。
        """
        # TODO: 實作評分邏輯
        raise NotImplementedError


# ── Registry 登記（放在檔案最末，import 時自動執行）──────────────────────────
#
# factory_fn 簽名：(args: argparse.Namespace, config: dict) -> BaseAnnotator
# config 是 config.yaml 的內容（含 model_list 等）。
# 若 annotator 需要 LLM 呼叫，可從 config["model_list"] 建立 litellm.Router。
# 若需要額外 CLI 參數（如 --judge），在 _factory 中透過 getattr(args, "xxx") 讀取，
# 並在 __main__.py 的 annotation gen 子命令中新增對應的 add_argument()。
#
from .registry import register  # noqa: E402


def _factory(args, config):
    # 範例：threshold = getattr(args, "threshold", 0.5)
    return MyAnnotator()


register("my_strategy", _factory)
