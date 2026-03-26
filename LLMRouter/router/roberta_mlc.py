from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData


class RoBERTaMLCRouter(BaseRouter):
    """
    RoBERTa Multi-Label Classification Router（改編自 RouterEval RoBERTa-MLC）。

    使用 HuggingFace Trainer 微調 RoBERTa，以多標籤回歸方式預測各模型分數。

    Args:
        model_name: HuggingFace 模型名稱（預設 "roberta-base"）
        epochs: 訓練 epochs（預設 10）
        train_batch_size: 訓練 batch size（預設 10）
        eval_batch_size: 評估 batch size（預設 12）
        warmup_steps: Warmup steps（預設 500）
        weight_decay: Weight decay（預設 0.01）
        output_dir: checkpoint 儲存路徑（預設使用系統暫存目錄）
        seed: 隨機種子（預設 43）
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        epochs: int = 10,
        train_batch_size: int = 10,
        eval_batch_size: int = 12,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        output_dir: Optional[str] = None,
        seed: int = 43,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.seed = seed
        self._trainer = None
        self._tokenizer = None
        self._num_labels: int = 0

    def _make_dataset(self, prompts: List[str], scores: np.ndarray):
        from datasets import Dataset

        return Dataset.from_dict(
            {"labels": scores.tolist(), "sentence": prompts}
        ).map(self._preprocess, batched=True, desc="Tokenizing")

    def _preprocess(self, examples):
        return self._tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )

    def _fit(self, data: RouterData) -> None:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        num_labels = data.train_score.shape[1]
        self._num_labels = num_labels

        output_dir = self.output_dir or str(
            Path(tempfile.gettempdir()) / "roberta_mlc_checkpoint"
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            finetuning_task="text-classification",
        )
        config.problem_type = "multi_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )

        train_ds = self._make_dataset(data.train_prompt, data.train_score)
        eval_ds = self._make_dataset(data.test_prompt, data.test_score)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_strategy="no",
        )

        self._trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self._tokenizer,
            data_collator=default_data_collator,
        )

        print(f"[RoBERTa] 開始訓練（{self.epochs} epochs, {num_labels} labels）...")
        self._trainer.train()
        print("[RoBERTa] 訓練完成。")

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        if self._trainer is None:
            raise RuntimeError("請先呼叫 fit()")

        dummy_scores = np.zeros((len(prompts), self._num_labels), dtype=np.float32)
        ds = self._make_dataset(prompts, dummy_scores)
        raw = self._trainer.predict(ds).predictions
        probs = np.where(raw > 0, raw, 0).astype(np.float32)
        return probs
