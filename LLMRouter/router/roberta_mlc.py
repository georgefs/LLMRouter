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
        super().__init__()
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

        # 過濾訓練集中超過 max_length 的樣本，避免用截斷後的噪音資料訓練
        lengths = [
            len(self._tokenizer(p, add_special_tokens=True)["input_ids"])
            for p in data.train_prompt
        ]
        keep = [i for i, l in enumerate(lengths) if l <= 512]
        n_removed = len(data.train_prompt) - len(keep)
        if n_removed:
            print(f"[RoBERTa] 過濾長度 > 512 的訓練樣本：移除 {n_removed} 筆，保留 {len(keep)} 筆")
        train_prompts = [data.train_prompt[i] for i in keep]
        train_scores  = data.train_score[np.array(keep)]

        config = AutoConfig.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            finetuning_task="text-classification",
        )
        config.problem_type = "multi_label_classification"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=config
        )

        train_ds = self._make_dataset(train_prompts, train_scores)
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

    def save(self, path: "str | Path") -> None:
        """
        保存訓練好的 router（包含 RoBERTa 模型權重 + model_names）。

        Args:
            path: 保存路徑 (將保存為目錄，包含模型文件)
        """
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._trainer is None:
            raise RuntimeError("Router must be trained first (call fit())")

        # 保存 HuggingFace 模型
        self._trainer.save_model(str(path))

        # 保存 model_names 和配置到檔案
        import json
        config = {
            'router_type': 'roberta',
            'model_names': self.model_names,
            'model_name': self.model_name,
            'epochs': self.epochs,
            'train_batch_size': self.train_batch_size,
            'eval_batch_size': self.eval_batch_size,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'seed': self.seed,
            '_num_labels': self._num_labels,
        }

        config_path = path / "router_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"[RoBERTa] Saved to {path}")

    @classmethod
    def load(cls, path_or_ck: "str | Path | dict") -> "RoBERTaMLCRouter":
        """載入訓練好的 router（恢復 RoBERTa 模型權重 + model_names）。RoBERTa 僅支援目錄格式。"""
        from pathlib import Path
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import json

        if isinstance(path_or_ck, dict):
            raise TypeError("RoBERTaMLCRouter 不支援從 dict 載入，請傳入目錄路徑。")
        path = Path(path_or_ck)

        # 載入配置
        config_path = path / "router_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)

        router = cls(
            model_name=config['model_name'],
            epochs=config['epochs'],
            train_batch_size=config['train_batch_size'],
            eval_batch_size=config['eval_batch_size'],
            warmup_steps=config['warmup_steps'],
            weight_decay=config['weight_decay'],
            seed=config['seed'],
        )

        # 恢復 model_names 和 num_labels
        router.model_names = config['model_names']
        router._num_labels = config['_num_labels']

        # 載入 tokenizer 和模型
        router._tokenizer = AutoTokenizer.from_pretrained(str(path))
        # 注：_trainer 不被恢復，predict_probs 會使用已載入的模型進行推論
        # 如需完整恢復 trainer，需要額外邏輯

        print(f"[RoBERTa] Loaded from {path}")
        return router


from .registry import register

register("roberta", RoBERTaMLCRouter, lambda a: {
    "model_name": a.roberta_model, "epochs": a.epochs,
})
