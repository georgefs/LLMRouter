"""
SFT + GRPO LLM-based Router (Qwen2.5 + LoRA via unsloth / trl).

Training pipeline (mirrors RouterEval/router/SFT-Router & GRPO-Router):
  Phase 1 SFT  — imitate cheapest-correct-model oracle decisions
  Phase 2 GRPO — RL fine-tuning with cost-aware reward (§4.3)

Inference: generates "[[model_idx]]" text; predict_probs() returns a
one-hot (N, M) array.

Heavy deps:  unsloth, trl, datasets
Install:     pip install -e ".[rl]"
"""
from __future__ import annotations

import gc
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import BaseRouter
from .data import RouterData


# ─────────────────────────────────────────────────────────────────────────────
# Validation callback
# ─────────────────────────────────────────────────────────────────────────────

class _ValSaveCallback:
    """
    Evaluate on val set every `eval_steps` GRPO steps;
    save best model checkpoint.  Imported lazily to avoid requiring
    transformers at import time.
    """

    def __new__(cls, *args, **kwargs):
        from transformers import TrainerCallback

        class _Impl(TrainerCallback):
            def __init__(self, val_prompts, val_scores, tokenizer, system_prompt,
                         num_models, penalty_vec, alpha, save_dir,
                         eval_steps, eval_samples):
                self._step = 0
                self.val_prompts = list(val_prompts) if val_prompts else []
                self.val_scores  = list(val_scores)  if val_scores  is not None else []
                self.tokenizer    = tokenizer
                self.system_prompt = system_prompt
                self.num_models   = num_models
                self.penalty_vec  = penalty_vec
                self.alpha        = alpha
                self.save_dir     = save_dir
                self.eval_steps   = eval_steps
                self.eval_samples = min(eval_samples, len(self.val_prompts)) if self.val_prompts else 0
                self.best_reward  = -float("inf")

            def on_step_begin(self, args, state, control, **kwargs):
                self._step = state.global_step

            def on_step_end(self, args, state, control, **kwargs):
                if not self.val_prompts or self.eval_samples == 0:
                    return
                if state.global_step <= 0 or state.global_step % self.eval_steps != 0:
                    return

                import torch
                model = kwargs["model"]
                indices = np.random.choice(len(self.val_prompts), self.eval_samples, replace=False)
                total, valid = 0.0, 0

                model.eval()
                with torch.no_grad():
                    for i in indices:
                        q = self.val_prompts[i].strip()
                        messages = [
                            {"role": "system",  "content": self.system_prompt},
                            {"role": "user",    "content": f"Question: {q}"},
                        ]
                        inp = self.tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True, return_tensors="pt"
                        ).to("cuda")
                        out = model.generate(
                            inp, max_new_tokens=128, temperature=0.1, top_p=0.95,
                            pad_token_id=self.tokenizer.eos_token_id,
                        )
                        decoded = self.tokenizer.decode(
                            out[0][inp.shape[1]:], skip_special_tokens=True
                        )
                        decoded = re.sub(r"<think>.*?</think>", "", decoded,
                                         flags=re.DOTALL).strip()
                        matches = re.findall(r"\[\[(\d+)\]\]", decoded)
                        if matches:
                            idx = int(matches[-1])
                            if 0 <= idx < self.num_models:
                                s = float(self.val_scores[i][idx])
                                p = float(self.penalty_vec[idx])
                                if s > 0:
                                    total += max(s * (1.0 - self.alpha * p), 0.0)
                                valid += 1
                model.train()

                avg  = total / self.eval_samples
                rate = valid / self.eval_samples
                print(f"[Val@{state.global_step}] reward={avg:.4f}  format={rate:.1%}")

                if avg > self.best_reward:
                    self.best_reward = avg
                    best = Path(self.save_dir) / "best_model"
                    best.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(best))
                    self.tokenizer.save_pretrained(str(best))
                    print(f"[Val] New best → {best}")

        return _Impl(*args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Router class
# ─────────────────────────────────────────────────────────────────────────────

class SFTGRPORouter(BaseRouter):
    """
    LLM-based router using SFT + GRPO (Qwen2.5-3B + LoRA via unsloth).

    Args:
        base_model:              Hugging Face / unsloth model name or path
        max_seq_length:          Maximum token length for model context
        lora_rank:               LoRA rank (r)
        sft_epochs:              SFT training epochs (0 = skip SFT)
        sft_lr:                  SFT learning rate
        sft_grad_acc:            SFT gradient accumulation steps
        grpo_steps:              GRPO training steps (0 = skip GRPO)
        grpo_lr:                 GRPO learning rate
        grpo_alpha:              Cost-penalty weight in GRPO reward (§4.3)
        grpo_num_generations:    Candidate completions per prompt per GRPO step
        grpo_grad_acc:           GRPO gradient accumulation steps
        grpo_beta:               KL-divergence weight (β)
        grpo_temperature:        Sampling temperature during GRPO training
        grpo_eval_steps:         Validate and checkpoint every N steps
        grpo_eval_samples:       Number of val samples per evaluation
        inference_temperature:   Temperature for greedy inference
        inference_max_new_tokens: Token budget for generation
        seed:                    Random seed
    """

    def __init__(
        self,
        base_model: str = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        max_seq_length: int = 9000,
        lora_rank: int = 16,
        # SFT
        sft_epochs: int = 1,
        sft_lr: float = 2e-4,
        sft_grad_acc: int = 8,
        # GRPO
        grpo_steps: int = 1000,
        grpo_lr: float = 5e-7,
        grpo_alpha: float = 0.2,
        grpo_num_generations: int = 16,
        grpo_grad_acc: int = 16,
        grpo_beta: float = 0.2,
        grpo_temperature: float = 0.9,
        grpo_eval_steps: int = 50,
        grpo_eval_samples: int = 200,
        # Inference
        inference_temperature: float = 0.1,
        inference_max_new_tokens: int = 512,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.base_model              = base_model
        self.max_seq_length          = max_seq_length
        self.lora_rank               = lora_rank
        self.sft_epochs              = sft_epochs
        self.sft_lr                  = sft_lr
        self.sft_grad_acc            = sft_grad_acc
        self.grpo_steps              = grpo_steps
        self.grpo_lr                 = grpo_lr
        self.grpo_alpha              = grpo_alpha
        self.grpo_num_generations    = grpo_num_generations
        self.grpo_grad_acc           = grpo_grad_acc
        self.grpo_beta               = grpo_beta
        self.grpo_temperature        = grpo_temperature
        self.grpo_eval_steps         = grpo_eval_steps
        self.grpo_eval_samples       = grpo_eval_samples
        self.inference_temperature   = inference_temperature
        self.inference_max_new_tokens = inference_max_new_tokens
        self.seed                    = seed

        # Runtime state (populated after fit / load)
        self._model        = None
        self._tokenizer    = None
        self._adapter_dir: Optional[str] = None
        self._system_prompt: Optional[str] = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_costs(self) -> np.ndarray:
        """Unit costs ($/1M tokens) from MODEL_PRICING; zeros if unknown."""
        from .eval import model_unit_costs
        return model_unit_costs(self.model_names)

    def _cost_penalty_vector(self, costs: np.ndarray) -> np.ndarray:
        mx = costs.max()
        return (costs / mx).astype(float) if mx > 0 else np.zeros_like(costs, dtype=float)

    def _build_system_prompt(self, costs: np.ndarray) -> str:
        menu = "Available Models:\n"
        for i, name in enumerate(self.model_names):
            menu += f"Index {i}: {name}\n"
        return (
            "You are an intelligent router aimed at optimizing cost-efficiency. "
            "Your goal is to route the user's input to the cheapest model that can answer it correctly.\n\n"
            f"{menu}\n"
            "Instructions:\n"
            "1. **Analyze**: First, analyze the complexity and specific requirements of the user's question.\n"
            "2. **Evaluate**: Compare the capabilities of the available models against these requirements.\n"
            "3. **Decide**: Select the most cost-efficient model capable of handling the task.\n"
            "4. **Format**: End your response strictly with the format: '[[Index]]'.\n\n"
            "Example Response:\n"
            "The user asks for a simple factual lookup. "
            "Model x is the cheapest and capable enough. Model y is overkill.\n"
            "Final Decision: [[x]]"
        )

    # ── Model loading ─────────────────────────────────────────────────────────

    def _require_unsloth(self):
        try:
            from unsloth import FastLanguageModel  # noqa: F401
        except ImportError:
            raise ImportError(
                "unsloth is required for SFTGRPORouter.\n"
                "Install: pip install -e '.[rl]'"
            )

    def _load_llm(self, model_path: str, gpu_memory_utilization: float = 0.5):
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=self.max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=self.lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return model, tokenizer

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._adapter_dir is None:
            raise RuntimeError("Router not trained. Call fit() or load() first.")
        self._require_unsloth()
        from unsloth import FastLanguageModel
        model, tokenizer = self._load_llm(self._adapter_dir)
        FastLanguageModel.for_inference(model)
        self._model     = model
        self._tokenizer = tokenizer

    # ── Training ──────────────────────────────────────────────────────────────

    def _fit(self, data: RouterData) -> None:
        import torch
        self._require_unsloth()

        if not self.sft_epochs and not self.grpo_steps:
            raise ValueError("sft_epochs and grpo_steps are both 0; nothing to train.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        costs       = self._get_costs()
        penalty_vec = self._cost_penalty_vector(costs)
        self._system_prompt = self._build_system_prompt(costs)

        # Prefer raw (un-penalized) scores for SFT oracle labels
        train_scores = (
            data.train_score if not hasattr(data, "raw_train_score")
            else getattr(data, "raw_train_score", data.train_score)
        )

        # ── Phase 1: SFT ──────────────────────────────────────────────────────
        sft_out = None
        model, tokenizer = self._load_llm(self.base_model, gpu_memory_utilization=0.4)

        if self.sft_epochs > 0:
            sft_out = tempfile.mkdtemp(prefix="llmrouter_sft_")
            print(f"[SFT-GRPO] Phase 1 SFT — {self.sft_epochs} epoch(s)")
            model, tokenizer = self._run_sft(
                model, tokenizer,
                data.train_prompt, train_scores, costs,
                sft_out,
            )

        # ── Phase 2: GRPO ─────────────────────────────────────────────────────
        best_adapter = None

        if self.grpo_steps > 0:
            grpo_out = tempfile.mkdtemp(prefix="llmrouter_grpo_")
            print(f"[SFT-GRPO] Phase 2 GRPO — {self.grpo_steps} step(s), α={self.grpo_alpha}")
            model, tokenizer, best_adapter = self._run_grpo(
                model, tokenizer,
                data, train_scores, penalty_vec,
                grpo_out,
            )
        elif sft_out is not None:
            best_adapter = str(Path(sft_out) / "final")

        # Keep in memory for immediate predict_probs()
        from unsloth import FastLanguageModel
        FastLanguageModel.for_inference(model)
        self._model        = model
        self._tokenizer    = tokenizer
        self._adapter_dir  = best_adapter

    # ── SFT phase ─────────────────────────────────────────────────────────────

    def _run_sft(self, model, tokenizer, prompts, scores, costs, output_dir: str):
        import torch
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import SFTTrainer
        from transformers import TrainingArguments

        # Oracle label: cheapest model that answered correctly
        dataset_list = []
        for i, q in enumerate(prompts):
            correct = np.where(scores[i] > 0)[0]
            best_idx = (
                int(correct[np.argmin(costs[correct])]) if len(correct) > 0
                else int(np.argmin(costs))
            )
            dataset_list.append({"messages": [
                {"role": "system",    "content": self._system_prompt},
                {"role": "user",      "content": f"Question: {q.strip()}"},
                {"role": "assistant", "content": f"Final Decision: [[{best_idx}]]"},
            ]})

        ds = Dataset.from_list(dataset_list)
        ds = ds.map(
            lambda ex: {"text": [
                tokenizer.apply_chat_template(m, tokenize=False)
                for m in ex["messages"]
            ]},
            batched=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha=self.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,
            args=TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=self.sft_grad_acc,
                warmup_steps=10,
                num_train_epochs=self.sft_epochs,
                learning_rate=self.sft_lr,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                seed=self.seed,
                save_strategy="no",
            ),
        )
        trainer.train()

        final = str(Path(output_dir) / "final")
        model.save_pretrained(final)
        tokenizer.save_pretrained(final)
        print(f"[SFT-GRPO] SFT done → {final}")

        del trainer
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        return model, tokenizer

    # ── GRPO phase ────────────────────────────────────────────────────────────

    def _build_reward_fn(
        self,
        prompt_to_scores: Dict[str, np.ndarray],
        penalty_vec: np.ndarray,
    ):
        """Return a reward-function closure for GRPOTrainer."""
        alpha      = self.grpo_alpha
        num_models = len(self.model_names)

        def reward_fn(prompts, completions, **kwargs):
            rewards = []
            for prompt, completion in zip(prompts, completions):
                # Extract question text
                if isinstance(prompt, list):
                    txt = prompt[-1]["content"]
                    m   = re.search(r"Question:\s*(.*)", txt, re.DOTALL)
                    q   = m.group(1).strip() if m else txt.strip()
                else:
                    q = str(prompt).strip()

                row = prompt_to_scores.get(q)
                if row is None:
                    rewards.append(0.0)
                    continue

                content = (
                    completion[0]["content"]
                    if isinstance(completion, list) else str(completion)
                )
                matches = re.findall(r"\[\[(\d+)\]\]", content)
                if not matches:
                    rewards.append(0.0)
                    continue

                idx = int(matches[-1])
                if not (0 <= idx < num_models):
                    rewards.append(0.0)
                    continue

                raw   = float(row[idx])
                pen   = float(penalty_vec[idx])
                FMT   = 0.05
                if raw > 0:
                    n_ok    = max(float(np.sum(np.asarray(row) > 0)), 1.0)
                    task    = max(1.0 + 1.0 / n_ok - alpha * pen, 0.0)
                    rewards.append(FMT + task)
                else:
                    rewards.append(FMT)
            return rewards

        return reward_fn

    def _run_grpo(self, model, tokenizer, data: RouterData,
                  train_scores, penalty_vec, output_dir: str):
        import torch
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel, PatchFastRL
        PatchFastRL("GRPO", FastLanguageModel)

        num_models = len(self.model_names)

        # Freeze early layers if SFT was already applied
        if self.sft_epochs > 0:
            n_layers     = model.config.num_hidden_layers
            train_layers = set(range(max(0, n_layers - 3), n_layers))
            for name, param in model.named_parameters():
                if "lora" in name:
                    m = re.search(r"layers\.(\d+)\.", name)
                    param.requires_grad = bool(m) and int(m.group(1)) in train_layers
                else:
                    param.requires_grad = False
        else:
            # Pure GRPO from scratch — apply LoRA first
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.lora_rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.lora_rank,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=self.seed,
            )

        # Dataset & reward
        prompt_to_scores = {
            q.strip(): train_scores[i]
            for i, q in enumerate(data.train_prompt)
        }
        reward_fn  = self._build_reward_fn(prompt_to_scores, penalty_vec)
        train_ds   = Dataset.from_list([
            {"prompt": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user",   "content": f"Question: {q.strip()}"},
            ]}
            for q in data.train_prompt
        ])

        val_prompts = list(data.val_prompt) if len(data.val_prompt) > 0 else []
        val_scores  = list(data.val_score)  if len(data.val_prompt) > 0 else None

        callback = _ValSaveCallback(
            val_prompts=val_prompts,
            val_scores=val_scores,
            tokenizer=tokenizer,
            system_prompt=self._system_prompt,
            num_models=num_models,
            penalty_vec=penalty_vec,
            alpha=self.grpo_alpha,
            save_dir=output_dir,
            eval_steps=self.grpo_eval_steps,
            eval_samples=self.grpo_eval_samples,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_fn],
            args=GRPOConfig(
                output_dir=output_dir,
                learning_rate=self.grpo_lr,
                weight_decay=0.01,
                warmup_ratio=0.05,
                lr_scheduler_type="cosine",
                logging_steps=10,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=self.grpo_num_generations,
                gradient_accumulation_steps=self.grpo_grad_acc,
                num_generations=self.grpo_num_generations,
                max_prompt_length=self.max_seq_length,
                max_completion_length=512,
                max_steps=self.grpo_steps,
                use_vllm=False,
                report_to="none",
                beta=self.grpo_beta,
                temperature=self.grpo_temperature,
                top_p=0.95,
                seed=self.seed,
            ),
            train_dataset=train_ds,
            callbacks=[callback],
        )
        trainer.train()

        best_dir = str(Path(output_dir) / "best_model")
        if not Path(best_dir).exists():
            # No val improvement recorded — save final state
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)

        # Free trainer memory, reload best checkpoint
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

        model, tokenizer = self._load_llm(best_dir, gpu_memory_utilization=0.5)
        print(f"[SFT-GRPO] GRPO done → {best_dir}")
        return model, tokenizer, best_dir

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        self._ensure_loaded()
        m   = len(self.model_names)
        out = np.zeros((len(prompts), m), dtype=np.float32)
        for i, prompt in enumerate(prompts):
            idx = self._generate_index(prompt.strip())
            out[i, idx] = 1.0
        return out

    def _generate_index(self, q: str) -> int:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user",   "content": f"Question: {q}"},
        ]
        device = next(self._model.parameters()).device
        inputs = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        outputs = self._model.generate(
            inputs,
            max_new_tokens=self.inference_max_new_tokens,
            temperature=self.inference_temperature,
            top_p=0.95,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        decoded = self._tokenizer.decode(
            outputs[0][inputs.shape[1]:], skip_special_tokens=True
        )
        decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.DOTALL).strip()
        matches = re.findall(r"\[\[(\d+)\]\]", decoded)
        if matches:
            return max(0, min(int(matches[-1]), len(self.model_names) - 1))
        return 0

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: "str | Path") -> None:
        import pickle
        import shutil

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._adapter_dir is None:
            raise RuntimeError("Router must be trained first.")

        # Copy adapter next to the pkl
        adapter_dst = path.parent / (path.stem + "_adapter")
        if adapter_dst.resolve() != Path(self._adapter_dir).resolve():
            if adapter_dst.exists():
                shutil.rmtree(adapter_dst)
            shutil.copytree(self._adapter_dir, adapter_dst)

        ck = {
            "router_type":             "sft_grpo",
            "model_names":             self.model_names,
            "system_prompt":           self._system_prompt,
            "adapter_dir":             str(adapter_dst),
            "base_model":              self.base_model,
            "max_seq_length":          self.max_seq_length,
            "lora_rank":               self.lora_rank,
            "sft_epochs":              self.sft_epochs,
            "sft_lr":                  self.sft_lr,
            "sft_grad_acc":            self.sft_grad_acc,
            "grpo_steps":              self.grpo_steps,
            "grpo_lr":                 self.grpo_lr,
            "grpo_alpha":              self.grpo_alpha,
            "grpo_num_generations":    self.grpo_num_generations,
            "grpo_grad_acc":           self.grpo_grad_acc,
            "grpo_beta":               self.grpo_beta,
            "grpo_temperature":        self.grpo_temperature,
            "grpo_eval_steps":         self.grpo_eval_steps,
            "grpo_eval_samples":       self.grpo_eval_samples,
            "inference_temperature":   self.inference_temperature,
            "inference_max_new_tokens": self.inference_max_new_tokens,
            "seed":                    self.seed,
        }
        with open(path, "wb") as f:
            pickle.dump(ck, f)
        print(f"[SFT-GRPO] Saved to {path}  (adapter → {adapter_dst})")

    @classmethod
    def load(cls, path_or_ck: "str | Path | dict") -> "SFTGRPORouter":
        import pickle

        if isinstance(path_or_ck, dict):
            ck = path_or_ck
        else:
            with open(Path(path_or_ck), "rb") as f:
                ck = pickle.load(f)
            print(f"[SFT-GRPO] Loaded metadata from {path_or_ck}")

        router = cls(
            base_model=ck["base_model"],
            max_seq_length=ck["max_seq_length"],
            lora_rank=ck["lora_rank"],
            sft_epochs=ck["sft_epochs"],
            sft_lr=ck["sft_lr"],
            sft_grad_acc=ck["sft_grad_acc"],
            grpo_steps=ck["grpo_steps"],
            grpo_lr=ck["grpo_lr"],
            grpo_alpha=ck["grpo_alpha"],
            grpo_num_generations=ck["grpo_num_generations"],
            grpo_grad_acc=ck["grpo_grad_acc"],
            grpo_beta=ck["grpo_beta"],
            grpo_temperature=ck["grpo_temperature"],
            grpo_eval_steps=ck["grpo_eval_steps"],
            grpo_eval_samples=ck["grpo_eval_samples"],
            inference_temperature=ck["inference_temperature"],
            inference_max_new_tokens=ck["inference_max_new_tokens"],
            seed=ck["seed"],
        )
        router.model_names    = ck["model_names"]
        router._system_prompt = ck["system_prompt"]
        router._adapter_dir   = ck["adapter_dir"]
        return router


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

from .registry import register  # noqa: E402

def _kwargs(args):
    return {
        k: getattr(args, k)
        for k in (
            "base_model", "sft_epochs", "grpo_steps", "grpo_alpha",
            "grpo_eval_steps", "seed",
        )
        if hasattr(args, k)
    }

register("sft_grpo", SFTGRPORouter, _kwargs)
