# LLMRouter — 開發指引

## 專案定位

多模型 LLM routing 框架，提供：資料集管理、模型回應收集、annotation、router 訓練與評估、HTTP endpoint。
對應 Technical Report §4（評估指標）、§7（四維根因分析）。

---

## 安裝

```bash
# 核心（eval、dataset、endpoint）
./install.sh

# 嵌入式 routers（KNN / SW / MF / RoBERTa）
./install.sh ml

# LLM inference + LLM-as-Judge
./install.sh llm

# SFT+GRPO LLM router（需要 CUDA）
./install.sh rl

# 完整環境
./install.sh all
```

Python >= 3.10 必要。`rl` profile 需要 CUDA GPU。

---

## 套件結構

```
LLMRouter/
  __main__.py          CLI 進入點（所有子命令）
  manager.py           DatasetManager：載入/列出 dataset
  router/
    base.py            BaseRouter：fit / predict_probs / save / load
    data.py            RouterData（NPZ splits）、DataPreparer
    eval.py            RouterBenchmark、§4.3 metrics（HR/Cost/TER/NBS）
    registry.py        register() / build() / list_routers()
    oracle.py          OracleRouter、RandomRouter
    knn.py             KNNRouter（embedding-based）
    mf.py              MFRouter（matrix factorization）
    sw_ranking.py      SWRankingRouter
    roberta_mlc.py     RoBERTaMLCRouter
    grpo.py            GRPORouter（embedding-based RL，MLP policy）
    sft_grpo.py        SFTGRPORouter（LLM-based，Qwen2.5 + LoRA，SFT+GRPO）
    semantic_api.py    SemanticAPIRouter
    _template.py       新增 router 的範本
  annotator/           annotation 策略（llm_judge、official…）
  scorer/              scorer 實作（field、point…）
  endpoint/server.py   RouterEndpointServer（HTTP）
  inferencer.py        LLM 推理 wrapper
  test/                pytest 測試套件
```

---

## CLI 快速參考

```bash
# 列出 dataset / model
python -m LLMRouter dataset list
python -m LLMRouter model list --dataset <ds>

# 對比多個 router（直接撈 dataset，顯示 §4.3 metrics）
python -m LLMRouter router bench oracle,random,knn \
  --datasets mmlu_pro_test --models m1,m2 --strategy llm \
  --fractions 1.0 --repeats 1 --show-cost

# 載入已訓練的 router（type:path 語法）
python -m LLMRouter router bench sft_grpo:path/to/router.pkl \
  --datasets mmlu_pro_test --models m1,m2 --strategy llm

# 單一 router 詳細評估
python -m LLMRouter router eval --datasets mmlu_pro_test \
  --models m1,m2 --strategy llm

# 訓練並儲存
python -m LLMRouter router train knn \
  --datasets mmlu_pro_test --models m1,m2 --strategy llm \
  --output router.pkl

# 分析 dataset routing 價值（§7 四維分析）
python -m LLMRouter router analyze --datasets mmlu_pro_test \
  --models m1,m2 --strategy llm
```

資料來源二選一（互斥）：`--data <npz>` 或 `--datasets D1,D2`（後者需加 `--models`、`--strategy`）。

---

## §4.3 評估指標

| 指標 | 公式 | 說明 |
|------|------|------|
| HR   | correct_routings / total | Hit Rate |
| Cost | avg_tokens × model_unit_cost | 加權成本 |
| TER  | ΔCost_Savings% / ΔHR_Sacrifice% | 效率比；HR ≥ baseline 時顯示 `Inv` |
| NBS  | 3×ΔHR% + ΔCost_Savings% | Net Benefit Score，以 oracle 為基準 |

TER / NBS 以最強 baseline（oracle 除外）為參照點，`--show-cost` 才顯示。

---

## 新增 Router

1. 複製 `LLMRouter/router/_template.py` 為新檔
2. 繼承 `BaseRouter`，實作 `_fit(data)` 與 `predict_probs(prompts)`
3. 實作 `save(path)` / `load(path_or_ck)`（或用 BaseRouter 的 pickle 預設）
4. 檔案底部呼叫 `register("名稱", MyRouter, kwargs_fn)`
5. 在 `LLMRouter/router/__init__.py` 加上 import 與 `__all__` 條目

`kwargs_fn(args)` 從 argparse.Namespace 取超參數，回傳 dict 傳給建構子。

---

## SFTGRPORouter 注意事項

- 依賴 `unsloth`、`trl`、`datasets`；需 CUDA — 不要在 CPU 環境 import 後直接呼叫 `fit()`
- `_run_grpo()` 開頭必須呼叫 `PatchFastRL("GRPO", FastLanguageModel)`（已實作）
- 訓練時優先讀 `raw_train_score`（純 0/1）而非 `train_score`（可能有懲罰分）
- System prompt menu 格式：`Index i: model_name`，**不帶成本標注**（與 RouterEval reference 一致）
- Reward 公式：`FORMAT_BONUS(0.05) + max(1.0 + 1/n_correct - α×penalty, 0.0)`
- `save()` 輸出：`<name>.pkl`（metadata）+ `<name>_adapter/`（LoRA weights），兩者必須並排

---

## 測試

```bash
# 跑全部測試
pytest LLMRouter/test/

# 跑單一檔
pytest LLMRouter/test/test_eval_metrics.py -v

# 跳過需要 GPU / 外部 API 的測試
pytest LLMRouter/test/ -m "not slow"
```

KNN benchmark 在大 dataset（7k+）上跑嵌入計算耗時很長，測試時可用小型合成 RouterData（200 筆）。

---

## 重要注意事項

- `DataPreparer.from_manager()` 是從 DatasetManager 直接建 RouterData 的入口，**不要另外寫 NPZ 再讀回**
- `RouterBenchmark.strongest_baseline()` 以非 oracle 的 router 中 HR 最高者作為 TER/NBS 基準；沒有其他 router 時回傳 `None`
- `model_unit_costs()` 從 `MODEL_PRICING` dict 查單價；未知 model 回傳 0，Cost 欄位會顯示 token 數而非金額
- `save_strategy="no"` 在 SFT TrainingArguments 中是刻意設定，最終 checkpoint 統一存到 `final/`
- `.gitignore` 排除了 `models/`、`datasets/`、`*.pkl`、`*.npz` — 訓練產物**不進 git**
