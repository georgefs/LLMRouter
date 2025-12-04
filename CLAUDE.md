# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

這是一個 LLM Router 評估系統,用於測試和比較不同 LLM 模型在特定資料集上的表現。系統使用 LiteLLM Router 管理多個模型端點,並提供資料集載入、模型回應生成和評估功能。

## 環境設定

### 必要環境變數

在執行任何命令前,必須設定以下環境變數:

```bash
export OPENROUTER_API_BASEURL=<your-openrouter-base-url>
export OPENROUTER_API_KEY=<your-openrouter-api-key>
export NCHC_API_BASEURL=<your-nchc-base-url>
export NCHC_API_KEY=<your-nchc-api-key>
export LLMROUTER_CONFIG=/home/george/work/delta/routebench2/config.yaml
```

參考 `env.sample` 檔案以獲取設定範本。

### 安裝依賴

```bash
pip install -r requirements.txt
```

主要依賴包括:
- `litellm[proxy]` - 用於路由和管理多個 LLM API
- `openai==1.106.1` - OpenAI API 客戶端
- `pyaml_env` - 支援環境變數的 YAML 解析

## 核心架構

### 資料結構

資料以 JSONL 格式儲存,依據 `dataset`、`model`、`eval_method` 組織:

```
datasets/
├── datasets/{dataset_name}.jsonl          # 原始問答資料集
├── responses/{dataset}/{model}.jsonl      # 模型回應
└── evals/{dataset}/{eval_method}/{model}.jsonl  # 評估結果
```

### 模型設定

所有模型在 `config.yaml` 中定義,使用 LiteLLM 格式:

```yaml
model_list:
  - model_name: <internal-name>
    litellm_params:
      model: <provider/model-id>
      api_base: !ENV ${API_BASEURL}
      api_key: !ENV ${API_KEY}
```

模型名稱在檔案系統中會被 URL encode (`urllib.parse.quote_plus`)。

### 評估方法

位於 `LLMRouter/datasets/evals/`:
- `similar.py` - 使用 sentence transformers (mxbai-embed-large-v1) 計算答案與 ground truth 的 cosine similarity
- `random.py` - 隨機評分(用於測試)

每個評估模組必須實作 `eval(dataset, response)` 函數。

## 常用命令

### 1. 載入資料集

```python
from LLMRouter import datasets

# 載入特定資料集、模型和評估方法的結果
data = datasets.load_dataset(
    ["gsm8k"],  # 資料集列表
    ["gpt-oss-20b", "Llama-3.1-70B"],  # 模型列表
    ["similar"]  # 評估方法列表
)
```

### 2. 新增模型回應

使用 CLI:
```bash
./cli/add_model_response.py --dataset gsm8k_train --model "gpt-oss-20b"
```

或使用 Python API:
```python
from LLMRouter import datasets
datasets.add_model_response("gsm8k", "gpt-oss-20b")
```

此命令會:
- 讀取指定資料集的問題
- 透過 LiteLLM Router 呼叫模型
- 儲存回應、token 使用量和回應時間
- 支援斷點續傳(已存在的回應會被跳過)

### 3. 新增評估結果

使用 CLI:
```bash
./cli/add_model_response_eval.py --dataset gsm8k_train --model "gpt-oss-20b" --eval_method similar
```

或使用 Python API:
```python
from LLMRouter import datasets
datasets.add_model_response_eval("gsm8k", "gpt-oss-20b", "similar")
```

### 4. 準備 Router 評估資料集

```bash
./cli/prepare_routereval.py \
  --dataset gsm8k_train \
  --models "gpt-oss-20b,Llama-3.1-70B,Llama-3.1-8B-Instruct" \
  --output /path/to/output.pkl
```

此命令會:
- 載入指定資料集和模型的回應與評估
- 將評估分數二值化(threshold=0.5)
- 分割資料為 train/val/test (60/20/20)
- 輸出包含 prompts、scores、tokens、times 的 pickle 檔案
- 顯示每個模型的準確率和效能統計

### 5. 執行測試

```bash
# 執行所有測試
pytest

# 執行特定測試檔案
pytest tests/test_datasets.py

# 執行特定測試類別或函數
pytest tests/test_datasets.py::TestLoadDataset
pytest tests/test_evals.py::TestSimilarEval::test_cosine_similarity_identical_vectors

# 執行特定標記的測試
pytest -m unit              # 僅執行單元測試
pytest -m "not slow"        # 跳過耗時測試
pytest -m integration       # 僅執行整合測試

# 產生測試覆蓋率報告
pytest --cov=LLMRouter --cov-report=html

# 顯示詳細輸出
pytest -v -s
```

## 資料流程

典型的工作流程:

1. **準備資料集** - 將原始資料集轉換為 JSONL 格式,放置於 `datasets/datasets/`
2. **生成回應** - 使用 `add_model_response` 為每個模型生成回應
3. **評估回應** - 使用 `add_model_response_eval` 評估模型回應品質
4. **載入完整資料** - 使用 `load_dataset` 合併所有資料用於分析
5. **準備訓練資料** - 使用 `prepare_routereval` 生成用於 router 訓練的資料集

## 重要實作細節

### LiteLLM Router
- 全域單例在 `LLMRouter/datasets/__init__.py` 初始化
- 所有模型呼叫統一透過 `litellm_router.completion()`
- 自動處理不同 provider 的 API 差異

### 資料集格式
- 問答資料集必須包含 `question` 和 `answer` 欄位
- GSM8K 格式答案以 `####` 分隔說明和最終答案
- 每筆資料必須有唯一的 `key` (格式: `{dataset}_{index}`)

### 錯誤處理
- `add_model_response` 使用 `.tmp` 暫存檔避免中斷導致資料損壞
- 已存在的回應會被跳過以支援斷點續傳
- 缺少的回應或評估在 `load_dataset` 中會被忽略

### 檔案路徑處理
- `real_path()` 函數將相對路徑轉換為 `dataset_path` 下的絕對路徑
- 自動建立缺失的目錄
- 模型名稱使用 URL encoding 避免檔名衝突

## 測試架構

### 測試組織

測試位於 `tests/` 目錄，使用 pytest 框架，**僅涵蓋核心公開 API**:

```
tests/
├── conftest.py          # 共用 fixtures 和測試設定
├── test_datasets.py     # 資料集核心功能測試 (5 個測試)
└── test_evals.py        # 評估方法核心功能測試 (4 個測試)
```

### 測試涵蓋的核心功能

**test_datasets.py** - 資料集主要 API:
- `load_dataset()` - 載入完整資料集（含單/多模型）
- `add_model_response()` - 生成模型回應
- `add_model_response_eval()` - 生成評估結果

**test_evals.py** - 評估方法:
- `random.eval()` - 隨機評估
- `similar.eval()` - 相似度評估
- `cosine_similarity()` - 向量相似度計算

### 測試哲學

這些測試專注於**主要功能而非實作細節**，特點：
- 只測試公開 API，不測試內部函數（如 `load_file`, `real_path`）
- 避免測試實作細節（如 URL encoding）
- 方便未來重構，不會因內部實作改變而需大量修改測試

### 重要 Fixtures

- `temp_dataset_dir` - 暫時的資料集目錄
- `sample_qa_data` - 範例問答資料
- `mock_litellm_completion` - Mock LiteLLM 回應
- `create_test_dataset/response/eval` - 建立測試檔案的輔助函數
