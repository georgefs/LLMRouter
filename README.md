# LLMRouter

LLM Router 評估系統，用於測試和比較不同 LLM 模型在特定資料集上的表現。

## 快速開始

### 使用 Docker (推薦)

```bash
# 1. 複製環境變數範本
cp .env.example .env

# 2. 編輯 .env，填入你的 API keys

# 3. 啟動容器
docker-compose up -d

# 4. 進入容器
docker-compose exec llmrouter bash

# 5. 在容器內執行命令
./cli/add_model_response.py --dataset gsm8k --model "gpt-oss-20b"
```

### 本地安裝

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定環境變數
export OPENROUTER_API_KEY=your-key
export NCHC_API_KEY=your-key
export LLMROUTER_CONFIG=$(pwd)/config.yaml

# 3. 執行命令
./cli/add_model_response.py --dataset gsm8k --model "gpt-oss-20b"
```

## 主要功能

### 1. 載入資料集

```python
from LLMRouter import datasets

data = datasets.load_dataset(
    ["gsm8k"],
    ["gpt-oss-20b", "Llama-3.1-70B"],
    ["similar"]
)
```

### 2. 生成模型回應

```bash
./cli/add_model_response.py \
  --dataset gsm8k_train \
  --model "gpt-oss-20b"
```

### 3. 生成評估結果

```bash
./cli/add_model_response_eval.py \
  --dataset gsm8k_train \
  --model "gpt-oss-20b" \
  --eval_method similar
```

### 4. 準備訓練資料

```bash
./cli/prepare_routereval.py \
  --dataset gsm8k_train \
  --models "gpt-oss-20b,Llama-3.1-70B" \
  --output /path/to/output.pkl
```

## 執行測試

```bash
# 本地執行
pytest

# 使用 Docker
docker-compose --profile test up test
```

## 專案結構

```
.
├── LLMRouter/              # 主要程式碼
│   └── datasets/           # 資料集處理
│       └── evals/          # 評估方法
├── cli/                    # CLI 工具
├── tests/                  # 測試檔案
├── datasets/               # 資料集目錄
├── config.yaml             # 模型設定
└── CLAUDE.md              # 詳細開發文檔
```

## 文檔

- [CLAUDE.md](CLAUDE.md) - 完整的開發文檔和架構說明
- [tests/README.md](tests/README.md) - 測試說明

## 授權

請參考專案授權文件。
