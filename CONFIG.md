# LLMRouter 設定文件

## 目錄

- [環境變數](#環境變數)
- [config.yaml](#configyaml)
- [CLI 常用參數速查](#cli-常用參數速查)

---

## 環境變數


| 變數名稱 | 必填 | 說明 |
|---|---|---|
| `LLMROUTER_CONFIG` | 否 | `config.yaml` 的絕對路徑, 基本上是litellm 的config  |
| `DATA_PATH` | 是 | dataset 根目錄的絕對路徑，對應 `config.yaml` 中的 `data_path` |
| `OPENROUTER_API_BASEURL` | 視需求 | OpenRouter API base URL，例如 `https://openrouter.ai/api/v1` |
| `OPENROUTER_API_KEY` | 視需求 | OpenRouter API 金鑰（`sk-or-v1-...`） |
| `NCHC_API_BASEURL` | 視需求 | NCHC GenAI API base URL |
| `NCHC_API_KEY` | 視需求 | NCHC API 金鑰 |
| `NCHC_API_HOST` | 視需求 | NCHC API host（部分 SDK 需要獨立設定） |

> **注意**：`routebench.env` 含有真實金鑰，請勿提交至版本控制。可將其加入 `.gitignore`。

---

## config.yaml

`config.yaml` 由兩個頂層欄位組成：

```yaml
model_list:   # litellm Router 模型清單
  - ...
data_path:    # dataset 根目錄（支援 !ENV 展開環境變數）/ 自訂的
```

### `data_path`

```yaml
data_path: !ENV ${DATA_PATH}
```

指定 dataset 根目錄。使用 `!ENV` 標籤從環境變數讀取，也可直接寫死絕對路徑。

目錄結構慣例：

```
<data_path>/
  <dataset_name>.jsonl          # 原始問題集
  responses/<dataset>/<model>.jsonl      # 模型回答
  annotations/<dataset>/<strategy>/<model>.jsonl  # LLM-as-Judge 結果
```


## CLI 常用參數速查

所有子命令共用：

```
python -m LLMRouter [--config <path>] [--base <data_path>] <subcommand>
```

| 全域參數 | 說明 |
|---|---|
| `--config` | 指定 `config.yaml` 路徑（覆蓋自動偵測） |
| `--base` | 指定 data 根目錄（優先於 `config.yaml` 的 `data_path`） |

### response gen

```bash
python -m LLMRouter --config config.yaml \
    response gen <dataset> <model_name> \
    [--concurrency 8] [--overwrite]
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--concurrency` | `8` | 同時發出的非同步請求數 |
| `--overwrite` | `false` | 忽略已有結果，重跑全部 |

### annotation gen

```bash
python -m LLMRouter --config config.yaml \
    annotation gen <dataset> <model_name> \
    --strategy llm --judge <judge_model_name> \
    [--concurrency 8] [--overwrite]
```

| 參數 | 必填 | 說明 |
|---|---|---|
| `--strategy` | 是 | 目前支援 `llm`（LLM-as-Judge） |
| `--judge` | 是（strategy=llm） | judge 模型的 `model_name`（需在 `config.yaml` 中定義） |

### router prepare

```bash
python -m LLMRouter --config config.yaml \
    router prepare \
    --datasets <d1,d2> --models <m1,m2> \
    --strategy llm --scorer point \
    -o data.npz \
    [--train-ratio 0.6] [--val-ratio 0.1] [--seed 42] \
    [--emb-model <model>] [--emb-batch-size 32] \
    [--min-var <float>] \
    [--dedup-eps <float>] [--dedup-sample-ratio 0.3]
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `--train-ratio` | `0.6` | 訓練集比例 |
| `--val-ratio` | `0.1` | 驗證集比例（剩餘為 test） |
| `--seed` | `42` | 資料分割隨機種子 |
| `--scorer` | `point` | 評分方式（`point` = 0/1 二元分） |
| `--emb-model` | 無 | 預存 embedding 的模型名稱；不指定則 router fit 時即時計算 |
| `--emb-batch-size` | `32` | embedding 計算 batch size |
| `--min-var` | 無 | 過濾低鑑別度樣本：移除 `var(scores) ≤ min_var` 的訓練資料 |
| `--dedup-eps` | 無 | 語意去重 DBSCAN eps（e.g. `0.15`）；不指定則不啟用 |
| `--dedup-sample-ratio` | `0.3` | 每個重複 cluster 保留比例 |

### router bench

```bash
python -m LLMRouter \
    router bench <router1,router2,...> \
    --data data.npz \
    [--fractions 0.1,0.2,...,1.0] [--sizes 50,100,...] \
    [--repeats 3] \
    [--k 10] [--temperature 0.1] \
    [--latent-dim 64] [--epochs 50] [--lr 0.001] \
    [--emb-model <model>]
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `router_types` | — | 逗號分隔，可用：`oracle`, `random`, `knn`, `mf`, `sw`, `grpo`, `roberta` |
| `--fractions` | `0.1,...,1.0` | 訓練資料比例列表（與 `--sizes` 二擇一） |
| `--sizes` | 無 | 固定筆數列表，e.g. `50,100,200`（與 `--fractions` 二擇一） |
| `--repeats` | `3` | 每個大小重複幾次（不同隨機種子） |
| `--k` | `10` | KNN / SW：近鄰數量 |
| `--temperature` | `0.1` | SW：Softmax 溫度 |
| `--latent-dim` | `64` | MF：隱空間維度 |
| `--epochs` | `50` | MF / RoBERTa：訓練 epochs |
| `--lr` | `0.001` | MF：學習率 |
| `--emb-model` | `mixedbread-ai/mxbai-embed-large-v1` | KNN / MF / SW 嵌入模型 |

### 輸出指標說明

| 指標 | 意義 |
|---|---|
| `mu` | 選出模型的平均分數（越高越好） |
| `vb` | `mu / oracle`，越接近 1 越好 |
| `ep` | 預測分佈 entropy（bits），越高表示路由越分散 |
