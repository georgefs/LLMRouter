# LLMRouter

Data-driven LLM router 訓練與評估框架。

Router 的品質取決於資料——相同的 KNN 演算法，在不同資料集上的 Hit Rate 可能天差地遠。LLMRouter 的核心理念是：**先評估資料集的 routing 價值，再客製化訓練最適合這批資料的 router**，而不是套用通用 router 或靠直覺選模型。

---

## 核心特性

**Data-driven 訓練流程**
從 inference → LLM-as-Judge annotation → RouterData 一條龍，所有資料都存在你自己的 DATA_PATH，可斷點續跑、可重複實驗。

**資料集品質評估（§7 四維分析）**
在訓練 router 之前，先評估這個資料集適不適合訓練。指標：CH Score（embedding 可分性）、Avg_Sim（語意結構）、Dec_Var（模型鑑別度）、N（樣本量）。POOR 的資料集訓練出來的 router 不會比 random 好多少。

**多 router 橫向對比（§4.3 metrics）**
一行指令同時評估多個 router 實作，輸出 HR / Cost / TER / NBS，精確量化「花多少 HR 換多少省錢」的 trade-off。

**納入 semantic router 一起評估**
把正在運行的 semantic router 包成 `SemanticAPIRouter`，直接放進同一個 benchmark，與 KNN、GRPO、SFT+GRPO 等本地 router 並排對比。

**部署為 HTTP Endpoint**
訓練好的 router 可作為獨立 HTTP server，接受 `POST /route` 請求回傳選定模型，並可直接掛入 semantic router 的 `rl_driven.router_r1_server_url`。

---

## 架構

```
你的資料                       LLMRouter 核心                       外部整合
─────────                      ────────────                          ────────

datasets/         →  response gen     \                              semantic router
responses/        →  annotation gen    ├─→  RouterData (.npz)  →  ┌─── SemanticAPIRouter
annotations/      →  router prepare  /    │                        │      （wrap HTTP API）
                                          ↓                        │
                               ┌─────────────────────┐             │  HTTP Endpoint
                               │    RouterBenchmark   │  ←─────────┘  POST /route
                               │  oracle / random     │
                               │  knn / mf / roberta  │
                               │  grpo / sft_grpo     │
                               │  semantic_api        │
                               └─────────────────────┘
                                         ↓
                               HR / Cost / TER / NBS
```

---

## Router 實作

| Router | 類型 | 特性 | 依賴 |
|--------|------|------|------|
| `oracle` | 基準上界 | 每次都選最佳模型 | — |
| `random` | 基準下界 | 隨機選模型 | — |
| `knn` | embedding-based | K 近鄰，fast & 穩定 | `ml` extras |
| `mf` | 矩陣分解 | 協同過濾風格 | `ml` extras |
| `sw_ranking` | sliding window | 依排行得分分配 | `ml` extras |
| `roberta_mlc` | fine-tuned LM | RoBERTa 多標籤分類 | `ml` extras |
| `grpo` | RL (embedding) | MLP policy + GRPO，不需 GPU | `ml` extras |
| `sft_grpo` | RL (LLM) | Qwen2.5 + LoRA，SFT → GRPO 兩階段 | `rl` extras (CUDA) |
| `semantic_api` | HTTP proxy | 包裝外部 semantic router API | — |

---

## 快速開始

### 安裝

```bash
./install.sh        # 核心（eval、dataset、endpoint）
./install.sh ml     # + embedding-based routers
./install.sh rl     # + SFTGRPORouter（需 CUDA）
./install.sh all    # 全部
```

Python >= 3.10。

### 端對端範例

```bash
# 1. 查看可用 dataset
python3 -m LLMRouter dataset list

# 2. 產生 model response
python3 -m LLMRouter response gen mmlu_pro_test gpt-oss-20b

# 3. LLM-as-Judge annotation
python3 -m LLMRouter annotation gen mmlu_pro_test gpt-oss-20b \
  --strategy llm --judge gpt-oss-120b

# 4. 評估資料集 routing 價值（先做，再決定要不要繼續）
python3 -m LLMRouter.scripts.analyze_datasets \
  --datasets mmlu_pro_test \
  --models   gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B \
  --strategy llm

# 5. 打包 RouterData
python3 -m LLMRouter router prepare \
  --datasets mmlu_pro_test \
  --models   gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B \
  --strategy llm -o data.npz

# 6. 橫向對比多個 router（含 §4.3 metrics）
python3 -m LLMRouter router bench oracle,random,knn \
  --data data.npz --fractions 1.0 --show-cost
```

---

## 橫向評估指標（§4.3）

| 指標 | 意義 | 方向 |
|------|------|------|
| **HR** | Hit Rate：選出模型達到最高分的比例 | ↑ |
| **Cost** | 平均 token 數 × model 單價 | ↓ |
| **TER** | Cost_Savings% ÷ HR_Sacrifice%（效率比） | ↑ |
| **NBS** | 3×ΔHR% + ΔCost_Savings%（淨收益） | ↑ |

TER / NBS 以最強非 oracle baseline 為參照；HR ≥ baseline 時 TER 顯示 `Inv`（Dominant）。

範例輸出：

```
router          size  n_train       HR        Cost         TER       NBS
------------------------------------------------------------------------
oracle          100%     7219   1.0000      844.80           —         —
random          100%     7219   0.7028      772.91        0.29    -80.66
knn             100%     7219   0.8438     1063.72         Inv    -72.78
sft_grpo        100%     7219   0.9012      891.20         Inv     +8.43
```

---

## 整合 semantic router

### 方向 A：把 semantic router 納入評估

```python
from LLMRouter.router import SemanticAPIRouter, RouterBenchmark, RouterData

data = RouterData.load("data.npz")
bench = RouterBenchmark(data)

# semantic router 與本地 router 放進同一個 benchmark
bench.run(SemanticAPIRouter,
          {"base_url": "http://localhost:8080", "timeout": 10},
          label="semantic_api")
bench.print_table(show_cost_metrics=True)
```

### 方向 B：把訓練好的 router 部署給 semantic router 使用

```bash
# 啟動 HTTP Endpoint（監聽宿主機 port）
python3 -m LLMRouter.scripts.start_endpoint \
    --router best_router.pkl --port 8888
```

```yaml
# semantic_router.yaml
# router_r1_server_url 用宿主機 IP，不能用 localhost
# （semantic router 以 Docker 部署，容器內 localhost = 容器自身）
routing:
  rl_driven:
    enabled: true
    router_r1_server_url: http://172.17.0.1:8888   # docker0 bridge IP
    llm_routing_fallback: thompson
    router_r1_timeout: 5
```

宿主機 IP 查詢：`ip addr show docker0 | grep 'inet '`；自訂 network 請用 `docker network inspect <name> | grep Gateway`。

---

## 資料集四維分析（§7）

在投入訓練資源之前，先確認這個資料集有沒有 routing 價值：

```bash
python3 -m LLMRouter.scripts.analyze_datasets \
  --datasets mmlu_pro_test,arc_challenge,gpqa_diamond \
  --models   gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B \
  --strategy llm --detail --output analysis.csv
```

| 指標 | 閾值 | 意義 |
|------|------|------|
| CH Score | > 2.0 | embedding 空間中不同「最佳模型」群組可否被分離 |
| Avg_Sim | > 0.025 | prompt 特徵空間的語意結構密度 |
| Dec_Var σ² | > 0.015 | 模型間能力差距——差距太小代表 router 無用武之地 |
| N | ≥ 3,000 | 訓練樣本基線 |

評級：**GOOD** → 推薦訓練；**MARGINAL** → 可試訓；**POOR** → 先改善資料集。

---

## 情景範例

詳細說明與完整驗證 script 存放於 [`examples/`](examples/)。

---

### 情景 1 — 從零開始：inference → 訓練 → benchmark

**適用時機**：手上只有 dataset，要從頭跑完整個 pipeline。

涵蓋 inference、LLM-as-Judge annotation、§7 資料集品質檢查（CH Score < 2 時警告）、RouterData 打包（含預存 embedding）、多 router 橫向對比、儲存最終 router。

```bash
export DATASETS=mmlu_pro_test
export MODELS="gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B"
export JUDGE=gpt-oss-120b
bash examples/01_full_workflow.sh
```

→ 產出：`data.npz`、`best_router.pkl`、`dataset_analysis.csv`

[`examples/01_full_workflow.sh`](examples/01_full_workflow.sh)

---

### 情景 2 — 批次評估多個資料集的 routing 價值

**適用時機**：有多個 dataset，要決定哪些值得投入訓練資源。

批次計算四維指標、自動分類 GOOD / MARGINAL / POOR、輸出診斷報告與 CSV，並驗證 grade 與數值一致性。

```bash
export DATASETS=mmlu_pro_test,arc_challenge,gpqa_diamond
export MODELS="gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B"
bash examples/02_dataset_analysis.sh
# → analysis_results.csv + analysis_results.json
```

POOR 的資料集：CH Score 低 → router 在 embedding 空間找不到可分離的決策邊界，訓練成果不會比 random 好。優先修復資料集（換 embedding 模型、擴大樣本、調整模型池），再投入訓練。

[`examples/02_dataset_analysis.sh`](examples/02_dataset_analysis.sh)

---

### 情景 3 — 訓練 router → 部署 endpoint → 串接 semantic router

**適用時機**：已有 RouterData，要把 router 部署給 semantic router 作為 `rl_driven` 決策引擎。

訓練 KNN router → 啟動 HTTP endpoint → 驗證 `/health` / `/models` / `/route` API → 自動產生 `semantic_router.yaml`。

```bash
export DATA_NPZ=data.npz
bash examples/03_deploy_endpoint.sh
# → deployed_router.pkl + semantic_router.yaml
```

產生的 config 直接可用：

```yaml
routing:
  rl_driven:
    enabled: true
    # 使用宿主機 IP，而非 localhost
    # （semantic router 以 Docker 部署，容器內無法用 localhost 連到宿主機）
    router_r1_server_url: http://172.17.0.1:8888
    llm_routing_fallback: thompson   # LLMRouter 不可用時自動降級
    router_r1_timeout: 5
```

script 會自動偵測宿主機 IP（`docker0` 介面 → 預設路由 → `host.docker.internal`），也可手動覆蓋：`export DOCKER_HOST_IP=<ip>`。

[`examples/03_deploy_endpoint.sh`](examples/03_deploy_endpoint.sh)

---

### 情景 4 — 把 semantic router 納入橫向評估

**適用時機**：semantic router 已在運行，想知道它的路由品質相對於 oracle / random / KNN 落在哪裡。

`SemanticAPIRouter` 把 `POST /api/v1/classify/intent` 包成 `BaseRouter`，可直接放進 `RouterBenchmark` 與其他 router 並排比較。內建 mock server，無需真實 semantic router 即可驗證流程。

```bash
export DATA_NPZ=data.npz
export SR_BASE_URL=http://localhost:8080   # 若未運行，script 自動啟動 mock
bash examples/04_semantic_api_router.sh
```

結果解讀：
- HR 明顯高於 random → semantic router RL 訓練有效
- HR 接近 random → 需要更多 feedback 資料
- HR 接近 oracle → 考慮直接用它，不需額外訓練本地 router

[`examples/04_semantic_api_router.sh`](examples/04_semantic_api_router.sh)

---

## 文件

| 文件 | 內容 |
|------|------|
| [`LLMRouter/README.md`](LLMRouter/README.md) | 完整 CLI reference、Python API、所有參數說明 |
| [`CLAUDE.md`](CLAUDE.md) | 開發指引、注意事項、新增 router checklist |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | 系統架構、資料流、模組職責 |
| [`examples/`](examples/) | 4 個情景範例（含完整驗證 script） |

---

## 新增自訂 Router

1. 複製 `LLMRouter/router/_template.py`
2. 繼承 `BaseRouter`，實作 `_fit(data)` 與 `predict_probs(prompts)`
3. `register("名稱", MyRouter, kwargs_fn)` 加入 registry
4. 在 `LLMRouter/router/__init__.py` 加 import

詳見 [`CLAUDE.md`](CLAUDE.md)。
