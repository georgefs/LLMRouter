# LLMRouter Examples

四個獨立情景範例，每個都是獨立資料夾，設定與流程腳本分離。

---

## 結構說明

每個情景資料夾包含：

```
examples/XX_name/
  config.sh        # 使用者設定（資料集、模型、路徑、port 等）
  run.sh           # 流程腳本，source config.sh
  [其他設定檔]      # 如 03 的 semantic_router.yaml
```

**執行方式**（在任意目錄均可）：
```bash
vim examples/03_deploy_endpoint/config.sh   # 修改設定
bash examples/03_deploy_endpoint/run.sh     # 執行
```

所有輸出檔案（.pkl、.npz、.csv）預設存在對應的情景資料夾內。
所有 config.sh 中的變數均可在執行前以環境變數覆蓋：
```bash
ENDPOINT_PORT=9000 bash examples/03_deploy_endpoint/run.sh
```

---

## 情景總覽

| 資料夾 | 情景 | 主要產出 |
|--------|------|----------|
| [01_full_workflow/](01_full_workflow/) | inference → annotation → §7 分析 → RouterData → bench → 訓練 | `data.npz`、`best_router.pkl` |
| [02_dataset_analysis/](02_dataset_analysis/) | 批次 §7 四維分析，評估 routing 價值 | `analysis_results.csv` |
| [03_deploy_endpoint/](03_deploy_endpoint/) | 訓練 router → HTTP endpoint → 串接 semantic router | `router.pkl`、`sr_resolved.yaml` |
| [04_semantic_api_router/](04_semantic_api_router/) | 把 semantic router 包成 BaseRouter 參與橫向評估 | `semantic_api_router.pkl` |

---

## 情景 1 — 完整端到端 workflow

**[01_full_workflow/](01_full_workflow/)**

```
config.sh  DATASETS, MODELS, JUDGE, EMB_MODEL, FRACTIONS, REPEATS …
run.sh
  Step 1  確認 dataset 存在
  Step 2  對每個 model 產生 response（斷點續跑）
  Step 3  LLM-as-Judge annotation（斷點續跑）
  Step 4  確認三層資料齊全
  Step 5  §7 四維分析（CH Score < 2 時警告並詢問是否繼續）
  Step 6  router prepare → data.npz（含預存 embedding）
  Step 7  router bench：oracle / random / knn（§4.3 metrics）
  Step 8  訓練 KNN router → best_router.pkl，驗證 + quick eval
```

---

## 情景 2 — §7 四維資料集分析

**[02_dataset_analysis/](02_dataset_analysis/)**

```
config.sh  DATASETS, MODELS, STRATEGY, EMB_MODEL …
run.sh
  Step 1  批次分析，印橫向對比表，輸出 CSV
  Step 2  解析 CSV，分類 GOOD / MARGINAL / POOR，輸出 JSON
  Step 3  對 GOOD + MARGINAL datasets 印完整診斷報告
  Step 4  數值合理性驗證（範圍檢查 + grade 一致性）
```

**四個指標**：

| 指標 | 閾值 | 意義 |
|------|------|------|
| CH Score | > 2.0 | embedding 空間可分性（最關鍵） |
| Avg_Sim | > 0.025 | 特徵空間語意結構 |
| Dec_Var σ² | > 0.015 | 模型能力差距 |
| N | ≥ 3,000 | 樣本量基線 |

---

## 情景 3 — 訓練 Router → 部署 Endpoint → 串接 semantic router

**[03_deploy_endpoint/](03_deploy_endpoint/)**

```
config.sh              DATA_NPZ, ROUTER_PKL, ENDPOINT_PORT, DOCKER_HOST_IP …
semantic_router.yaml   semantic router 設定模板（含 ${DOCKER_HOST_IP}、${ENDPOINT_PORT} 佔位符）
run.sh
  Step 1  訓練 KNN router → router.pkl（存在此情景資料夾）
  Step 2  讀取 checkpoint 模型清單，展開 semantic_router.yaml → sr_resolved.yaml
  Step 3  啟動 LLMRouter Endpoint（背景）
  Step 4  API 驗證：/health、/models、/route、空 query → 400
  Step 5  印部署說明（sr_resolved.yaml 路徑、持續運行指令）
  Step 6  清理
```

**semantic_router.yaml 說明**：

`semantic_router.yaml` 含變數佔位符，`run.sh` 用 `envsubst` 展開後產生 `sr_resolved.yaml`：

```yaml
rl_driven:
  router_r1_server_url: "http://${DOCKER_HOST_IP}:${ENDPOINT_PORT}"
```

`DOCKER_HOST_IP` 由 `run.sh` 自動偵測（`docker0` → 預設路由 → `host.docker.internal`），
或在 `config.sh` 手動指定。

---

## 情景 4 — SemanticAPIRouter 整合橫向評估

**[04_semantic_api_router/](04_semantic_api_router/)**

```
config.sh  DATA_NPZ, SR_BASE_URL, SR_TIMEOUT, SR_ROUTER_PKL …
run.sh
  Step 0  若 semantic router 未在 SR_BASE_URL 運行，啟動 mock server
  Step 1  驗證 semantic router API 格式
  Step 2  訓練 SemanticAPIRouter（fit = 驗證連線）→ semantic_api_router.pkl
  Step 3  CLI bench：semantic_api vs oracle / random / knn（§4.3 metrics）
  Step 4  Python API：RouterBenchmark 完整評估
  Step 5  驗證 predict_probs：one-hot 格式、sum=1、index 有效
  Step 6  清理
```

**mock server**：若無真實 semantic router，Step 0 自動啟動 mock（回傳隨機模型）。
mock 下 SemanticAPIRouter 的 HR 接近 random，屬正常行為。

---

## 情景間的資料流

```
01_full_workflow/data.npz  ──→  03_deploy_endpoint/  （DATA_NPZ 預設指向此路徑）
                           ──→  04_semantic_api_router/

03_deploy_endpoint/sr_resolved.yaml  ──→  semantic-router start --config ...
```
