# LLMRouter Examples

四個獨立情景範例，每個都附帶完整驗證步驟。

---

## 情景總覽

| 範例 | 情景 | 前提 |
|------|------|------|
| [01_full_workflow.sh](01_full_workflow.sh) | 從 inference → annotation → 資料集分析 → router benchmark 完整流程 | dataset 已在 DATA_PATH |
| [02_dataset_analysis.sh](02_dataset_analysis.sh) | §7 四維指標批次分析，評估哪些 dataset 適合訓練 router | dataset 已有 response + annotation |
| [03_deploy_endpoint.sh](03_deploy_endpoint.sh) | 訓練 router → 部署 HTTP endpoint → 產生 semantic router 串接 config | RouterData .npz |
| [04_semantic_api_router.sh](04_semantic_api_router.sh) | 將 semantic router 加入 RouterBenchmark 橫向評估 | RouterData .npz + semantic router 運行中（或使用內建 mock） |

---

## 快速開始

```bash
# 設定共用環境變數
export DATA_NPZ=data.npz
export DATASETS=mmlu_pro_test
export MODELS="gpt-oss-20b,Microsoft-Phi-4,Google-Gemma-3-27B"
export STRATEGY=llm
export JUDGE=gpt-oss-120b

# 情景 1：完整端到端流程（從 inference 開始）
bash examples/01_full_workflow.sh

# 情景 2：只做資料集分析（已有 response + annotation）
bash examples/02_dataset_analysis.sh

# 情景 3：部署訓練好的 router（需先跑情景 1 得到 data.npz）
bash examples/03_deploy_endpoint.sh

# 情景 4：SemanticAPIRouter 評估（情景 3 的 endpoint 要先啟動，或使用內建 mock）
bash examples/04_semantic_api_router.sh
```

---

## 情景 1 — 完整端到端 workflow

**`01_full_workflow.sh`**

```
Step 1  查看 dataset / model 清單，確認 dataset 存在
Step 2  對每個 model 產生 response（斷點續跑）
Step 3  LLM-as-Judge annotation
Step 4  確認 response + annotation 三層資料齊全
Step 5  §7 四維分析（CH Score < 2 時發出警告並詢問是否繼續）
Step 6  router prepare → RouterData .npz（含預存 embedding）
Step 7  router bench：oracle / random / knn 橫向對比，顯示 §4.3 metrics
Step 8  訓練 KNN router 並儲存 .pkl，驗證 checkpoint + quick eval
```

**驗證重點**：
- Step 4 確認每個 model 都有 annotation
- Step 5 解析 CSV 自動判斷 POOR，詢問是否繼續
- Step 6 驗證 .npz 包含所有必要欄位
- Step 8 驗證 checkpoint router_type / model_names

**關鍵輸出**：
```
data.npz              RouterData（train/val/test splits + embedding）
best_router.pkl       KNN router checkpoint
dataset_analysis.csv  §7 四維分析報告
```

---

## 情景 2 — §7 四維資料集分析

**`02_dataset_analysis.sh`**

```
Step 1  批次分析，印橫向對比表
Step 2  解析 CSV，分類 GOOD / MARGINAL / POOR
Step 3  對 GOOD + MARGINAL datasets 印完整診斷報告（含修復建議）
Step 4  數值合理性驗證（範圍檢查 + grade 一致性）
```

**四個指標與閾值**：

| 優先 | 指標 | 閾值 | 意義 |
|------|------|------|------|
| P1 | CH Score | > 2.0 | 標籤能否在 embedding 空間被分離（最關鍵） |
| P2 | Avg_Sim | > 0.025 | 特徵空間的語意結構 |
| P3 | Dec_Var σ² | > 0.015 | 模型間能力差距夠大 |
| — | N | ≥ 3,000 | 訓練樣本數基線 |

**修復方向**（POOR）：
- CH 低 → 換更敏感的 embedding 模型 / 合併重疊 GT 類別
- Avg_Sim 低 → 縮小 domain 範圍 / 增加 prompt density
- Dec_Var 低 → 剔除能力重疊的模型

---

## 情景 3 — 訓練 Router → 部署 → 串接 semantic router

**`03_deploy_endpoint.sh`**

```
Step 1  訓練 KNN router（100% 訓練資料）
Step 2  驗證 checkpoint（router_type + model_names）
Step 3  啟動 LLMRouter HTTP Endpoint（背景）
Step 4  API 功能驗證：
          GET  /health  → status=healthy, model_count > 0
          GET  /models  → models 清單不為空
          POST /route   → 正常查詢回傳 selected_model
          POST /route   → 一致性：同 query 5 次結果一致
          POST /route   → 空 query 回傳 400
Step 5  產生 semantic_router.yaml（最小化設定）
Step 6  說明串接流程
Step 7  清理
```

**semantic_router.yaml 重點**（自動生成）：
```yaml
routing:
  rl_driven:
    enabled: true
    router_r1_server_url: http://localhost:8888   # LLMRouter endpoint
    llm_routing_fallback: thompson               # 不可用時降級
    router_r1_timeout: 5
```

**持續部署**（不受 script 結束影響）：
```bash
nohup python3 -m LLMRouter.scripts.start_endpoint \
    --router deployed_router.pkl --port 8888 \
    > endpoint.log 2>&1 &
```

---

## 情景 4 — SemanticAPIRouter 整合評估

**`04_semantic_api_router.sh`**

```
Step 0  若 semantic router 未在 8080 運行，啟動 mock server（隨機選模型）
Step 1  驗證 semantic router API 格式（POST /api/v1/classify/intent）
Step 2  CLI 訓練 SemanticAPIRouter（fit = 驗證連線 + 記錄 model_names）
Step 3  CLI bench：semantic_api vs oracle / random / knn，含 §4.3 metrics
Step 4  Python API：直接操作 RouterBenchmark，完整評估流程
Step 5  驗證 predict_probs：one-hot 格式、sum=1、index 有效
Step 6  清理
```

**核心特性**：
- `fit()` 只驗證連線，routing 邏輯完全由 semantic router 管理
- `predict_probs()` 對每個 prompt 發一次 HTTP 請求，回傳 one-hot (N, M)
- 可直接放入 `RouterBenchmark.run()` 與其他 router 等值比較
- 無 GPU / 無本地 model，純 HTTP 橋接

**結果解讀**：
- HR 高於 random → semantic router RL 訓練有效
- HR 接近 random → semantic router 可能需要更多訓練資料
- TER > 1 → 在可接受的 HR 代價下有效降低成本

---

## 注意事項

1. **情景 1-4 共用環境變數**，建議在執行前統一 `export`
2. **情景 3、4 都會在背景啟動 server**，trap 確保 script 結束時自動清理
3. **情景 4 的 mock server** 回傳隨機結果，SemanticAPIRouter 的 HR 會接近 random；接真實 semantic router 才有意義
4. **情景 2 的 Step 4** 會驗證 grade 與數值一致性，若 grade 與閾值不符會報錯
