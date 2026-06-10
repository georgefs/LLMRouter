# LLMRouter × semantic-router RL-driven 整合指南

本指南說明如何將 LLMRouter 作為 semantic-router 的 RL-driven router 進行整合。

## 📋 架構

```
User Query
    ↓
semantic-router (RLDrivenSelector)
    ↓
    ├─ Intent Classification (確定查詢類型)
    ├─ PII Detection (隱私檢測)
    └─ Model Selection (LLMRouter 智能路由)
         ↓
    POST /route to LLMRouter endpoint
         ↓
    LLMRouter (KNN-based)
         ↓
    {"selected_model": "gpt-4"}
         ↓
Route to Selected Model
    ↓
Return Response
```

## 🚀 快速開始

### 1️⃣ 準備 LLMRouter Endpoint

```bash
# 終端 1: 啟動 LLMRouter endpoint
python3 LLMRouter/scripts/start_endpoint.py \
  --router LLMRouter/test_data/test_router.pkl \
  --port 8888

# 輸出:
# ============================================================
# Starting LLMRouter Endpoint Server
# ============================================================
# Router: LLMRouter/test_data/test_router.pkl
# Host: 0.0.0.0
# Port: 8888
# URL: http://localhost:8888
```

**驗證端點正常**:
```bash
curl http://localhost:8888/health
# {"status": "ok", "router_type": "KNNRouter"}
```

### 2️⃣ 配置 semantic-router

**使用提供的配置文件**:
```bash
# 將配置複製到 semantic-router 的配置目錄
cp semantic_router_config.yaml /path/to/semantic-router/config/config.yaml
```

**關鍵配置**（已在 `semantic_router_config.yaml` 中設置）:

```yaml
algorithm:
  type: rl_driven
  rl_driven:
    # ← 這裡是關鍵！
    enable_llm_routing: true
    router_r1_server_url: "http://localhost:8888"  # ← 指向 LLMRouter endpoint
    llm_routing_fallback: thompson
```

### 3️⃣ 啟動 semantic-router

```bash
# 終端 2: 啟動 semantic-router
# （根據你的環境，具體命令可能有所不同）
docker run -p 8899:8899 \
  -v $(pwd)/semantic_router_config.yaml:/etc/semantic-router/config.yaml \
  semantic-router:latest

# 或本地運行:
# make vllm-sr-dev
# vllm-sr serve
```

## 🔄 工作流程詳解

### semantic-router RLDrivenSelector 調用流程

```python
# semantic-router 的 RLDrivenSelector.Select() 方法：

def Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error):
    # 1. 如果啟用 LLM routing，先嘗試 LLM 路由
    if r.config.EnableLLMRouting && r.routerR1Client != nil:
        result, err := r.selectWithRouterR1(ctx, selCtx)
        if err == nil:
            return result, nil
        # 失敗時降級到 Thompson Sampling
        
    # 2. 使用 Thompson Sampling 進行路由決策
    # 3. 返回選擇的模型和信心分數
```

### LLMRouter Endpoint 調用

```python
# semantic-router RouterR1Client 調用：

POST /route
Content-Type: application/json

{
  "query": "Task: code generation\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"
}

# LLMRouter 回應：
{
  "selected_model": "gpt-4"
}
```

## 📊 配置選項解釋

### RL-driven 參數

| 參數 | 值 | 說明 |
|------|-----|------|
| `enable_llm_routing` | `true` | 啟用 LLMRouter 進行路由 |
| `router_r1_server_url` | `http://localhost:8888` | LLMRouter endpoint URL |
| `llm_routing_fallback` | `thompson` | LLM 失敗時降級到 Thompson Sampling |
| `exploration_rate` | `0.15` | 初始探索率 (15%) |
| `exploration_decay` | `0.97` | 每 100 次選擇後探索率衰減 |
| `use_thompson_sampling` | `true` | 使用 Thompson Sampling 平衡探索/利用 |
| `cost_awareness` | `true` | 成本感知探索 |
| `cost_weight` | `0.2` | 成本在決策中的權重 |

### 模型配置

配置中定義了 3 個模型：

| 模型 | 適用場景 | 特點 |
|------|--------|------|
| **gpt-4** | 複雜查詢、代碼生成 | 高能力，較慢，較貴 |
| **gpt-3.5-turbo** | 簡單查詢、通用 | 中等能力，快速，便宜 |
| **claude-3** | 推理、分析 | 高能力，快速 |

### 決策配置

```yaml
decisions:
  - name: coding
    categories: [code_generation, debugging, code_review]
    modelRefs: [gpt-4, claude-3, gpt-3.5-turbo]
    default_model: gpt-4
```

- **code_generation**: 代碼生成優先用 GPT-4
- **debugging**: 調試用 GPT-4 或 Claude-3
- **code_review**: 代碼審查用 GPT-4

## 🔧 定制配置

### 修改 LLMRouter Endpoint URL

如果 LLMRouter endpoint 運行在不同的主機/端口：

```yaml
algorithm:
  rl_driven:
    router_r1_server_url: "http://your-host:your-port"
```

### 修改降級策略

```yaml
algorithm:
  rl_driven:
    llm_routing_fallback: error  # LLM 失敗時直接返回錯誤
    # 或
    llm_routing_fallback: thompson  # LLM 失敗時使用 Thompson Sampling
```

### 調整成本權重

```yaml
algorithm:
  rl_driven:
    cost_weight: 0.3  # 增加成本因素的影響
    cost_reward_alpha: 0.4  # 40% 成本 + 60% 品質
```

## 🧪 測試

### 1. 驗證整合

```bash
# 檢查 LLMRouter endpoint
curl http://localhost:8888/health

# 檢查可用模型
curl http://localhost:8888/models

# 測試路由請求
curl -X POST http://localhost:8888/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Task: code generation\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"}'
```

### 2. 測試 semantic-router

```bash
# 向 semantic-router 發送測試查詢
curl -X POST http://localhost:8899/api/v1/classify/intent \
  -H "Content-Type: application/json" \
  -d '{"text": "請幫我生成一個 Python 函數"}'
```

### 3. 運行集成測試

```bash
# 運行我們的集成測試
python3 -m pytest LLMRouter/test/test_integration_workflow.py -v -s
```

## 📈 監控和調試

### 日誌級別

```yaml
observability:
  logging:
    level: debug  # 設置為 debug 查看詳細日誌
```

### 常見問題

#### 問題 1: 連接拒絕

```
Error: failed to send request to router_r1_server_url: connection refused
```

**解決**:
- 確認 LLMRouter endpoint 已啟動
- 檢查 URL 配置是否正確
- 檢查防火牆設置

#### 問題 2: LLM 路由超時

```
Error: Router-R1 routing failed: context deadline exceeded
```

**解決**:
- 增加超時時間
- 檢查 LLMRouter endpoint 的性能
- 查看降級策略是否工作（應該降級到 Thompson Sampling）

#### 問題 3: 模型名稱不匹配

```
Error: selected model not in candidate list
```

**解決**:
- 檢查 LLMRouter 訓練時使用的模型名稱
- 確認 semantic-router 配置中的模型名稱一致
- 運行 `curl http://localhost:8888/models` 驗證

## 🔐 生產部署建議

1. **使用環境變數存儲 API 密鑰**:
   ```yaml
   backend_refs:
     - api_key_env: OPENAI_API_KEY  # 從環境變數讀取
   ```

2. **配置 LLMRouter endpoint 的負載均衡**:
   ```bash
   # 多個 endpoint 實例
   python3 start_endpoint.py --router ... --port 8888 &
   python3 start_endpoint.py --router ... --port 8889 &
   # 使用 nginx 或負載均衡器
   ```

3. **啟用持久化狀態**:
   ```yaml
   algorithm:
     rl_driven:
       storage_path: /persistent/state/rl-driven.json
       auto_save_interval: 30s
   ```

4. **監控和告警**:
   ```yaml
   observability:
     metrics:
       enabled: true
       interval: 60s
   ```

## 📚 相關資源

- **LLMRouter 文檔**: `LLMRouter/README.md`
- **semantic-router 文檔**: `semantic-router/website/docs/`
- **配置範例**: `semantic_router_config.yaml`
- **集成測試**: `LLMRouter/test/test_integration_workflow.py`

## 🎯 下一步

1. ✅ 啟動 LLMRouter endpoint
2. ✅ 配置 semantic-router
3. ⏳ 測試整合工作流
4. ⏳ 優化路由決策
5. ⏳ 監控性能指標
6. ⏳ 生產部署
