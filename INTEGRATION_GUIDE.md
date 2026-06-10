# LLMRouter × semantic-router RL-driven 整合指南

本指南說明如何將 LLMRouter 作為 semantic-router 的 RL-driven router 進行整合。

## 📋 架構

```
User Query
    ↓
semantic-router (RLDrivenSelector)
    ↓
    ├─ Decision Matching (catch-all: general)
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
  --port 8123

# 輸出:
# ============================================================
# Starting LLMRouter Endpoint Server
# ============================================================
# Router: LLMRouter/test_data/test_router.pkl
# Host: 0.0.0.0
# Port: 8123
# URL: http://localhost:8123
```

**驗證端點正常**:
```bash
curl http://localhost:8123/health
# {"status": "healthy", "router_type": "KNNRouter", "model_count": 3}
```

### 2️⃣ 配置 semantic-router

使用提供的配置文件 `semantic_router_config_minimal.yaml`。

**關鍵配置**（`algorithm` 必須放在 `decisions[]` 層級內）:

```yaml
routing:
  decisions:
    - name: general
      description: "General purpose queries"
      priority: 1
      rules:
        operator: AND   # ← catch-all: 空 conditions 匹配所有請求
        conditions: []
      modelRefs:
        - model: gpt-3.5-turbo
          weight: 0.33
        - model: gpt-4
          weight: 0.33
        - model: claude-3
          weight: 0.34
      default_model: gpt-3.5-turbo
      algorithm:          # ← algorithm 在 decision 層級，不是頂層
        type: rl_driven
        rl_driven:
          enable_llm_routing: true
          router_r1_server_url: "http://<HOST>:8123"  # ← 見下方說明
          llm_routing_fallback: thompson
```

### 3️⃣ router_r1_server_url 設定

semantic-router 以 Docker 容器運行，因此 `localhost` 指的是容器本身，無法連到宿主機。

**查詢 Docker bridge gateway IP**:
```bash
docker network inspect vllm-sr-network | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('Gateway:', data[0]['IPAM']['Config'][0]['Gateway'])
"
```

**從容器內驗證連線**:
```bash
docker exec vllm-sr-router-container curl http://<GATEWAY_IP>:8123/health
# {"status": "healthy", "router_type": "KNNRouter", "model_count": 3}
```

**填入配置**:
```yaml
router_r1_server_url: "http://172.21.0.1:8123"  # 替換為實際 gateway IP
```

### 4️⃣ 啟動 semantic-router

```bash
vllm-sr serve --config semantic_router_config_minimal.yaml
```

## 🔄 工作流程詳解

### Decision 匹配

semantic-router 使用 `rules.conditions` 決定是否進入某個 decision：

```yaml
rules:
  operator: AND
  conditions: []   # 空 conditions = catch-all，匹配所有請求
                   # 非空 conditions = 需要 signal 評分達標才匹配
```

> **注意**: `rules: patterns: []` 是錯誤格式，會導致 `decision: (none)`。

### LLMRouter Endpoint 調用

```
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
| `router_r1_server_url` | `http://<gateway>:8123` | LLMRouter endpoint URL（Docker 部署用 gateway IP） |
| `llm_routing_fallback` | `thompson` | LLM 失敗時降級到 Thompson Sampling |
| `exploration_rate` | `0.15` | 初始探索率 (15%) |
| `exploration_decay` | `0.97` | 每 100 次選擇後探索率衰減 |
| `use_thompson_sampling` | `true` | 使用 Thompson Sampling 平衡探索/利用 |
| `cost_awareness` | `true` | 成本感知探索 |
| `cost_weight` | `0.2` | 成本在決策中的權重 |

### Decision 配置格式

```yaml
routing:
  decisions:
    - name: general
      description: "說明"
      priority: 1
      rules:
        operator: AND         # AND 或 OR
        conditions:           # 空陣列 = catch-all
          - type: domain      # signal 類型: domain, keyword, embedding, ...
            name: business    # signal 名稱
      modelRefs:
        - model: gpt-4
          weight: 0.5
        - model: gpt-3.5-turbo
          weight: 0.5
      default_model: gpt-4
      algorithm:
        type: rl_driven
        rl_driven:
          # ... RL 參數
```

## 🔧 定制配置

### 修改 LLMRouter Endpoint URL

```yaml
# decisions[].algorithm.rl_driven 層級下修改
algorithm:
  type: rl_driven
  rl_driven:
    router_r1_server_url: "http://172.21.0.1:8123"  # Docker bridge gateway
```

### 修改降級策略

```yaml
algorithm:
  type: rl_driven
  rl_driven:
    llm_routing_fallback: error    # LLM 失敗時直接返回錯誤
    # 或
    llm_routing_fallback: thompson # LLM 失敗時使用 Thompson Sampling（建議）
```

### 調整成本權重

```yaml
algorithm:
  type: rl_driven
  rl_driven:
    cost_weight: 0.3        # 增加成本因素的影響
    cost_reward_alpha: 0.4  # 40% 成本 + 60% 品質
```

## 🧪 測試

### 1. 驗證 LLMRouter endpoint

```bash
# 從宿主機
curl http://localhost:8123/health
curl http://localhost:8123/models

# 從容器內（使用 gateway IP）
docker exec vllm-sr-router-container curl http://172.21.0.1:8123/health
```

### 2. 驗證 decision 匹配

```bash
vllm-sr eval --prompt "test"
# 應看到: decision: general

vllm-sr eval --prompt "test" --json
# 應看到: "decision_name": "general"
```

### 3. 確認 Router-R1 連線

```bash
vllm-sr logs router 2>&1 | grep "Router-R1"
# 應看到:
# Dependency check: Router-R1 Server at http://172.21.0.1:8123/health — OK
# Router-R1 selected <model> (method=rl_driven, ...)
```

### 4. 測試路由請求

```bash
curl -X POST http://localhost:8123/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Task: code generation\nAvailable models: gpt-4, gpt-3.5-turbo, claude-3"}'
# {"selected_model": "gpt-4"}
```

### 5. 運行集成測試

```bash
python3 -m pytest LLMRouter/test/test_integration_workflow.py -v -s
```

## 📈 監控和調試

### 查看路由日誌

```bash
vllm-sr logs router 2>&1 | grep -E "Router-R1|rl_driven|decision"
```

### 常見問題

#### 問題 1: `decision: (none)`

**原因**: `rules` 格式錯誤，沒有任何 decision 匹配。

**錯誤格式** ❌:
```yaml
rules:
  patterns: []      # 不是合法格式
```
```yaml
categories: [general]  # 需要 classifier 運行才能分類
```

**正確格式** ✅:
```yaml
rules:
  operator: AND
  conditions: []    # 空 conditions = catch-all
```

#### 問題 2: Router-R1 unreachable（Docker 部署）

```
dial tcp [::1]:8123: connect: connection refused
```

**原因**: Docker 容器內 `localhost` / `127.0.0.1` 指向容器本身，不是宿主機。

**解決**:
```bash
# 找到 Docker bridge gateway IP
docker network inspect vllm-sr-network | python3 -c "
import sys, json; data = json.load(sys.stdin)
print(data[0]['IPAM']['Config'][0]['Gateway'])"
# 輸出: 172.21.0.1

# 驗證可連線
docker exec vllm-sr-router-container curl http://172.21.0.1:8123/health

# 填入配置
router_r1_server_url: "http://172.21.0.1:8123"
```

#### 問題 3: LLM 路由超時

```
Router-R1 routing failed: context deadline exceeded
```

**解決**:
- 確認 LLMRouter endpoint 仍在運行
- 降級策略 `llm_routing_fallback: thompson` 會自動接管

#### 問題 4: 模型名稱不匹配

```
selected model not in candidate list
```

**解決**:
```bash
# 查看 LLMRouter 的模型名稱
curl http://localhost:8123/models
# 確認與 modelRefs 中的 model 名稱一致
```

## 🔐 生產部署建議

1. **使用環境變數存儲 API 密鑰**:
   ```yaml
   backend_refs:
     - api_key_env: OPENAI_API_KEY
   ```

2. **啟用持久化狀態**:
   ```yaml
   algorithm:
     type: rl_driven
     rl_driven:
       storage_path: /persistent/state/rl-driven.json
       auto_save_interval: 30s
   ```

3. **使用 debug 日誌排查問題**:
   ```bash
   vllm-sr serve --config semantic_router_config_minimal.yaml --log-level debug
   ```

## 📚 相關資源

- **LLMRouter 文檔**: `LLMRouter/README.md`
- **semantic-router 原始碼**: `../semantic-router/src/`
- **配置範例**: `semantic_router_config_minimal.yaml`
- **集成測試**: `LLMRouter/test/test_integration_workflow.py`

## 🎯 下一步

1. ✅ 啟動 LLMRouter endpoint
2. ✅ 配置 semantic-router（正確的 decision 格式）
3. ✅ 確認 Docker 網路連線（gateway IP）
4. ✅ 驗證 Router-R1 routing 正常運作
5. ⏳ 優化路由決策
6. ⏳ 監控性能指標
7. ⏳ 生產部署
