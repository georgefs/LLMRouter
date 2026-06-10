# LLMRouter × semantic-router 架構設計

## 📐 系統架構

### 整體架構

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Request                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │      semantic-router           │
        │      (Port 8899)               │
        ├────────────────────────────────┤
        │ • Intent Classification        │
        │ • PII Detection                │
        │ • Model Selection (RL-driven)  │
        └────────────┬───────────────────┘
                     │
                     │ enable_llm_routing=true
                     │
        ┌────────────▼──────────────────────────────────────────┐
        │      semantic-router RLDrivenSelector                 │
        ├───────────────────────────────────────────────────────┤
        │  1. selectWithRouterR1()                              │
        │     └─ RouterR1Client.Route(query)                    │
        │        └─ HTTP POST /route                            │
        │           └─ router_r1_server_url                     │
        │              = http://localhost:8888                  │
        └────────────┬──────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           LLMRouter Endpoint                           │
        │           (Port 8888)                                  │
        ├────────────────────────────────────────────────────────┤
        │  HTTP Server:                                          │
        │  • POST /route       ← Main routing endpoint           │
        │  • GET  /health      ← Health check                    │
        │  • GET  /models      ← Model list                      │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           LLMRouter Core                               │
        ├────────────────────────────────────────────────────────┤
        │  1. Load pre-trained router (pkl)                      │
        │  2. Compute embedding (sentence-transformers)          │
        │  3. KNNRouter.predict_indices()                        │
        │  4. Map indices to model_names                         │
        │  5. Return selected_model                              │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           HTTP Response                                │
        │           {"selected_model": "gpt-4"}                  │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │      semantic-router (返回)                             │
        │      使用選中的 selected_model                          │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           Select LLM & Call API                        │
        │           (gpt-4, gpt-3.5-turbo, or claude-3)          │
        └────────────┬─────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────────────────────────────────┐
        │           Return Response to User                      │
        └────────────────────────────────────────────────────────┘
```

---

## 🔄 數據流

### Request Flow

```
Step 1: User Query
────────────────────
Input: Natural language query
Example: "請幫我生成一個 Python 函數來計算斐波那契數列"

Step 2: semantic-router Intent Classification
────────────────────────────────────────────────
Intent Detector: "code_generation"
Category: "coding"
Models Available: [gpt-4, gpt-3.5-turbo, claude-3]

Step 3: RL-driven Selector Decision
──────────────────────────────────
if enable_llm_routing:
    Call LLMRouter endpoint
else:
    Use Thompson Sampling

Step 4: LLMRouter Endpoint Request
──────────────────────────────────
POST /route
{
  "query": "Task: code_generation
           Available models: gpt-4, gpt-3.5-turbo, claude-3"
}

Step 5: LLMRouter Processing
─────────────────────────────
1. Query → Embedding (384-dim)
2. KNN Search (k=5)
3. Select Model: argmax(scores)
4. Map Index → "gpt-4"

Step 6: Response
────────────────
HTTP 200
{
  "selected_model": "gpt-4"
}

Step 7: semantic-router Routes
──────────────────────────────
Call: gpt-4 API
Method: OpenAI GPT-4
Prompt: "請幫我生成一個 Python 函數..."

Step 8: Final Response
──────────────────────
Return: Generated Python function to user
```

### Error Flow

```
Normal Flow:
  LLMRouter Success
        ↓
  Return selected_model

Error Flow 1: LLMRouter Timeout/Unavailable
  ↓
  llm_routing_fallback = "thompson"
  ↓
  Use Thompson Sampling
  ↓
  Return selected_model

Error Flow 2: Invalid Response
  ↓
  Model not in candidates
  ↓
  Fallback to Thompson Sampling
  ↓
  Return selected_model

Error Flow 3: Network Error
  ↓
  Connection refused
  ↓
  Immediate fallback
  ↓
  Return selected_model
```

---

## 📦 組件設計

### 1. BaseRouter（抽象基類）

```python
class BaseRouter(ABC):
    """所有 router 的抽象基類。"""

    def __init__(self):
        self.model_names: List[str] | None = None

    def fit(self, data: RouterData):
        # 自動綁定 model_names；重複 fit 時做一致性檢查
        if self.model_names is None:
            self.model_names = data.models.copy()
        elif self.model_names != data.models:
            raise ValueError("Model mismatch")
        self._fit(data)

    @abstractmethod
    def _fit(self, data: RouterData) -> None: ...

    @abstractmethod
    def predict_probs(self, prompts: List[str]) -> np.ndarray:
        """回傳 (N, M) 機率矩陣。"""

    @abstractmethod
    def save(self, path: "str | Path") -> None:
        """序列化到 .pkl 或目錄。checkpoint 需包含 'router_type' 鍵。"""

    @classmethod
    @abstractmethod
    def load(cls, path_or_ck: "str | Path | dict") -> "BaseRouter":
        """從路徑或已載入的 checkpoint dict 還原。"""
```

#### 自描述 Checkpoint（Self-Describing Checkpoints）

所有 router 的 `save()` 在 checkpoint dict 中寫入 `"router_type"` 鍵：

```python
# KNNRouter.save() 內部
checkpoint = {
    "router_type": "knn",   # ← 讓 server 無需猜測類型
    "model_names": self.model_names,
    ...
}
```

`LLMRouterEndpointServer` 只載入一次 pickle，直接從 dict 傳給 `cls.load(ck)`（Single-Load Pattern），省去舊版的二次讀取：

```python
def _load_router(self):
    with open(self.router_path, "rb") as f:
        ck = pickle.load(f)
    cls = self._detect_type_checkpoint(ck)   # 從 router_type 鍵查 registry
    self.router = cls.load(ck)               # 不再重讀檔案
```

---

### 2. Router 類型一覽

| 類型 | 模組 | 訓練方式 | 說明 |
|------|------|----------|------|
| `oracle` | `oracle.py` | 無（直接使用 GT） | 上界基準 |
| `random` | `oracle.py` | 無 | 下界基準 |
| `knn` | `knn.py` | embedding + KNN | K 近鄰平均分數 |
| `mf` | `mf.py` | SGD | Matrix Factorization |
| `sw` | `sw_ranking.py` | embedding + win stats | Similarity-Weighted Ranking |
| `roberta` | `roberta_mlc.py` | fine-tuning | RoBERTa 多標籤迴歸 |
| `grpo` | `grpo.py` | RL（PPO-clip） | Group Relative Policy Optimization |
| `semantic_api` | `semantic_api.py` | 連線驗證（無本地訓練） | 呼叫 semantic-router HTTP API |

#### SemanticAPIRouter

將 semantic-router 的 `POST /api/v1/classify/intent`（Port 8080）包裝成標準 `BaseRouter`，使其可直接參與 benchmark：

```
User Prompt
    ↓
SemanticAPIRouter.predict_probs()
    ↓  HTTP POST /api/v1/classify/intent
semantic-router API（Port 8080）
    ↓  {"recommended_model": "gpt-4"}
one-hot 機率向量  [0, 1, 0]
    ↓
evaluate() → HR / Cost / TER / NBS
```

- `_fit()` 只驗證連線可達性；模型選擇邏輯由 semantic-router 管理
- API 回傳未知模型名稱時，fallback 為均勻分布
- `save()` 只儲存 URL / timeout / model_names，不含 local model weights

---

### 3. LLMRouterEndpointServer

```python
class LLMRouterEndpointServer:
    def __init__(self, router_path, port=8888, host="0.0.0.0"):
        self._load_router()     # 單次載入，從 router_type 鍵決定 cls

    def _load_router(self):
        if self.router_path.is_dir():
            cls = self._detect_type_dir(self.router_path)  # HuggingFace 格式
            self.router = cls.load(self.router_path)
        else:
            with open(self.router_path, "rb") as f:
                ck = pickle.load(f)                        # 只載入一次
            cls = self._detect_type_checkpoint(ck)
            self.router = cls.load(ck)                     # 傳 dict，不重讀

    def shutdown(self):
        if self.server:
            self.server.shutdown()     # 停止 serve_forever()
            self.server.server_close() # 釋放 socket fd
```

**HTTP API**：

| 方法 | 路徑 | 說明 |
|------|------|------|
| `POST` | `/route` | `{"query": "..."} → {"selected_model": "gpt-4"}` |
| `GET` | `/health` | `{"status": "healthy", "router_type": "KNNRouter", "model_count": N}` |
| `GET` | `/models` | `{"models": [{"name": "gpt-4"}, ...]}` |

---

### 4. DatasetAnalyzer（四維根因框架）

```python
from LLMRouter.router.dataset_eval import analyze, format_report

result = analyze(data)        # RouterData → DatasetAnalysisResult
print(format_report(result))  # 人類可讀報告 + 閾值判斷 + 修復建議
```

**四個指標**（Technical Report §7）：

| 優先 | 指標 | 公式 | 閾值 |
|------|------|------|------|
| P1 | CH Score | Calinski-Harabasz Index（sklearn） | > 2.0 |
| P2 | Avg_Sim | `1/(N*(N-1)) * Σ cos(v_i, v_j)` | > 0.025 |
| P3 | Dec_Var σ² | `1/M * Σ(win_rate_k - mean)²` | > 0.015 |
| — | N | `len(train_prompt)` | ≥ 3,000 |

評級邏輯：CH fail → **POOR**；Sim / Var fail → **MARGINAL**；全 pass → **GOOD**。

---

## 🏗️ 設計決策

### 決策 1: Model Names 自動綁定

**決策**: 自動綁定到 fit() 的 RouterData

**理由**:
- 避免人為錯誤
- 保證 train/serve 一致性
- 簡化 API

**替代方案考慮**:
- ❌ 手動指定 (容易出錯)
- ❌ 運行時推斷 (不確定)
- ✅ 自動綁定 (確定且一致)

### 決策 2: HTTP vs gRPC

**決策**: HTTP/JSON (RESTful)

**理由**:
- 簡單易用
- 跨語言支持
- 易於調試
- semantic-router 期望 HTTP

**替代方案考慮**:
- ❌ gRPC (複雜，但快)
- ✅ HTTP (簡單，足夠快)

### 決策 3: 預訓練 Router vs 動態訓練

**決策**: 預訓練 router (endpoint 無狀態)

**理由**:
- 清晰的責任分離
- Endpoint 無狀態設計
- 易於擴展
- 符合 microservice 架構

**替代方案考慮**:
- ❌ 動態訓練 (stateful, 複雜)
- ✅ 預訓練 (stateless, 簡單)

### 決策 4: Router 類型推斷

**決策**: 自動推斷 (從 checkpoint 特徵)

**理由**:
- 無需配置
- 支持所有 router 類型
- 降低使用難度

**替代方案考慮**:
- ❌ 手動指定 (需要配置)
- ✅ 自動推斷 (零配置)

### 決策 5: 錯誤降級策略

**決策**: 降級到 Thompson Sampling

**理由**:
- 優雅的故障處理
- RL 系統本身可以處理
- 無需特殊邏輯

**替代方案考慮**:
- ❌ 返回錯誤 (user 失敗)
- ✅ 降級處理 (用戶不受影響)

---

## 🔐 安全設計

### 輸入驗證

```python
def validate_query(query: str) -> bool:
    """驗證 query 合法性"""
    
    # 1. 非空
    if not query:
        raise ValueError("Query cannot be empty")
    
    # 2. 長度限制
    if len(query) > 10000:
        raise ValueError("Query too long")
    
    # 3. 字符集
    if not query.isprintable():
        raise ValueError("Invalid characters")
    
    return True
```

### 並發安全

```python
# BaseRouter.predict_indices() 是無狀態的
# 完全線程安全

# KNNRouter
# - _X_train, _Y_train: 只讀
# - _nn (NearestNeighbors): 只讀
# 完全線程安全
```

### API 限制

建議配置 (nginx):
```nginx
location / {
    limit_req zone=api burst=10;    # 限流
    limit_conn addr 5;              # 限制連接
}
```

---

## 📈 性能考慮

### 延遲分解

```
POST /route 總延遲 = 100-200ms

分解:
├─ HTTP 往返: ~10ms
├─ Query embedding: ~50-100ms
│  └─ first-time: ~1500ms (模型加載)
│  └─ 快取: ~50ms (已加載)
├─ KNN 搜索: ~10-20ms
├─ 模型映射: <1ms
└─ JSON 序列化: <1ms
```

### 優化方案

1. **Embedding 快取**
   ```python
   # embedding 模型在進程中保留
   # 第一次請求慢，後續快速
   ```

2. **負載均衡**
   ```
   LB → Endpoint #1 (8888)
      → Endpoint #2 (8889)
      → Endpoint #3 (8890)
   ```

3. **預熱**
   ```python
   # 啟動後執行預測，預熱模型
   router.predict(["warm up query"])
   ```

---

## 🧪 測試架構

全套測試共 235 個，`pytest LLMRouter/test/ -v` 全部通過。

### 共用 Fixtures（conftest.py）

`LLMRouter/test/conftest.py` 提供 session/function 範圍的共用 fixture，
新增測試直接注入，無需重複定義 setUp 邏輯：

| Fixture | Scope | 內容 |
|---|---|---|
| `router_data` | session | RouterData 50筆、384-dim、seed=42 |
| `trained_knn` | function | 已訓練的 KNNRouter（k=5） |
| `saved_router_path` | function | .pkl 暫存路徑，teardown 自動清除 |
| `live_endpoint` | function | 執行中的 endpoint，提供 `.base_url` |

### 單元測試

```
test_model_binding.py       model_names 綁定 / Save-Load 完整工作流
test_router.py              DataPreparer / 各 Router / 評估指標
test_annotator.py           Scorer registry / Annotator / AnnotationRunner
test_manager.py             DatasetManager CRUD
test_eval.py                評估指標函數
test_semantic_api_router.py SemanticAPIRouter HTTP mock / 一熱編碼 / fallback（11 個）
test_dataset_eval.py        DatasetAnalyzer 四維指標數學正確性（22 個）
```

### 行為契約測試（Contract Tests）

```
test_endpoint_behavior.py（24 個）
├─ TestHealthEndpoint       /health 正常回傳 200、status、model_count
├─ TestRouteEndpoint        /route POST 回傳 selected_model、結果一致性
├─ TestProtocolCompliance   Content-Type、必填欄位驗證
├─ TestEdgeCases            短/長/特殊字符 query、畸形 JSON
├─ TestPerformance          P50 < 100ms、P99 < 500ms、10 並發
└─ TestSemanticRouterIntegration  完整 semantic-router 調用流程
```

### 整合測試

```
test_endpoint_server.py       LLMRouterEndpointServer 功能
test_integration_workflow.py  semantic-router RL-driven 端到端（動態埠，避免與 Envoy 衝突）
```

### 動態埠分配

`test_integration_workflow.py` 使用 `_free_port()` 讓系統核心分配可用埠（`socket.bind(("127.0.0.1", 0))`），避免與 semantic-router Envoy proxy（固定 Port 8899）衝突。測試執行期間埠號透過 `server._test_base_url` 傳遞給各 helper method。

---

## 📚 擴展點

### 設計原則：Registry 自登記模式

Router 和 Annotator 都採用「模組自登記」設計：
在各自的實作檔案末尾呼叫 `register()`，
**無需修改 `__main__.py` 或任何其他檔案**，CLI 即自動感知。

```
router/registry.py         全域 router registry（name → cls, kwargs_fn）
annotator/registry.py      全域 annotator registry（strategy → factory_fn）
router/_template.py        新 router 複製起點
annotator/_template.py     新 annotator 複製起點
```

### 1. 新增 Router 類型

複製 `router/_template.py`，實作三個方法後在末尾登記：

```python
# my_router.py 末尾
from .registry import register

register("my_router", MyRouter, lambda a: {"param": a.param})
```

CLI 立即可用，無需其他修改：

```bash
python3 -m LLMRouter router train my_router --data data.npz
python3 -m LLMRouter router eval  my_router --data data.npz --model r.pkl
```

### 2. 新增 Annotator

複製 `annotator/_template.py`，實作 `annotate()` 後登記：

```python
# my_annotator.py 末尾
from .registry import register

register("my_strategy", lambda args, config: MyAnnotator())
```

### 3. 新增 HTTP 端點

```python
def do_GET(self):
    if self.path == "/debug":
        # 添加調試端點
        response = {
            "router_type": self.router.__class__.__name__,
            "model_names": self.router.model_names,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        }
        self._send_json(200, response)
```

### 3. 自定義降級策略

```python
if enable_llm_routing and routerR1Client:
    try:
        result = selectWithRouterR1(...)
        return result
    except Exception as e:
        if llm_routing_fallback == "custom":
            # 自定義降級邏輯
            return customFallback()
        else:
            return thompsonSampling()
```

---

## 🚀 部署架構

### 開發環境

```
Developer Machine
├─ LLMRouter Endpoint (localhost:8888)
└─ semantic-router (localhost:8899)
```

### 生產環境

```
┌─────────────────────────────────────────┐
│      Load Balancer (nginx)              │
│      (semantic-router.company.com)      │
└─────────────────────────────────────────┘
                    ↓
     ┌──────────────┼──────────────┐
     ↓              ↓              ↓
  Endpoint #1   Endpoint #2   Endpoint #3
  (8888)        (8888)        (8888)
  Pod A         Pod B         Pod C
  
  + Monitoring (Prometheus)
  + Logging (ELK)
  + Alerting (PagerDuty)
  + State Backup
```

---

## 🎯 性能指標

基於實測:

| 指標 | 數值 | 備註 |
|------|------|------|
| Endpoint 啟動 | < 1s | 不含模型加載 |
| 首個請求 | ~1.5s | embedding 模型初始化 |
| 後續請求 | 100-200ms | embedding 快取 |
| 並行請求 | 10+ | 無 blocking |
| 成功率 | 99.9% | 包括降級處理 |
| 記憶體佔用 | ~500MB | KNN + embedding |

---

**✅ 架構設計完成，可投入生產。**
