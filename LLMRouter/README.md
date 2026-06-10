# LLMRouter — Dataset Management & Router Training

## 概覽

LLMRouter 管理三層資料的存取，並提供 router 訓練、評估、縮放實驗的完整流程。

```
<DATA_PATH>/
  datasets/              原始 benchmark（每行 {key, question, answer, ...}）
  responses/
    <dataset>/
      <model>.jsonl      model 對 dataset 的回應（每行 {key, text, usage, ...}）
  annotations/
    <dataset>/
      <strategy>/
        <model>.jsonl    標注結果（每行至少 {key}，其餘欄位由 annotator 決定）
                         llm strategy  → {key, point, verdict, brief_reason, judge, raw, judge_time}
                         cost strategy → {key, cost, input_tokens, output_tokens}
```

---

## 設定

### 環境變數

| 變數 | 說明 |
|---|---|
| `DATA_PATH` | data 根目錄的絕對路徑（含 datasets/、responses/、annotations/） |
| `NCHC_API_BASEURL` / `NCHC_API_KEY` | NCHC 後端 |
| `OPENROUTER_API_BASEURL` / `OPENROUTER_API_KEY` | OpenRouter 後端 |

`DATA_PATH` 透過 `config.yaml` 的 `data_path: !ENV ${DATA_PATH}` 統一讀取。

### 路徑解析優先順序

```
CLI --base 參數  >  config.yaml data_path  >  repo/datasets/（開發 fallback）
```

---

## 端對端流程

```bash
DS=arc_challenge_train
MODEL=Llama-3.1-70B
JUDGE=gpt-oss-120b

# 1. 對 dataset 做 inference，儲存 response
python3 -m LLMRouter response gen $DS $MODEL

# 2. 執行 LLM-as-Judge annotation，儲存評分結果
python3 -m LLMRouter annotation gen $DS $MODEL --strategy llm --judge $JUDGE

# 3. 將三層資料打包為 RouterData（.npz）
python3 -m LLMRouter router prepare \
  --datasets $DS --models $MODEL \
  --strategy llm -o data.npz

# 4. 評估 Oracle / Random 基準線
python3 -m LLMRouter router eval oracle --data data.npz
python3 -m LLMRouter router eval random --data data.npz

# 5. 訓練 KNN router 並評估
python3 -m LLMRouter router train knn --data data.npz -o knn.pkl
python3 -m LLMRouter router eval  knn --data data.npz --model knn.pkl

# 6. （選用）縮放實驗：以不同訓練資料量評估效能
python3 -m LLMRouter router bench knn \
  --data data.npz --fractions 0.1,0.3,0.5,1.0 --repeats 3
```

---

## CLI

```
python3 -m LLMRouter [--config <path>] [--base <path>] <subcommand>
```

### scorer list

```bash
python3 -m LLMRouter scorer list
# → point
# → cost        （若已在程式中 register）
```

### dataset / model / annotation

```bash
python3 -m LLMRouter dataset list
python3 -m LLMRouter response list arc_challenge_train
python3 -m LLMRouter annotation list arc_challenge_train
```

### response gen

```bash
# 對 dataset 做 inference（自動斷點續跑）
python3 -m LLMRouter response gen arc_challenge_train Llama-3.1-70B
python3 -m LLMRouter response gen arc_challenge_train Llama-3.1-70B --concurrency 16
python3 -m LLMRouter response gen arc_challenge_train Llama-3.1-70B --overwrite
```

### extract

```bash
# 抽取訓練資料（stdout）
python3 -m LLMRouter extract \
  --datasets arc_challenge_train,gsm8k_train \
  --models Llama-3.1-70B,gpt-oss-20b \
  --strategy llm

# 輸出到檔案
python3 -m LLMRouter extract ... -o train_data.jsonl

# 不包含 response 文字
python3 -m LLMRouter extract ... --no-response
```

**輸出格式**（每行一筆）：
```json
{"key": "arc_0", "dataset": "arc_challenge_train", "question": "...",
 "answer": "...", "model": "Llama-3.1-70B", "score": 1.0,
 "annotations": {
   "llm":  {"point": 1.0, "verdict": "correct", "brief_reason": "..."},
   "cost": {"cost": 0.0012, "tokens": 300}
 },
 "response": "..."}
```

各 strategy 的資料以 strategy 名稱為 key 獨立保存，不相互混雜。
`score` 預設由 `FieldScorer("point")` 計算（從各 strategy 中搜尋 `point` 欄位）。
CLI 固定使用預設 scorer；自訂評分邏輯請使用 Python API 傳入 scorer。

### annotation gen

對某 dataset / model 的所有回應呼叫 annotator，並儲存結果（支援斷點續跑）。

```bash
# LLM-as-Judge
python3 -m LLMRouter annotation gen arc_challenge_train Llama-3.1-70B \
  --strategy llm --judge gpt-oss-120b

# 調整 concurrency
python3 -m LLMRouter annotation gen arc_challenge_train Llama-3.1-70B \
  --strategy llm --judge gpt-oss-120b --concurrency 16

# 強制重新標注（覆蓋已有資料）
python3 -m LLMRouter annotation gen arc_challenge_train Llama-3.1-70B \
  --strategy llm --judge gpt-oss-120b --overwrite

# 其他 strategy（實作後即可使用，參數由各 strategy 自行定義）
# python3 -m LLMRouter annotation gen arc_challenge_train Llama-3.1-70B \
#   --strategy cost
```

**輸出格式**（llm strategy，每行一筆）：
```json
{"key": "arc_0", "point": 0.9, "verdict": "correct",
 "brief_reason": "...", "judge": "gpt-oss-120b", "raw": "...", "judge_time": 0.83}
```

> `annotation gen` 會自動偵測 dataset item 是否含有 `instruction_id_list` 欄位（IFEval 格式），
> 若有則改用格式驗證 prompt；否則使用標準 GT-based 語意正確性 prompt。

---

### router prepare

將三層資料轉換為 RouterData (.npz)，預設切割 60% train / 10% val / 30% test。

```bash
# 單一 strategy
python3 -m LLMRouter router prepare \
  --datasets arc_challenge_train \
  --models Google-Gemma-3-27B,gpt-oss-20b \
  --strategy llm \
  -o data.npz

# 多個 strategy（以逗號分隔，第一個為主要）
# 各 strategy 的資料以 strategy 名稱為 key 獨立保存
python3 -m LLMRouter router prepare \
  --datasets arc_challenge_train \
  --models Google-Gemma-3-27B,gpt-oss-20b \
  --strategy llm,cost \
  --scorer cost \
  -o data.npz

# 自訂切割比例
python3 -m LLMRouter router prepare ... --train-ratio 0.7 --val-ratio 0.1 -o data.npz

# 預存 embedding（避免 router fit 時重複計算，適合多次 bench）
python3 -m LLMRouter router prepare ... \
  --emb-model mixedbread-ai/mxbai-embed-large-v1 \
  --emb-batch-size 32 \
  -o data_with_emb.npz

# 準備時同步做訓練集預處理（去重 + 鑑別度篩選）
python3 -m LLMRouter router prepare ... --min-var 0.05 --dedup-eps 0.3 -o data_clean.npz
```

### router eval

```bash
# Oracle / Random 基準線（不需要 --model）
python3 -m LLMRouter router eval oracle --data data.npz
python3 -m LLMRouter router eval random --data data.npz

# 訓練後評估
python3 -m LLMRouter router train knn --data data.npz -o knn.pkl --k 10
python3 -m LLMRouter router eval  knn --data data.npz --model knn.pkl
```

**輸出格式**：
```
METRIC_MU   : 0.8123    # 選出模型的平均分數
METRIC_VB   : 0.9241    # mu / oracle（越接近 1 越好）
METRIC_EP   : 1.2340    # 預測分佈的 entropy（bits）
METRIC_TOKEN: 1234.0    # 平均 token 數
METRIC_LAT  : 0.8210    # 平均 latency（秒）
```

### router bench（訓練資料縮放實驗）

固定 test set，以不同 training data 大小評估 router 效能。每個大小重複多個隨機種子並取平均。

```bash
# 以比例指定（--fractions，預設）
python3 -m LLMRouter router bench knn \
  --data data.npz \
  --fractions 0.1,0.2,0.3,0.5,0.7,1.0 \
  --repeats 3

# 以固定筆數指定（--sizes，與 --fractions 互斥）
python3 -m LLMRouter router bench knn \
  --data data.npz \
  --sizes 50,100,200,500 \
  --repeats 3
```

**輸出範例**：
```
Benchmark: knn  |  test=N  |  full_train=N  |  repeats=3
  fraction   n_train        mu        vb        ep       tokens     latency
------------------------------------------------------------------------
       10%        N    0.XXXX    0.XXXX    0.XXXX          0.0       0.000
      100%        N    0.XXXX    0.XXXX    0.XXXX          0.0       0.000
```

### 訓練集預處理參數

`router prepare`、`router train`、`router bench` 都支援以下預處理參數，**只影響訓練集，test / val 不受影響**。

#### `--min-var`：過濾無鑑別度樣本

移除所有模型分數相近（variance ≤ 閾值）的訓練樣本。對 router 而言，若所有模型都答對或都答錯，該題沒有學習價值。

```bash
--min-var 0.0   # 只去除分數完全相同的樣本
--min-var 0.05  # 建議值（二元分數）
--min-var 0.1   # 更嚴格，只保留模型間有明顯差異的樣本
```

不指定則不套用。

#### `--dedup-eps`：DBSCAN 語義去重

對訓練 prompt 計算 embedding 後，以 DBSCAN 找出語義相近的群組，每群只保留一筆。距離以 **cosine distance**（= 1 − cosine similarity）計算，範圍 `[0, 2]`。

```bash
--dedup-eps 0.05  # 嚴格：只去除幾乎字面重複的 prompt
--dedup-eps 0.3   # 建議值：語義相近即視為重複
--dedup-eps 0.5   # 寬鬆：廣泛去除相似 prompt
```

- 孤立的 prompt（無法歸入任何 cluster）全部保留
- 去重用的嵌入模型可透過 `--dedup-emb-model` 指定（預設 `mixedbread-ai/mxbai-embed-large-v1`）
- 不指定 `--dedup-eps` 則不套用

#### 使用範例

```bash
# bench 時即時套用（預處理只做一次，之後每次 subsample 都從乾淨資料來）
python3 -m LLMRouter router bench knn \
  --data data.npz \
  --min-var 0.05 \
  --dedup-eps 0.3 \
  --sizes 50,100,200,500
```

### Router 類型與參數

| 類型 | 說明 | 主要參數 |
|---|---|---|
| `oracle` | 永遠選最佳模型（上界基準） | — |
| `random` | 隨機選模型（下界基準） | — |
| `knn` | K 近鄰平均分數 | `--k` |
| `mf` | Matrix Factorization | `--latent-dim`, `--epochs`, `--lr` |
| `sw` | Similarity-Weighted Ranking | `--k`, `--temperature` |
| `roberta` | RoBERTa 多標籤迴歸 | `--roberta-model`, `--epochs` |
| `grpo` | 強化學習（PPO-clip + Group Relative Advantage） | — |

KNN / MF / SW 共用 `--emb-model`（預設 `mixedbread-ai/mxbai-embed-large-v1`）。

---

## Python API

### Scorer

```python
from LLMRouter import BaseScorer, FieldScorer, register, get, list_scorers

# 查看已有哪些 scorer
list_scorers()   # → ["point"]

# 內建預設
get("point")     # FieldScorer("point") — 從 annotations 取 point 欄位

# 註冊新的 scorer（之後 CLI 就能用 --scorer cost 指定）
register("cost", FieldScorer("cost", strategy="cost"))

# 自訂 scorer：可存取 dataset_item、response、annotations
class CostPerCorrectness(BaseScorer):
    def __call__(self, dataset_item, response, annotations):
        point = annotations.get("llm", {}).get("point", 0.0)
        cost  = annotations.get("cost", {}).get("cost", 1e-9)
        return point / cost

register("efficiency", CostPerCorrectness())

# 傳入 extract() 或 from_manager()
rows = mgr.extract(datasets, models, strategies, scorer=get("efficiency"))
data = DataPreparer().from_manager(mgr, ..., scorer=get("efficiency"))
```

若希望 CLI `--scorer <name>` 能識別自訂 scorer，在 `LLMRouter/scorer/__init__.py` 最後加上 `register(...)` 呼叫即可：

```python
# scorer/__init__.py
register("point",      FieldScorer("point"))
register("efficiency", CostPerCorrectness())  # 新增
```

之後即可：

```bash
python3 -m LLMRouter scorer list          # → efficiency, point
python3 -m LLMRouter router prepare ... --scorer efficiency -o data.npz
```

### DatasetManager

```python
from LLMRouter import DatasetManager

mgr = DatasetManager()                        # 從 config.yaml 讀取 DATA_PATH
mgr = DatasetManager(base_path="/data/path")  # 明確指定

# 查詢
mgr.list_datasets()
mgr.list_models("arc_challenge_train")
mgr.list_annotation_strategies("arc_challenge_train")  # → {"llm": [...]}

# 讀取
items     = mgr.get_dataset("arc_challenge_train")
responses = mgr.get_responses("arc_challenge_train", "Llama-3.1-70B")
annots    = mgr.get_annotations("arc_challenge_train", "Llama-3.1-70B", "llm")

# 寫入（預設合併，已存在的 key 自動跳過）
path, added, skipped = mgr.add_responses("arc_challenge_train", "MyModel", data)
path, added, skipped = mgr.add_annotations("arc_challenge_train", "MyModel", "llm", data)
path, added, skipped = mgr.add_responses(..., overwrite=True)  # 完整覆蓋

# 抽取（join 三層，只保留三層都有交集的 key）
rows = mgr.extract(
    datasets=["arc_challenge_train"],
    models=["Llama-3.1-70B", "gpt-oss-20b"],
    strategies="llm",
)
# 每筆 row: {key, dataset, question, answer, model, score, annotations, response}
# annotations = {"llm": {"point": 1.0, "verdict": "correct", ...}}

# 自訂 scorer（以 cost 欄位作為 score）
from LLMRouter.scorer import FieldScorer
rows = mgr.extract(
    datasets=["arc_challenge_train"],
    models=["Llama-3.1-70B"],
    strategies="cost",
    scorer=FieldScorer("cost"),
)
```

### Annotator

```python
from LLMRouter.annotator import AnnotationRunner, LLMJudgeAnnotator

runner = AnnotationRunner(mgr, concurrency=8)

# LLM-as-Judge（標準 GT-based 或自動偵測 IFEval）
annotator = LLMJudgeAnnotator(router, judge="gpt-oss-120b")
runner.run(annotator, dataset="arc_challenge_train",
           model="Llama-3.1-70B", strategy="llm")

# 斷點續跑（預設行為，自動跳過已完成的 key）
runner.run(annotator, dataset="arc_challenge_train",
           model="Llama-3.1-70B", strategy="llm")

# 強制全部重跑
runner.run(annotator, ..., overwrite=True)

# 非同步版本
import asyncio
results = asyncio.run(runner.arun(annotator, ...))
```

#### 自訂 Annotator

只需繼承 `BaseAnnotator` 並實作 `annotate_one()`：

```python
from LLMRouter.annotator import BaseAnnotator
import asyncio

class CostAnnotator(BaseAnnotator):
    def __init__(self, price_per_token: float):
        self.price = price_per_token

    async def annotate_one(self, key, dataset_item, response, sem):
        async with sem:
            # response 文字已由 runner 傳入
            tokens = len(response.split())   # 示意，實際用 tiktoken 等
            return {
                "key": key,
                "cost": tokens * self.price,
                "tokens": tokens,
            }
```

`annotate_one()` 接收完整 `dataset_item`（原始 JSONL 記錄），可直接讀取任意欄位：

| 參數 | 說明 |
|---|---|
| `key` | 樣本唯一識別碼 |
| `dataset_item` | 完整 dataset 記錄（`question`, `answer`, `instruction_id_list` 等） |
| `response` | model 回應文字 |
| `sem` | concurrency 控制信號量（用 `async with sem:` 包住 IO 操作） |

回傳值需包含 `key`，其餘欄位任意，全部寫入 annotation JSONL。

#### LLMJudgeAnnotator — IFEval 支援

若 dataset item 含有 `instruction_id_list` 欄位（IFEval 格式），`LLMJudgeAnnotator` 自動切換為格式驗證 prompt：

```python
# dataset item 範例（IFEval）
{
  "key": "ifeval_0",
  "question": "Write something without commas.",
  "instruction_id_list": ["punctuation:no_comma"],
  "kwargs": [{}]
}
```

回傳欄位：`point`（0~1）、`verdict`（correct/partially_correct/incorrect）、`brief_reason`、`judge`、`raw`、`judge_time`

### Inferencer

```python
from LLMRouter.inferencer import Inferencer

inf = Inferencer(concurrency=16)
inf.generate("arc_challenge_train", "Llama-3.1-70B")          # 自動斷點續跑
inf.generate("arc_challenge_train", "Llama-3.1-70B", overwrite=True)
```

### Router — 資料準備

```python
from LLMRouter.router import DataPreparer, RouterData

# 從 DatasetManager 直接準備（含 token / latency 資訊）
data = DataPreparer().from_manager(
    mgr,
    datasets=["arc_challenge_train"],
    models=["Google-Gemma-3-27B", "gpt-oss-20b"],
    strategies="llm",
    train_ratio=0.6,   # 預設
    val_ratio=0.1,     # 預設，剩餘 0.3 為 test
)

# 多個 strategy：各自獨立保存，annotations = {"llm": {...}, "cost": {...}}
# 預設 scorer = FieldScorer("point")
data = DataPreparer().from_manager(
    mgr,
    datasets=["arc_challenge_train"],
    models=["ModelA", "ModelB"],
    strategies=["llm", "cost"],
)

# 自訂 scorer：以 cost 作為訓練分數
from LLMRouter.scorer import FieldScorer
data = DataPreparer().from_manager(
    mgr, ..., strategies=["llm", "cost"],
    scorer=FieldScorer("cost", strategy="cost"),
)

# 預存 embedding：router fit / bench 時直接使用，省略重複計算
data = DataPreparer().from_manager(
    mgr, ...,
    emb_model="mixedbread-ai/mxbai-embed-large-v1",
    emb_batch_size=32,
)
# RouterData.train_embed / val_embed / test_embed 會被填入
# KNNRouter / MFRouter / SWRankingRouter 在 fit() 和 evaluate() 時自動使用

data.save("data.npz")
data = RouterData.load("data.npz")  # 之後直接載入，不需重新準備
```

### Router — 訓練集預處理

兩個方法都只動 training set，**val / test 完全不受影響**，可串接使用。

```python
# 過濾無鑑別度樣本
# min_var=0.05：二元分數建議值；min_var=0.0：只去除完全相同的樣本
data = data.filter_by_variance(min_var=0.05)

# DBSCAN 語義去重
# eps 以 cosine distance 計算（0~2），越大越寬鬆
# 建議值：0.3（語義相近即視為重複）
data = data.deduplicate_train(eps=0.3)

# 若已有訓練集 embedding 可直接傳入，省略重新計算
data = data.deduplicate_train(eps=0.3, embeddings=X_train)
```

### Router — 訓練與評估

```python
from LLMRouter.router import KNNRouter, MFRouter, SWRankingRouter, GRPORouter
from LLMRouter.router import OracleRouter, RandomRouter

# 訓練
router = KNNRouter(k=10)
router.fit(data)

# 訓練時指定只用部分訓練資料（不影響 test set）
router.fit(data, train_fraction=0.5, seed=0)

# 評估（固定使用 RouterData.test_*）
metrics = router.evaluate(data)
# → {"mu": 0.82, "vb": 0.93, "ep": 1.23, "avg_tokens": 1234.0, "avg_latency": 0.82}

# 推論
indices = router.predict(["What is photosynthesis?", "Solve x+2=5"])
# → array([1, 0])  每個 prompt 對應的最佳模型 index

# GRPORouter：強化學習版本（PPO-clip + Group Relative Advantage）
# 尚未整合 CLI，只能透過 Python API 使用
router = GRPORouter(
    group_size=8,      # 每個 prompt 採樣的 routing decision 數量
    hidden_dim=256,    # policy network 隱藏層維度
    epochs=30,
    lr=3e-4,
    clip_eps=0.2,      # PPO clip ε
    kl_coef=0.01,      # KL 正則化係數
    entropy_coef=0.01, # entropy bonus（鼓勵探索）
)
router.fit(data)
metrics = router.evaluate(data)
```

### Router — 訓練資料縮放實驗

```python
fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

print(f"{'frac':>6}  {'n_train':>8}  {'mu':>7}  {'vb':>7}")
for frac in fractions:
    mu_list = []
    for seed in range(3):
        router = KNNRouter(k=10)
        router.fit(data, train_fraction=frac, seed=seed)
        mu_list.append(router.evaluate(data)["mu"])
    mu_avg = sum(mu_list) / 3
    n = max(1, round(len(data.train_prompt) * frac))
    print(f"{frac:>6.0%}  {n:>8d}  {mu_avg:>7.4f}")
```

### RouterData.subsample_train

若需要對同一 fraction 取不同子集，也可直接呼叫：

```python
sub = data.subsample_train(fraction=0.5, seed=42)
# sub.train_prompt 是 data.train_prompt 的隨機 50%
# sub.test_prompt / sub.val_prompt 與 data 完全相同
```

---

## 擴充指南

### 新增 Router

1. 複製 `LLMRouter/router/_template.py`，將 `MyRouter` / `my_router` 替換為實際名稱。
2. 實作三個方法：`_fit(data)`、`predict_probs(prompts)`、`save/load`。
3. 在檔案末尾呼叫 `register()`：

```python
from .registry import register

register("my_router", MyRouter, lambda a: {"param_a": a.param_a})
```

完成後不需修改任何其他檔案，CLI 即可辨識：

```bash
python3 -m LLMRouter router train my_router --data data.npz
python3 -m LLMRouter router eval  my_router --data data.npz --model r.pkl
```

若需要新的 CLI 參數，在 `__main__._add_router_args()` 新增 `add_argument()`，並在 `kwargs_fn` 中取用即可。

### 新增 Annotator

1. 複製 `LLMRouter/annotator/_template.py`，替換名稱。
2. 實作 `annotate(prompt, response, dataset_item) → (score, metadata)`。
3. 在檔案末尾呼叫 `register()`：

```python
from .registry import register

register("my_strategy", lambda args, config: MyAnnotator())
```

完成後：

```bash
python3 -m LLMRouter annotation gen <dataset> <model> --strategy my_strategy
```

### Registry 查詢

```python
from LLMRouter.router.registry import list_routers
from LLMRouter.annotator.registry import list_strategies

list_routers()     # ['grpo', 'knn', 'mf', 'oracle', 'random', 'roberta', 'sw']
list_strategies()  # ['llm', 'official']
```

---

## 測試

```bash
# 全部測試（197 個）
pytest LLMRouter/test/ -v

# 只跑特定模組
pytest LLMRouter/test/test_manager.py -v        # DatasetManager
pytest LLMRouter/test/test_router.py -v         # Router 訓練 / 資料處理
pytest LLMRouter/test/test_annotator.py -v      # Annotator / Scorer
pytest LLMRouter/test/test_model_binding.py -v  # model_names 綁定 / save-load
pytest LLMRouter/test/test_endpoint_server.py -v     # HTTP endpoint 功能
pytest LLMRouter/test/test_endpoint_behavior.py -v   # HTTP endpoint 行為契約
pytest LLMRouter/test/test_integration_workflow.py -v # semantic-router 整合流程
pytest LLMRouter/test/test_eval.py -v           # 評估指標函數
```

### 測試 Fixtures（conftest.py）

`LLMRouter/test/conftest.py` 提供四個共用 fixture，新增測試時直接使用：

| Fixture | Scope | 說明 |
|---|---|---|
| `router_data` | session | RouterData 50筆、384-dim 嵌入、seed=42 |
| `trained_knn` | function | 已訓練的 KNNRouter（k=5） |
| `saved_router_path` | function | .pkl 暫存路徑，teardown 自動清除 |
| `live_endpoint` | function | 執行中的 endpoint server，提供 `.base_url` |

```python
# 使用範例
def test_my_router(router_data, trained_knn):
    preds = trained_knn.predict(router_data.test_prompt)
    assert all(m in trained_knn.model_names for m in preds)

def test_my_endpoint(live_endpoint):
    import httpx
    r = httpx.get(f"{live_endpoint.base_url}/health")
    assert r.json()["status"] == "healthy"
```

測試使用 `tmp_path` fixture，不依賴任何外部服務或實際資料。LLM 呼叫透過 `unittest.mock` 模擬。
