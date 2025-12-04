# LLMRouter 測試說明

## 快速開始

### 安裝測試依賴

```bash
pip install -r requirements.txt
```

### 執行測試

```bash
# 執行所有測試
pytest

# 執行單元測試（快速，無外部依賴）
pytest -m unit

# 執行測試並顯示覆蓋率
pytest --cov=LLMRouter --cov-report=term-missing
```

## 測試結構

```
tests/
├── __init__.py          # Package 初始化
├── conftest.py          # Pytest fixtures 和共用設定
├── test_datasets.py     # 資料集核心功能測試 (5 個測試)
├── test_evals.py        # 評估方法核心功能測試 (10 個測試)
└── README.md           # 本檔案
```

**測試哲學**: 這些測試只涵蓋主要公開 API，專注於核心功能而非實作細節，方便未來重構。

## 測試內容

### test_datasets.py - 資料集核心功能 (5 個測試)

1. **TestLoadDataset** (2 個測試)
   - 載入包含回應和評估的完整資料集
   - 載入多個模型的資料

2. **TestAddModelResponse** (2 個測試)
   - 生成模型回應
   - 跳過已存在的回應

3. **TestAddModelResponseEval** (1 個測試)
   - 生成評估結果

### test_evals.py - 評估方法核心功能 (10 個測試)

1. **TestRandomEval** (1 個測試)
   - 隨機評估回傳有效分數

2. **TestSimilarEval** (3 個測試)
   - 相似度評估回傳有效分數
   - 正確擷取 #### 後的答案
   - cosine similarity 基本功能

3. **TestSquadEval** (6 個測試)
   - 答案標準化功能（移除標點、冠詞、空白）
   - 精確匹配計算
   - F1 分數計算
   - 處理 GSM8K 格式答案
   - 精確匹配評估函數
   - 同時返回兩種分數

## 撰寫測試

### 使用 Fixtures

```python
def test_example(temp_dataset_dir, sample_qa_data):
    # temp_dataset_dir 提供暫時的資料目錄
    # sample_qa_data 提供範例測試資料
    assert len(sample_qa_data) > 0
```

### 使用 Mock

```python
from unittest.mock import patch

@patch('LLMRouter.datasets.litellm_router')
def test_with_mock(mock_router):
    # mock_router 會替代實際的 API 呼叫
    mock_router.completion.return_value = {"text": "test"}
```

### 測試標記

```python
@pytest.mark.unit
def test_fast():
    # 快速單元測試
    pass

@pytest.mark.integration
def test_with_api():
    # 整合測試
    pass

@pytest.mark.slow
@pytest.mark.skipif(True, reason="需要下載模型")
def test_slow():
    # 耗時測試
    pass
```

## 常用命令

```bash
# 執行特定檔案的測試
pytest tests/test_datasets.py

# 執行特定測試類別
pytest tests/test_datasets.py::TestLoadDataset

# 執行特定測試函數
pytest tests/test_datasets.py::TestLoadDataset::test_load_dataset_basic

# 顯示詳細輸出
pytest -v

# 顯示 print 輸出
pytest -s

# 在第一個失敗時停止
pytest -x

# 執行失敗的測試
pytest --lf

# 平行執行測試（需要 pytest-xdist）
pytest -n auto
```

## 測試覆蓋率

產生 HTML 報告:
```bash
pytest --cov=LLMRouter --cov-report=html
# 在瀏覽器中開啟 htmlcov/index.html
```

產生終端報告:
```bash
pytest --cov=LLMRouter --cov-report=term-missing
```

## 注意事項

1. **不要實際呼叫 API**: 所有測試都應該使用 mock，避免實際的 API 呼叫
2. **使用暫時目錄**: 使用 `temp_dataset_dir` fixture，不要修改實際資料
3. **獨立測試**: 每個測試應該獨立，不依賴其他測試的執行順序
4. **清理資源**: pytest 會自動清理暫時檔案，但如果手動建立資源記得清理
