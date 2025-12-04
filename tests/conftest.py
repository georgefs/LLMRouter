import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(tmp_path_factory):
    """自動設定測試環境變數"""
    # 建立暫時的設定檔
    tmp_dir = tmp_path_factory.mktemp("config")
    config_path = tmp_dir / "test_config.yaml"
    dataset_dir = tmp_path_factory.mktemp("datasets")

    # 建立測試用的設定檔
    config_content = f"""model_list:
  - model_name: test-model
    litellm_params:
      model: openai/gpt-3.5-turbo
      api_base: https://test.api.com
      api_key: test-key
dataset_path: {dataset_dir}
"""
    config_path.write_text(config_content)

    # 設定環境變數
    os.environ['LLMROUTER_CONFIG'] = str(config_path)
    os.environ['OPENROUTER_API_BASEURL'] = 'https://test.api.com'
    os.environ['OPENROUTER_API_KEY'] = 'test-key'
    os.environ['NCHC_API_BASEURL'] = 'https://test.nchc.com'
    os.environ['NCHC_API_KEY'] = 'test-nchc-key'

    yield

    # 清理（可選）
    for key in ['LLMROUTER_CONFIG', 'OPENROUTER_API_BASEURL', 'OPENROUTER_API_KEY',
                'NCHC_API_BASEURL', 'NCHC_API_KEY']:
        os.environ.pop(key, None)


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """建立暫時的資料集目錄"""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    (dataset_dir / "datasets").mkdir()
    (dataset_dir / "responses").mkdir()
    (dataset_dir / "evals").mkdir()
    return dataset_dir


@pytest.fixture
def sample_qa_data():
    """範例問答資料"""
    return [
        {
            "key": "test_0",
            "question": "What is 2 + 2?",
            "answer": "The answer is 4.\n#### 4"
        },
        {
            "key": "test_1",
            "question": "What is 5 * 3?",
            "answer": "The answer is 15.\n#### 15"
        },
        {
            "key": "test_2",
            "question": "What is 10 - 3?",
            "answer": "The answer is 7.\n#### 7"
        }
    ]


@pytest.fixture
def sample_response_data():
    """範例模型回應資料"""
    return {
        "key": "test_0",
        "text": "The answer is 4.",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15
        },
        "response_time": 1.5
    }


@pytest.fixture
def mock_litellm_completion():
    """Mock LiteLLM completion response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "The answer is 4."
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.json.return_value = '{"response": "mock"}'
    return mock_response


@pytest.fixture
def create_test_dataset(temp_dataset_dir):
    """建立測試資料集檔案的輔助函數"""
    def _create(dataset_name, data):
        dataset_path = temp_dataset_dir / "datasets" / f"{dataset_name}.jsonl"
        with open(dataset_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return dataset_path
    return _create


@pytest.fixture
def create_test_response(temp_dataset_dir):
    """建立測試回應檔案的輔助函數"""
    def _create(dataset_name, model_name, data):
        import urllib.parse
        model_filename = urllib.parse.quote_plus(model_name)
        response_dir = temp_dataset_dir / "responses" / dataset_name
        response_dir.mkdir(parents=True, exist_ok=True)
        response_path = response_dir / f"{model_filename}.jsonl"
        with open(response_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return response_path
    return _create


@pytest.fixture
def create_test_eval(temp_dataset_dir):
    """建立測試評估檔案的輔助函數"""
    def _create(dataset_name, model_name, eval_method, data):
        import urllib.parse
        model_filename = urllib.parse.quote_plus(model_name)
        eval_dir = temp_dataset_dir / "evals" / dataset_name / eval_method
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_path = eval_dir / f"{model_filename}.jsonl"
        with open(eval_path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return eval_path
    return _create


@pytest.fixture
def mock_config(temp_dataset_dir, monkeypatch):
    """Mock 設定檔"""
    config = {
        'model_list': [
            {
                'model_name': 'test-model',
                'litellm_params': {
                    'model': 'openai/gpt-3.5-turbo',
                    'api_base': 'https://test.api.com',
                    'api_key': 'test-key'
                }
            }
        ],
        'dataset_path': str(temp_dataset_dir)
    }

    # Mock 環境變數
    monkeypatch.setenv('LLMROUTER_CONFIG', '/fake/config.yaml')

    return config


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer 模型"""
    mock_model = Mock()
    mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return mock_model
