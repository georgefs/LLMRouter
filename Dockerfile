# LLMRouter Dockerfile
# 基於 Python 3.10 的輕量級映像

FROM python:3.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴
# - build-essential: 編譯 Python packages 需要
# - git: 可能需要從 git 安裝某些依賴
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝依賴
# 分層複製以利用 Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 設定環境變數
# 這些應該在運行時透過 docker run -e 或 docker-compose 設定
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 建立資料集目錄
RUN mkdir -p /data/datasets

# 設定預設的資料集路徑環境變數
ENV LLMROUTER_CONFIG=/app/config.yaml

# 暴露可能需要的埠（如果之後要加 web service）
# EXPOSE 8000

# 預設命令：顯示幫助訊息
CMD ["python3", "-c", "print('LLMRouter Container\\n\\n可用命令：\\n- python3 -m pytest tests/  # 執行測試\\n- ./cli/add_model_response.py  # 生成模型回應\\n- ./cli/add_model_response_eval.py  # 生成評估\\n- ./cli/prepare_routereval.py  # 準備評估資料\\n\\n請參考 CLAUDE.md 獲取更多資訊')"]
