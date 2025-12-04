#!/bin/bash
# Docker 建置和測試腳本

set -e

echo "========================================="
echo "LLMRouter Docker 建置腳本"
echo "========================================="
echo ""

# 檢查 Docker 是否安裝
if ! command -v docker &> /dev/null; then
    echo "錯誤: Docker 未安裝"
    echo "請先安裝 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 顯示 Docker 版本
echo "Docker 版本:"
docker --version
echo ""

# 建置映像
echo "建置 Docker 映像..."
docker build -t llmrouter .

echo ""
echo "========================================="
echo "✓ 建置完成！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 設定環境變數:"
echo "   cp .env.example .env"
echo "   # 編輯 .env 檔案，填入你的 API keys"
echo ""
echo "2. 執行容器:"
echo "   docker-compose up -d"
echo ""
echo "3. 進入容器:"
echo "   docker-compose exec llmrouter bash"
echo ""
echo "或直接執行測試:"
echo "   docker run --rm llmrouter pytest tests/ -v"
echo ""
