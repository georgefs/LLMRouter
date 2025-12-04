#!/bin/bash
# LLMRouter 測試執行腳本

set -e

echo "========================================="
echo "LLMRouter 測試套件"
echo "========================================="
echo ""

# 檢查 pytest 是否安裝
if ! python3 -m pytest --version > /dev/null 2>&1; then
    echo "錯誤: pytest 未安裝"
    echo "請執行: pip install -r requirements.txt"
    exit 1
fi

# 預設執行所有單元測試
TEST_TYPE=${1:-"unit"}

case $TEST_TYPE in
    "all")
        echo "執行所有測試..."
        python3 -m pytest tests/ -v
        ;;
    "unit")
        echo "執行單元測試..."
        python3 -m pytest tests/ -m unit -v
        ;;
    "integration")
        echo "執行整合測試..."
        python3 -m pytest tests/ -m integration -v
        ;;
    "coverage")
        echo "執行測試並產生覆蓋率報告..."
        python3 -m pytest tests/ --cov=LLMRouter --cov-report=term-missing --cov-report=html
        echo ""
        echo "HTML 報告已產生: htmlcov/index.html"
        ;;
    "quick")
        echo "快速測試（跳過耗時測試）..."
        python3 -m pytest tests/ -m "unit and not slow" -v
        ;;
    *)
        echo "用法: $0 [all|unit|integration|coverage|quick]"
        echo ""
        echo "選項:"
        echo "  all         - 執行所有測試"
        echo "  unit        - 僅執行單元測試（預設）"
        echo "  integration - 僅執行整合測試"
        echo "  coverage    - 執行測試並產生覆蓋率報告"
        echo "  quick       - 快速測試，跳過耗時測試"
        exit 1
        ;;
esac
