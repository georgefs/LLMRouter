#!/usr/bin/env bash
# install.sh — LLMRouter 安裝腳本
#
# 用法:
#   ./install.sh [PROFILE]
#
# PROFILE:
#   core  — 核心套件：eval、dataset 管理、HTTP endpoint（預設）
#   ml    — core + PyTorch / Transformers / Sentence-Transformers
#           （KNN / SW-Ranking / MatrixFactorization / RoBERTa router）
#   llm   — core + litellm
#           （LLM inference、LLM-as-Judge annotation）
#   all   — core + ml + llm + dev（完整環境）
#   dev   — all + pytest（開發 / CI 環境）
#
# 環境變數:
#   PYTHON — Python 直譯器路徑（預設 python3）
#
# 範例:
#   ./install.sh           # 安裝 core
#   ./install.sh all       # 安裝全部
#   PYTHON=python3.11 ./install.sh ml

set -euo pipefail

PYTHON="${PYTHON:-python3}"
PROFILE="${1:-core}"

# ── Python 版本檢查 ────────────────────────────────────────────────────────────

PY_VER=$("$PYTHON" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2>/dev/null) || {
    echo "錯誤：找不到 Python 直譯器 '$PYTHON'" >&2
    exit 1
}
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    echo "錯誤：需要 Python >= 3.10（目前 $PY_VER）" >&2
    exit 1
fi
echo "Python $PY_VER ✓"

# ── 安裝 ──────────────────────────────────────────────────────────────────────

case "$PROFILE" in
    core)
        echo "安裝 profile: core（eval、dataset、endpoint）"
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -e .
        ;;
    ml)
        echo "安裝 profile: ml（core + PyTorch + Transformers）"
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -e ".[ml]"
        ;;
    llm)
        echo "安裝 profile: llm（core + litellm）"
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -e ".[llm]"
        ;;
    all)
        echo "安裝 profile: all（core + ml + llm + dev）"
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -e ".[all]"
        ;;
    dev)
        echo "安裝 profile: dev（all + pytest）"
        "$PYTHON" -m pip install --upgrade pip
        "$PYTHON" -m pip install -e ".[dev]"
        ;;
    *)
        echo "未知 profile: $PROFILE" >&2
        echo "用法: $0 [core|ml|llm|all|dev]" >&2
        exit 1
        ;;
esac

# ── 完成 ──────────────────────────────────────────────────────────────────────

echo ""
echo "安裝完成！"
echo ""
echo "快速驗證："
echo "  $PYTHON -m LLMRouter --help"
echo ""
echo "常用命令："
echo "  $PYTHON -m LLMRouter dataset list"
echo "  $PYTHON -m LLMRouter router bench oracle,random \\"
echo "    --datasets <dataset> --models <m1,m2> --strategy llm \\"
echo "    --fractions 1.0 --repeats 1 --show-cost"
