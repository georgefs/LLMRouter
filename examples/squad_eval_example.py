#!/usr/bin/env python3
"""
SQuAD 評估方法使用範例

展示如何使用新的 SQuAD 風格評估方法
"""
import sys
import os

# 將專案根目錄加入路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LLMRouter.datasets.evals import squad


def example_basic_usage():
    """基本使用範例"""
    print("=" * 60)
    print("範例 1: 基本 F1 評估")
    print("=" * 60)

    # 準備資料
    dataset = {
        "question": "What is the capital of France?",
        "answer": "The capital of France is Paris.\n#### Paris"
    }

    response = {
        "text": "Paris"
    }

    # 使用 F1 評估（預設）
    f1_score = squad.eval(dataset, response)
    print(f"問題: {dataset['question']}")
    print(f"標準答案: Paris")
    print(f"模型回應: {response['text']}")
    print(f"F1 分數: {f1_score:.4f}")
    print()


def example_exact_match():
    """精確匹配範例"""
    print("=" * 60)
    print("範例 2: 精確匹配評估")
    print("=" * 60)

    dataset = {"answer": "#### 42"}

    # 測試完全匹配
    response1 = {"text": "42"}
    score1 = squad.eval_exact(dataset, response1)
    print(f"標準答案: 42")
    print(f"回應 '42': 精確匹配分數 = {score1}")

    # 測試標點符號差異（標準化後仍匹配）
    response2 = {"text": "42!"}
    score2 = squad.eval_exact(dataset, response2)
    print(f"回應 '42!': 精確匹配分數 = {score2}")

    # 測試完全不同
    response3 = {"text": "99"}
    score3 = squad.eval_exact(dataset, response3)
    print(f"回應 '99': 精確匹配分數 = {score3}")
    print()


def example_both_scores():
    """同時獲取兩種分數"""
    print("=" * 60)
    print("範例 3: 同時獲取 F1 和精確匹配分數")
    print("=" * 60)

    dataset = {
        "answer": "The quick brown fox jumps over the lazy dog.\n#### quick brown fox"
    }

    # 完全匹配
    response1 = {"text": "quick brown fox"}
    scores1 = squad.eval_both(dataset, response1)
    print(f"標準答案: 'quick brown fox'")
    print(f"回應: '{response1['text']}'")
    print(f"  F1 分數: {scores1['f1']:.4f}")
    print(f"  精確匹配: {scores1['exact']}")
    print()

    # 部分匹配
    response2 = {"text": "quick brown dog"}
    scores2 = squad.eval_both(dataset, response2)
    print(f"回應: '{response2['text']}'")
    print(f"  F1 分數: {scores2['f1']:.4f}")
    print(f"  精確匹配: {scores2['exact']}")
    print()


def example_normalization():
    """答案標準化範例"""
    print("=" * 60)
    print("範例 4: 答案標準化展示")
    print("=" * 60)

    test_cases = [
        "The Quick Brown Fox!",
        "  multiple   spaces  ",
        "A cat and an elephant",
        "What's the answer?",
    ]

    for text in test_cases:
        normalized = squad.normalize_answer(text)
        print(f"原始: '{text}'")
        print(f"標準化: '{normalized}'")
        print()


def example_with_dataset():
    """模擬完整的資料集評估流程"""
    print("=" * 60)
    print("範例 5: 批量評估資料集")
    print("=" * 60)

    # 模擬資料集
    dataset_samples = [
        {
            "question": "What is 2 + 2?",
            "answer": "#### 4",
            "response": "4"
        },
        {
            "question": "What is the capital of Japan?",
            "answer": "#### Tokyo",
            "response": "tokyo"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "#### William Shakespeare",
            "response": "Shakespeare"
        }
    ]

    print("評估結果:")
    print("-" * 60)

    total_f1 = 0
    total_exact = 0

    for i, sample in enumerate(dataset_samples, 1):
        dataset = {"answer": sample["answer"]}
        response = {"text": sample["response"]}

        scores = squad.eval_both(dataset, response)
        total_f1 += scores['f1']
        total_exact += scores['exact']

        print(f"{i}. {sample['question']}")
        print(f"   回應: {sample['response']}")
        print(f"   F1: {scores['f1']:.4f} | 精確匹配: {scores['exact']}")

    print("-" * 60)
    print(f"平均 F1 分數: {total_f1 / len(dataset_samples):.4f}")
    print(f"精確匹配準確率: {total_exact / len(dataset_samples):.4f}")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_exact_match()
    example_both_scores()
    example_normalization()
    example_with_dataset()

    print("=" * 60)
    print("✓ 所有範例執行完成")
    print("=" * 60)
