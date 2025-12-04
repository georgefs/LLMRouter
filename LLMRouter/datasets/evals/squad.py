"""
SQuAD 風格的評估方法
基於 SQuAD 2.0 官方評估腳本，提供 Exact Match 和 F1 分數評估

參考: https://rajpurkar.github.io/SQuAD-explorer/
"""
import re
import string
import collections


def normalize_answer(s):
    """標準化答案文字

    處理步驟:
    1. 轉換為小寫
    2. 移除標點符號
    3. 移除冠詞 (a, an, the)
    4. 標準化空白字元

    Args:
        s: 原始答案文字

    Returns:
        標準化後的答案文字
    """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """將答案文字轉換為 token 列表

    Args:
        s: 答案文字

    Returns:
        Token 列表
    """
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    """計算精確匹配分數

    兩個答案標準化後完全相同則為 1，否則為 0

    Args:
        a_gold: 標準答案
        a_pred: 預測答案

    Returns:
        1 或 0
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """計算 F1 分數

    基於 token overlap 計算 F1 分數

    Args:
        a_gold: 標準答案
        a_pred: 預測答案

    Returns:
        F1 分數 (0.0 到 1.0)
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # 如果任一為空答案，則相同時 F1 為 1，否則為 0
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def eval(dataset, response):
    """評估函數 - 使用 F1 分數

    從 dataset 中提取標準答案（支援 GSM8K 格式的 #### 分隔符）
    與 response 的文字進行比對

    Args:
        dataset: 包含 'answer' 欄位的資料集記錄
        response: 包含 'text' 欄位的模型回應

    Returns:
        F1 分數 (0.0 到 1.0)
    """
    # 提取標準答案
    answer = dataset.get('answer', '')

    # 支援 GSM8K 格式: 答案在 #### 之後
    if '####' in answer:
        answer = answer.split('####')[-1].strip()

    # 提取預測答案
    prediction = response.get('text', '')

    # 計算 F1 分數
    return compute_f1(answer, prediction)


def eval_exact(dataset, response):
    """評估函數 - 使用精確匹配

    從 dataset 中提取標準答案（支援 GSM8K 格式的 #### 分隔符）
    與 response 的文字進行比對

    Args:
        dataset: 包含 'answer' 欄位的資料集記錄
        response: 包含 'text' 欄位的模型回應

    Returns:
        精確匹配分數 (0 或 1)
    """
    # 提取標準答案
    answer = dataset.get('answer', '')

    # 支援 GSM8K 格式: 答案在 #### 之後
    if '####' in answer:
        answer = answer.split('####')[-1].strip()

    # 提取預測答案
    prediction = response.get('text', '')

    # 計算精確匹配
    return compute_exact(answer, prediction)


def eval_both(dataset, response):
    """評估函數 - 同時返回 F1 和精確匹配

    Args:
        dataset: 包含 'answer' 欄位的資料集記錄
        response: 包含 'text' 欄位的模型回應

    Returns:
        dict: {'f1': F1分數, 'exact': 精確匹配分數}
    """
    # 提取標準答案
    answer = dataset.get('answer', '')

    # 支援 GSM8K 格式: 答案在 #### 之後
    if '####' in answer:
        answer = answer.split('####')[-1].strip()

    # 提取預測答案
    prediction = response.get('text', '')

    return {
        'f1': compute_f1(answer, prediction),
        'exact': compute_exact(answer, prediction)
    }
