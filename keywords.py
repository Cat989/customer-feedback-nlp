from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

from customer_feedback_nlp.config import TOP_K_KEYWORDS


def extract_keywords(processed_texts: Iterable[str], top_k: int = TOP_K_KEYWORDS) -> List[Tuple[str, int]]:
    """统计清洗后词语，返回可解释的 Top-N 关键词。"""
    counter = Counter(
        token
        for text in processed_texts
        for token in text.split()
        if token
    )
    return counter.most_common(top_k)
