from __future__ import annotations

import re

import jieba

from .config import STOPWORDS


def clean_text(text: str) -> str:
    """清除噪声、中文分词并过滤停用词。"""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text)
    text = re.sub(r"[@#][\w一-鿿-]+", " ", text)
    text = re.sub(r"[^一-鿿A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = []
    for token in jieba.lcut(text):
        token = token.strip()
        if (
            token
            and token not in STOPWORDS
            and len(token) >= 2
            and not re.fullmatch(r"\d+", token)
        ):
            tokens.append(token)
    return " ".join(tokens)
