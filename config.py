from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_FILE = DATA_DIR / "sample_feedback.csv"

INTENT_ORDER = ["物流投诉", "产品质量", "功能建议", "价格问题", "售后服务"]
SENTIMENT_ORDER = ["负面", "中性", "正面"]
VALID_SENTIMENTS = set(SENTIMENT_ORDER)

STOPWORDS = {
    "的", "了", "是", "很", "也", "都", "和", "与", "但", "而", "有", "在",
    "就", "还", "一个", "希望", "可以", "建议", "非常", "太", "一直", "更加",
}

PLOT_COLORS = {
    "负面": "#e34948",
    "中性": "#898781",
    "正面": "#2a78d6",
}

SENTIMENT_NEGATIVE_THRESHOLD = 0.3
SENTIMENT_POSITIVE_THRESHOLD = 0.7
OPENAI_BOUNDARY_LOW = 0.2
OPENAI_BOUNDARY_HIGH = 0.8
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

TEST_SIZE = 0.3
RANDOM_STATE = 42
TOP_K_KEYWORDS = 10
WORDCLOUD_MAX_WORDS = 50
