from __future__ import annotations

import json
import os
import re
from typing import Tuple

from snownlp import SnowNLP

from customer_feedback_nlp.config import (
    OPENAI_BOUNDARY_HIGH,
    OPENAI_BOUNDARY_LOW,
    OPENAI_MODEL,
    SENTIMENT_NEGATIVE_THRESHOLD,
    SENTIMENT_POSITIVE_THRESHOLD,
    VALID_SENTIMENTS,
)

try:
    from openai import OpenAI
except ImportError:  # OpenAI 增强是可选功能，SnowNLP 仍可独立运行
    OpenAI = None


def build_openai_client():
    """根据环境变量创建 OpenAI 客户端；未配置时返回 None。"""
    if OpenAI is None:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAI(api_key=api_key)
    except Exception as exc:
        print(f"提示：OpenAI 客户端初始化失败，将仅使用 SnowNLP：{exc}")
        return None


def mask_sensitive_text(text: str) -> str:
    """在发送给外部模型前，替换常见手机号、订单号和邮箱。"""
    masked = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号]", str(text))
    masked = re.sub(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", "[邮箱]", masked)
    masked = re.sub(r"(?<!\d)\d{8,}(?!\d)", "[编号]", masked)
    return masked


def classify_by_score(score: float) -> str:
    """按项目阈值把 SnowNLP 分数映射为情感标签。"""
    if score < SENTIMENT_NEGATIVE_THRESHOLD:
        return "负面"
    if score > SENTIMENT_POSITIVE_THRESHOLD:
        return "正面"
    return "中性"


def call_openai_sentiment(client, text: str) -> Tuple[str, str] | None:
    """调用 GPT-4 并校验结构化情感结果；失败时返回 None。"""
    if client is None:
        return None

    prompt = (
        "请判断下面中文客户反馈的情感倾向。只返回一个 JSON 对象，不要 Markdown。"
        '格式必须是 {"sentiment":"负面|中性|正面","reason":"不超过30字的简短理由"}。'
        f"\n客户反馈：{mask_sensitive_text(text)}"
    )

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "你是中文客户反馈情感分类器，只能输出合法 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        label = result.get("sentiment")
        reason = result.get("reason")
        if label not in VALID_SENTIMENTS or not isinstance(reason, str) or not reason.strip():
            return None
        return label, reason.strip()[:100]
    except Exception as exc:
        print(f"提示：GPT-4 判断失败，回退到 SnowNLP：{exc}")
        return None


def analyze_sentiment(text: str, client=None) -> Tuple[float, str, str, str]:
    """先保留 SnowNLP 原始分数，再对边界样本可选调用 GPT-4。"""
    score = float(SnowNLP(text).sentiments)
    label = classify_by_score(score)
    source = "SnowNLP"
    reason = f"SnowNLP 原始分数 {score:.3f}，按阈值判定为{label}。"

    if OPENAI_BOUNDARY_LOW <= score <= OPENAI_BOUNDARY_HIGH:
        enhanced = call_openai_sentiment(client, text)
        if enhanced is not None:
            label, reason = enhanced
            source = "OpenAI"

    return score, label, source, reason
