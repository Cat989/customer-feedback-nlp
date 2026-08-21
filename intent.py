from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from customer_feedback_nlp.config import INTENT_ORDER, RANDOM_STATE, TEST_SIZE
from customer_feedback_nlp.metrics import build_intent_metrics


def train_intent_model(
    processed_texts,
    labels,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """训练 TF-IDF + 逻辑回归意图分类器。"""
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=1,
    )
    features = vectorizer.fit_transform(processed_texts)
    classifier = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    classifier.fit(features, list(labels))
    return vectorizer, classifier


def evaluate_intent_model(data: pd.DataFrame) -> Dict[str, object]:
    """划分训练/验证集，输出评估结果和误判样本。"""
    train_df, test_df = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data["intent"],
    )
    vectorizer, classifier = train_intent_model(train_df["clean_text"], train_df["intent"])
    predicted = classifier.predict(vectorizer.transform(test_df["clean_text"]))
    test_eval = test_df.copy()
    test_eval["predicted_intent"] = predicted
    test_eval["is_correct"] = test_eval["intent"] == test_eval["predicted_intent"]
    test_eval["error_reason"] = test_eval.apply(
        lambda row: "" if row["is_correct"] else "模型在当前小样本下未学到足够区分特征",
        axis=1,
    )
    metrics = build_intent_metrics(test_eval["intent"], test_eval["predicted_intent"], INTENT_ORDER)
    return {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "metrics": metrics,
        "test_frame": test_eval,
    }
