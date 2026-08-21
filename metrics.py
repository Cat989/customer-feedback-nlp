from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from customer_feedback_nlp.config import INTENT_ORDER


def summarize_data_quality(data: pd.DataFrame) -> Dict[str, float | int | dict]:
    """汇总数据质量指标，便于做数分展示。"""
    stats = dict(data.attrs.get("load_stats", {}))
    text_lengths = data["text"].astype(str).str.len()
    intent_counts = data["intent"].value_counts().reindex(INTENT_ORDER, fill_value=0)

    stats.update(
        {
            "mean_text_length": float(text_lengths.mean()) if not text_lengths.empty else 0.0,
            "median_text_length": float(text_lengths.median()) if not text_lengths.empty else 0.0,
            "min_text_length": int(text_lengths.min()) if not text_lengths.empty else 0,
            "max_text_length": int(text_lengths.max()) if not text_lengths.empty else 0,
            "intent_counts": intent_counts.to_dict(),
        }
    )
    return stats


def build_intent_metrics(y_true: Iterable[str], y_pred: Iterable[str], labels: list[str]) -> Dict[str, object]:
    """计算分类指标和混淆矩阵。"""
    y_true = list(y_true)
    y_pred = list(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def export_misclassified_samples(frame: pd.DataFrame, output_path: Path) -> Path:
    """导出误判样本，帮助做错误分析。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [col for col in ["text", "clean_text", "intent", "predicted_intent"] if col in frame.columns]
    misclassified = frame.loc[frame["intent"] != frame["predicted_intent"], columns].copy()
    misclassified.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_report(report: dict, output_path: Path) -> Path:
    """把摘要报告写成 JSON，便于复盘。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
