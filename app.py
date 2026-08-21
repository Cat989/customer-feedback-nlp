from __future__ import annotations

import sys
from pathlib import Path

# Windows 重定向输出时可能使用 GBK；统一为 UTF-8，避免结果文件乱码。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

CURRENT_DIR = Path(__file__).resolve().parent
PARENT_DIR = CURRENT_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

import pandas as pd

from customer_feedback_nlp.config import OUTPUT_DIR
from customer_feedback_nlp.data_loader import load_data
from customer_feedback_nlp.intent import evaluate_intent_model, train_intent_model
from customer_feedback_nlp.keywords import extract_keywords
from customer_feedback_nlp.metrics import export_misclassified_samples, summarize_data_quality, write_report
from customer_feedback_nlp.preprocess import clean_text
from customer_feedback_nlp.sentiment import analyze_sentiment, build_openai_client
from customer_feedback_nlp.visualization import configure_matplotlib, draw_dashboard, draw_wordcloud, find_chinese_font


def main() -> None:
    """运行客户反馈 NLP 分析流程。"""
    font_path = find_chinese_font()
    configure_matplotlib(font_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    quality_stats = summarize_data_quality(data)
    data["clean_text"] = data["text"].map(clean_text)

    client = build_openai_client()
    if client is None:
        print("提示：未配置可用的 OPENAI_API_KEY 或 openai SDK，边界样本仅使用 SnowNLP。")

    data[[
        "sentiment_score",
        "sentiment",
        "sentiment_source",
        "sentiment_reason",
    ]] = data["text"].apply(lambda value: pd.Series(analyze_sentiment(value, client)))

    eval_result = evaluate_intent_model(data)
    intent_metrics = eval_result["metrics"]

    vectorizer, classifier = train_intent_model(data["clean_text"], data["intent"])
    data["predicted_intent"] = classifier.predict(vectorizer.transform(data["clean_text"]))

    keywords = extract_keywords(data["clean_text"])

    dashboard_path = OUTPUT_DIR / "dashboard.png"
    wordcloud_path = OUTPUT_DIR / "wordcloud.png"
    report_path = OUTPUT_DIR / "analysis_report.json"
    misclassified_path = OUTPUT_DIR / "misclassified_samples.csv"

    draw_dashboard(data, quality_stats, intent_metrics, dashboard_path)
    wordcloud_created = draw_wordcloud(keywords, wordcloud_path, font_path)

    report = {
        "data_quality": quality_stats,
        "intent_metrics": {
            **{key: value for key, value in intent_metrics.items() if key != "confusion_matrix"},
            "confusion_matrix": intent_metrics["confusion_matrix"].tolist(),
        },
        "sentiment_source_counts": data["sentiment_source"].value_counts().to_dict(),
        "top_keywords": keywords,
        "generated_files": {
            "dashboard": str(dashboard_path),
            "wordcloud": str(wordcloud_path),
            "analysis_report": str(report_path),
            "misclassified_samples": str(misclassified_path),
        },
    }
    write_report(report, report_path)
    export_misclassified_samples(eval_result["test_frame"], misclassified_path)

    print("\n=== 客户反馈 NLP 分析结果 ===")
    print(data[[
        "text",
        "clean_text",
        "sentiment_score",
        "sentiment",
        "sentiment_source",
        "sentiment_reason",
        "predicted_intent",
    ]].to_string(index=False))

    print("\n=== 数据质量概览 ===")
    print(
        f"总行数: {quality_stats.get('source_rows', 0)} | "
        f"有效行数: {quality_stats.get('loaded_rows', 0)} | "
        f"空值剔除: {quality_stats.get('dropped_missing_rows', 0)} | "
        f"空文本剔除: {quality_stats.get('dropped_empty_rows', 0)} | "
        f"平均长度: {quality_stats.get('mean_text_length', 0):.1f}"
    )

    print("\n=== 意图评估结果 ===")
    print(
        f"Accuracy={intent_metrics['accuracy']:.3f}, "
        f"Precision={intent_metrics['precision']:.3f}, "
        f"Recall={intent_metrics['recall']:.3f}, "
        f"F1={intent_metrics['f1']:.3f}"
    )

    print("\n=== Top 关键词 ===")
    print("、".join(f"{word}({count})" for word, count in keywords))

    print("\n=== 输出文件 ===")
    print(f"仪表盘：{dashboard_path}")
    print(f"词云：{wordcloud_path if wordcloud_created else '未生成（缺少中文字体或 wordcloud）'}")
    print(f"评估报告：{report_path}")
    print(f"误判样本：{misclassified_path}")
    print("\n提示：这是小样本演示，训练集预测不代表模型的真实泛化准确率。")


if __name__ == "__main__":
    main()
