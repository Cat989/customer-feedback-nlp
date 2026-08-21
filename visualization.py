from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

from customer_feedback_nlp.config import INTENT_ORDER, PLOT_COLORS, SENTIMENT_ORDER, WORDCLOUD_MAX_WORDS

try:
    from wordcloud import WordCloud
except ImportError:  # 词云是可选输出，主流程仍可运行
    WordCloud = None


def find_chinese_font() -> str | None:
    """查找常见中文字体，避免硬编码单一操作系统路径。"""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    for path in font_manager.findSystemFonts():
        name = Path(path).name.lower()
        if any(key in name for key in ("msyh", "simhei", "simsun", "noto", "wqy")):
            return path
    return None


def configure_matplotlib(font_path: str | None) -> None:
    """统一图表字体、背景和网格样式。"""
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#c3c2b7",
        "grid.color": "#e1e0d9",
        "grid.linewidth": 0.7,
    })
    if font_path:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        print("提示：未找到中文字体，图表中的中文可能显示为方框。")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#fcfcfb"
    plt.rcParams["axes.facecolor"] = "#fcfcfb"


def draw_dashboard(
    data: pd.DataFrame,
    quality_stats: dict,
    intent_metrics: dict,
    output_path: Path,
) -> None:
    """绘制情感、意图、分数和摘要四块仪表盘。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#fcfcfb")
    fig.suptitle("客户反馈 NLP 分析仪表盘（演示数据）", fontsize=18, fontweight="bold", color="#0b0b0b")

    sentiment_counts = data["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
    ax = axes[0, 0]
    bars = ax.bar(
        sentiment_counts.index,
        sentiment_counts.values,
        color=[PLOT_COLORS[item] for item in SENTIMENT_ORDER],
        width=0.62,
    )
    ax.set_title("情感倾向分布", loc="left", fontweight="bold")
    ax.set_ylabel("反馈数量")
    ax.set_ylim(0, max(1, int(sentiment_counts.max()) + 2))
    total = len(data)
    for bar, count in zip(bars, sentiment_counts.values):
        percent = count / total * 100 if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.1, f"{count} ({percent:.0f}%)", ha="center")

    intent_counts = data["predicted_intent"].value_counts().reindex(INTENT_ORDER, fill_value=0)
    ax = axes[0, 1]
    bars = ax.barh(INTENT_ORDER, intent_counts.values, color="#2a78d6", height=0.58)
    ax.invert_yaxis()
    ax.set_title("自动意图识别分布", loc="left", fontweight="bold")
    ax.set_xlabel("预测数量")
    ax.set_xlim(0, max(1, int(intent_counts.max()) + 2))
    for bar, count in zip(bars, intent_counts.values):
        ax.text(count + 0.08, bar.get_y() + bar.get_height() / 2, str(count), va="center")

    ax = axes[1, 0]
    sns.histplot(data["sentiment_score"], bins=np.linspace(0, 1, 11), color="#1baf7a", edgecolor="#fcfcfb", ax=ax)
    ax.axvline(0.3, color="#e34948", linestyle="--", linewidth=1.5, label="负面阈值 0.3")
    ax.axvline(0.7, color="#2a78d6", linestyle="--", linewidth=1.5, label="正面阈值 0.7")
    ax.set_xlim(0, 1)
    ax.set_title("SnowNLP 情感分数", loc="left", fontweight="bold")
    ax.set_xlabel("情感分数（0=负面，1=正面）")
    ax.set_ylabel("反馈数量")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.axis("off")
    positive = int((data["sentiment"] == "正面").sum())
    negative = int((data["sentiment"] == "负面").sum())
    openai_count = int((data["sentiment_source"] == "OpenAI").sum())
    snownlp_count = int((data["sentiment_source"] == "SnowNLP").sum())
    summary = (
        "运行摘要\n\n"
        f"总反馈数       {len(data)}\n"
        f"正面反馈       {positive} ({positive / total:.0%})\n"
        f"负面反馈       {negative} ({negative / total:.0%})\n"
        f"意图类别       {data['predicted_intent'].nunique()} / 5\n"
        f"意图准确率     {intent_metrics['accuracy']:.0%}\n"
        f"加权 F1       {intent_metrics['f1']:.0%}\n"
        f"GPT-4 增强      {openai_count}\n"
        f"SnowNLP 判定    {snownlp_count}\n"
        f"有效反馈数     {quality_stats.get('loaded_rows', len(data))}\n\n"
        "处理链路\n"
        "清洗 → 分词 → SnowNLP → GPT-4 → 意图\n\n"
        "注：仅 SnowNLP 原始分数位于 0.2~0.8 时尝试 GPT-4；"
        "当前为小样本演示。"
    )
    ax.text(0.05, 0.93, summary, va="top", fontsize=12, color="#52514e", linespacing=1.65)

    for sub_ax in axes.flat:
        if sub_ax.axison:
            sub_ax.grid(axis="y", color="#e1e0d9", linewidth=0.7)
            sub_ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_wordcloud(keywords: List[Tuple[str, int]], output_path: Path, font_path: str | None) -> bool:
    """生成中文关键词词云；缺少依赖或字体时返回 False。"""
    if WordCloud is None:
        print("提示：未安装 wordcloud，跳过词云生成。")
        return False
    if not font_path:
        print("提示：未找到中文字体，跳过词云生成。")
        return False
    if not keywords:
        print("提示：没有可用关键词，跳过词云生成。")
        return False

    cloud = WordCloud(
        font_path=font_path,
        width=1000,
        height=560,
        background_color="#fcfcfb",
        colormap="Blues",
        max_words=WORDCLOUD_MAX_WORDS,
        margin=8,
    ).generate_from_frequencies(dict(keywords))
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#fcfcfb")
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title("客户反馈关键词词云", fontsize=18, fontweight="bold", pad=16, color="#0b0b0b")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True
