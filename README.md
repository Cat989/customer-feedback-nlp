# 客户反馈 NLP 分析项目

一个可本地运行的客户反馈分析原型，演示：

- 中文清洗与分词
- SnowNLP 情感初筛 + GPT-4 边界增强
- TF-IDF + 逻辑回归意图识别
- 数据质量概览、模型评估、误判分析
- 关键词提取与可视化仪表盘

## 目录结构

- `app.py`：入口
- `config.py`：配置
- `data_loader.py`：数据读取与校验
- `preprocess.py`：文本清洗
- `sentiment.py`：情感分析
- `intent.py`：意图识别与评估
- `metrics.py`：数据质量、误判导出、报告写出
- `visualization.py`：仪表盘与词云
- `keywords.py`：关键词提取
- `data/sample_feedback.csv`：演示数据
- `outputs/`：运行产物

## 运行方式

建议使用 Python 3.10+，并在项目目录执行：

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

运行测试：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

如果要启用 GPT-4 边界增强，配置环境变量：

```bash
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4
```

## 产物

运行后会生成：

- `outputs/dashboard.png`
- `outputs/wordcloud.png`
- `outputs/analysis_report.json`
- `outputs/misclassified_samples.csv`

## 项目亮点

- SnowNLP 先做快速打分，只对 0.2~0.8 的边界样本尝试 GPT-4
- 对手机号、邮箱、长编号做脱敏后再发给模型
- 情感、意图、关键词、可视化分层清晰，适合后续继续扩展成看板或 API
- 增加了数据质量统计和模型评估，偏数据分析和业务分析表达

## 局限性

- 演示数据量小，评估结果仅用于展示流程
- 正式业务应接入更大规模人工标注数据
- GPT-4 只作为边界样本增强，不能替代完整的数据标注体系
