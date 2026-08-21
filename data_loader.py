from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_FILE, INTENT_ORDER


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"不支持的数据格式：{path.suffix}")


def load_data(path: Path = DATA_FILE) -> pd.DataFrame:
    """读取并校验反馈数据，保留可追踪的数据质量信息。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    raw = _read_table(path)
    required = {"text", "intent"}
    if not required.issubset(raw.columns):
        raise ValueError("数据文件必须包含 text 和 intent 两列。")

    source_rows = len(raw)
    raw = raw.copy()
    missing_mask = raw[["text", "intent"]].isna().any(axis=1)
    dropped_missing_rows = int(missing_mask.sum())
    data = raw.loc[~missing_mask].copy()

    data["text"] = data["text"].astype(str).str.strip()
    data["intent"] = data["intent"].astype(str).str.strip()
    empty_mask = (data["text"] == "") | (data["intent"] == "")
    dropped_empty_rows = int(empty_mask.sum())
    data = data.loc[~empty_mask].copy()

    duplicate_rows = int(data.duplicated(subset=["text", "intent"]).sum())
    if duplicate_rows:
        data = data.drop_duplicates(subset=["text", "intent"]).copy()

    unknown = sorted(set(data["intent"]) - set(INTENT_ORDER))
    if unknown:
        raise ValueError(f"发现未定义的意图类别：{unknown}")

    data.attrs["load_stats"] = {
        "source_rows": source_rows,
        "dropped_missing_rows": dropped_missing_rows,
        "dropped_empty_rows": dropped_empty_rows,
        "duplicate_rows": duplicate_rows,
        "loaded_rows": len(data),
    }
    return data
