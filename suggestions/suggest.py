import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

def suggest_issues(df: pd.DataFrame, top_n: int = 5):
    """
    Generate smart suggestions for data cleaning:
    - High missing values
    - Duplicate-like columns
    - Date parsing
    - Numeric outliers
    - String normalization
    """

    suggestions = []


    missing_ratio = df.isnull().mean()
    high_missing = missing_ratio[missing_ratio > 0.3]
    for col, frac in high_missing.items():
        suggestions.append({
            "type": "missing_high",
            "column": col,
            "message": f"'{col}' has {int(frac * 100)}% missing values. Consider imputing or dropping it."
        })


    columns = list(df.columns)
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            sim = fuzz.ratio(columns[i].lower(), columns[j].lower())
            if sim > 85 and columns[i] != columns[j]:
                suggestions.append({
                    "type": "similar_columns",
                    "columns": [columns[i], columns[j]],
                    "message": f"Columns '{columns[i]}' and '{columns[j]}' look similar. Maybe duplicates."
                })


    for col in df.select_dtypes(include=[np.number]).columns:
        mean, std = df[col].mean(), df[col].std()
        if std == 0 or np.isnan(std):
            continue
        z_scores = np.abs((df[col] - mean) / std)
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            suggestions.append({
                "type": "numeric_outliers",
                "column": col,
                "message": f"{outliers} outlier values detected in '{col}'. Consider removing or capping them."
            })


    for col in df.select_dtypes(include=['object']).columns:
        sample = df[col].dropna().astype(str).head(50).tolist()
        date_like = sum(1 for v in sample if any(sep in v for sep in ['/', '-', '.']) and any(ch.isdigit() for ch in v))
        if date_like > 10:
            suggestions.append({
                "type": "parse_dates",
                "column": col,
                "message": f"Column '{col}' has many date-like strings. Consider parsing it as datetime."
            })


    for col in df.select_dtypes(include=['object']).columns:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio < 0.1 and df[col].nunique() > 3:
            suggestions.append({
                "type": "string_cleanup",
                "column": col,
                "message": f"'{col}' may have inconsistent text casing or extra spaces. Consider normalization."
            })


    unique_msgs = {s['message']: s for s in suggestions}.values()
    return list(unique_msgs)[:top_n]