import pandas as pd


def suggest_improvements(df: pd.DataFrame) -> dict:
    """
    Provide high-level dataset quality improvement suggestions.
    Focuses on structure, formatting, schema, and completeness.
    """

    suggestions = {}

    # ---------- 1. Missing values overview ----------
    null_ratio = df.isna().mean()
    high_null_cols = null_ratio[null_ratio > 0.3].index.tolist()

    if high_null_cols:
        suggestions["High Missing Value Columns"] = high_null_cols

    # ---------- 2. Columns with mixed data types ----------
    mixed_type_cols = []
    for col in df.columns:
        unique_types = set(type(x).__name__ for x in df[col].dropna())
        if len(unique_types) > 1:
            mixed_type_cols.append(col)

    if mixed_type_cols:
        suggestions["Columns with Mixed Data Types"] = mixed_type_cols

    # ---------- 3. Columns that should likely be categorical ----------
    categorical_candidates = []
    for col in df.columns:
        if df[col].nunique() < len(df) * 0.05:
            categorical_candidates.append(col)

    if categorical_candidates:
        suggestions["Potential Categorical Columns"] = categorical_candidates

    # ---------- 4. Very long text fields ----------
    long_text_cols = []
    for col in df.columns:
        if df[col].dtype == "object":
            if df[col].str.len().mean() > 60:
                long_text_cols.append(col)

    if long_text_cols:
        suggestions["Long Text Columns"] = long_text_cols

    # ---------- 5. Duplicate column names ----------
    if df.columns.duplicated().any():
        suggestions["Duplicate Column Names"] = list(df.columns[df.columns.duplicated()])

    # ---------- 6. Empty or near-empty columns ----------
    empty_cols = df.columns[(df.count() == 0)].tolist()
    sparse_cols = df.columns[(df.count() <= 2)].tolist()

    if empty_cols:
        suggestions["Completely Empty Columns"] = empty_cols

    if sparse_cols:
        suggestions["Sparse Columns (very low data)"] = sparse_cols

    # ---------- 7. ID column detection ----------
    id_like_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].isna().sum() == 0:
            id_like_cols.append(col)

    if id_like_cols:
        suggestions["Possible ID Columns"] = id_like_cols

    return suggestions
