import pandas as pd


def suggest_column_fixes(df: pd.DataFrame) -> dict:
    """
    Detect potential column-level issues and return suggestions.
    """
    suggestions = {}

    for col in df.columns:
        col_suggestions = []

        # 1. Suspicious capitalization
        if df[col].dtype == "object":
            inconsistent = df[col].str.contains(r"[A-Z]", na=False).any() and \
                           df[col].str.contains(r"[a-z]", na=False).any()
            if inconsistent:
                col_suggestions.append("Inconsistent capitalization")

        # 2. Leading/trailing spaces
        if df[col].dtype == "object":
            if df[col].str.contains(r"^\s+|\s+$", regex=True, na=False).any():
                col_suggestions.append("Leading/trailing spaces found")

        # 3. Many missing values
        null_ratio = df[col].isna().mean()
        if null_ratio > 0.4:
            col_suggestions.append("High missing value ratio")

        if col_suggestions:
            suggestions[col] = col_suggestions

    return suggestions


def suggest_row_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Suggests row-level issues such as incomplete or malformed records.
    """
    suggestions = []

    for idx, row in df.iterrows():
        issues = []

        # Empty row
        if row.isna().all():
            issues.append("Row completely empty")

        # Too few filled columns
        if row.count() <= (len(df.columns) * 0.3):
            issues.append("Mostly empty row")

        if issues:
            suggestions.append({
                "index": idx,
                "issues": issues
            })

    return pd.DataFrame(suggestions)
