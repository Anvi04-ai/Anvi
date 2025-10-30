import pandas as pd
import numpy as np

def get_missing_counts(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum()

def count_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

def numeric_outlier_counts(df: pd.DataFrame, z_thresh=3) -> dict:
    nums = df.select_dtypes(include=np.number)
    out = {}
    for col in nums.columns:
        col_series = nums[col].dropna()
        mean = col_series.mean()
        std = col_series.std()
        if std == 0 or np.isnan(std):
            continue
        mask = (col_series > mean + z_thresh*std) | (col_series < mean - z_thresh*std)
        ct = int(mask.sum())
        if ct > 0:
            out[col] = ct
    return out

def detect_type_issues(df: pd.DataFrame, sample_size=100) -> dict:
    issues = {}
    for col in df.columns:
        try:
            pd.to_numeric(df[col].dropna().sample(min(sample_size, df.shape[0])))
        except Exception:
            issues[col] = "contains non-numeric values"
    return issues