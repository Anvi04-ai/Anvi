import pandas as pd
import numpy as np
from typing import List, Optional
from fuzzywuzzy import process

# ---------------- Basic fixes ----------------

def trim_and_strip(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names and string values, normalize 'nan' strings to actual NaN."""
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, 'NaN': np.nan, '': np.nan})
    return df

def fill_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Fill missing numeric values by strategy; for non-numeric fill with 'Unknown'.
    strategy: 'mean' | 'median' | 'zero'
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "zero":
                    df[col] = df[col].fillna(0)
        else:
            # Leave object columns to be handled explicitly (or fill with 'Unknown')
            df[col] = df[col].fillna(df[col].dtype == object and "Unknown" or df[col])
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try safer conversions:
    - convert obvious numeric strings to numeric
    - try to coerce date-like columns to datetime where many parse
    """
    df = df.copy()
    # convert numeric-like columns
    for col in df.columns:
        try:
            if df[col].dtype == object:
                # if most values convert to numeric, convert entire column
                parsed = pd.to_numeric(df[col].dropna().astype(str).str.replace(',', ''), errors='coerce')
                if parsed.notna().sum() > 0 and (parsed.notna().sum() / max(1, len(df)) > 0.5):
                    df[col] = parsed
        except Exception:
            pass

    # try lightweight datetime conversion (keep as-is if fails)
    for col in df.columns:
        try:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                # only convert if a sizeable fraction parsed
                if parsed.notna().sum() / max(1, len(df)) > 0.3:
                    df[col] = parsed
        except Exception:
            pass

    return df

# ---------------- Fuzzy dedupe helpers ----------------

def fuzzy_dedupe_by_column(df: pd.DataFrame, col: str, threshold: int = 90) -> pd.DataFrame:
    """
    Replace values in a column by canonical representatives for groups of similar strings.
    This avoids dropping rows; instead it normalizes text values to the chosen canonical value.
    """
    if col not in df.columns:
        return df
    df = df.copy()
    # build list of unique non-null strings
    vals = [v for v in df[col].dropna().astype(str).unique().tolist() if v.strip() != ""]
    canonical = {}
    seen = set()
    for v in vals:
        if v in seen:
            continue
        # find matches to v among remaining values
        matches = process.extract(v, vals, limit=None)
        for match_val, score in matches:
            if score >= threshold:
                canonical[match_val] = v
                seen.add(match_val)
    # apply mapping to column (keep NaN as NaN)
    def map_val(x):
        if pd.isna(x):
            return x
        s = str(x)
        return canonical.get(s, s)
    df[col] = df[col].apply(map_val)
    # convert back explicit 'nan' strings to np.nan
    df[col] = df[col].replace({'nan': np.nan, '': np.nan})
    return df

# ---------------- Text normalization ----------------

def normalize_text_case(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Capitalize each word in selected text columns (or all text columns if None)."""
    df = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=['object', 'string']).columns.tolist()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            # Capitalize each word, keep empty/NaN as-is
            def cap(v):
                if v is None or pd.isna(v) or str(v).strip() == "":
                    return np.nan
                return " ".join(word.capitalize() for word in str(v).split())
            df[col] = df[col].apply(cap)
    return df

# ---------------- Date parsing helper ----------------

def parse_dates_candidates(df: pd.DataFrame, min_fraction: float = 0.3, dayfirst: bool = False) -> pd.DataFrame:
    """
    Attempt to parse columns that look date-like. Converts columns when a sizeable fraction parses.
    """
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        sample = df[col].dropna().astype(str).head(200).tolist()
        date_like_count = sum(1 for v in sample if any(sep in v for sep in ['/', '-', '.']) and any(ch.isdigit() for ch in v))
        if date_like_count > len(sample) * 0.2:  # heuristic
            try:
                parsed = pd.to_datetime(df[col], errors='coerce', dayfirst=dayfirst, infer_datetime_format=True)
                if parsed.notna().sum() / max(1, len(df)) >= min_fraction:
                    df[col] = parsed
            except Exception:
                pass
    return df

# ---------------- Full pipeline ----------------

def apply_auto_corrections(df: pd.DataFrame, strategy: str = "mean", normalize_text: bool = True, parse_dates_flag: bool = True) -> pd.DataFrame:
    """
    Full, safe pipeline:
    1. trim/strip whitespace
    2. parse obvious dates (optional)
    3. convert numeric-like columns
    4. fill missing numeric values
    5. normalize text (optional)
    6. fuzzy normalize values per text column
    7. remove exact duplicate rows
    """
    df = df.copy()

    # 1. Trim & clean basic issues
    df = trim_and_strip(df)

    # 2. Parse date-like columns early if requested
    if parse_dates_flag:
        df = parse_dates_candidates(df)

    # 3. Convert numeric-like columns
    df = convert_data_types(df)

    # 4. Fill numeric missing values (and simple fill for non-numeric)
    df = fill_missing_values(df, strategy=strategy)

    # 5. Normalize case for text to reduce noise (lower-case first to group), then fix capitalization later
    if normalize_text:
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    # 6. Fuzzy-normalize text column values (canonical mapping)
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df = fuzzy_dedupe_by_column(df, col, threshold=90)

    # 7. Capitalize nicely
    if normalize_text:
        df = normalize_text_case(df)

    # 8. Remove exact duplicate rows now that values are normalized
    df = remove_duplicates(df)

    return df