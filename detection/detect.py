import pandas as pd
from Levenshtein import ratio


def detect_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return exact duplicate rows.
    """
    df = df.copy()
    duplicates = df[df.duplicated(keep=False)]
    return duplicates


def detect_fuzzy_duplicates(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """
    Detect fuzzy duplicates row-by-row using string similarity.
    Suitable for small–medium size datasets.
    """
    df = df.copy()
    fuzzy_matches = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            row1 = " ".join(df.iloc[i].astype(str))
            row2 = " ".join(df.iloc[j].astype(str))

            sim = ratio(row1, row2)

            if sim >= threshold:
                fuzzy_matches.append({
                    "row_i": i,
                    "row_j": j,
                    "similarity": sim
                })

    return pd.DataFrame(fuzzy_matches)
