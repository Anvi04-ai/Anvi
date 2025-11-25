import pandas as pd


def apply_basic_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply automatic basic corrections without requiring a corrections list.
    This prevents missing argument errors.
    """

    df = df.copy()

    # 1. Trim spaces from all string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # 2. Convert column names to consistent formatting
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    # 3. Replace obvious wrong values
    df.replace(
        to_replace={
            "nan": None,
            "NaN": None,
            "": None,
            "none": None,
            "None": None,
        },
        inplace=True
    )

    # 4. Fix common typos
    typo_fixes = {
        "malee": "male",
        "femlae": "female",
        "femlAe": "female",
        "unkown": "unknown"
    }

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].replace(typo_fixes)

    return df


def apply_corrections(df: pd.DataFrame, corrections=None) -> pd.DataFrame:
    """
    OPTIONAL: For future manual corrections.
    """
    df = df.copy()

    if corrections:
        for col, mapping in corrections.items():
            if col in df.columns:
                df[col] = df[col].replace(mapping)

    return df
