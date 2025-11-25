import pandas as pd


def fix_gender(val: str) -> str:
    v = val.lower().strip()

    mapping = {
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female",
        "femlae": "female",
        "girl": "female",
        "boy": "male",
    }

    return mapping.get(v, val)


def fix_age(val):
    try:
        v = int(float(val))
        if 0 < v < 120:
            return v
        return None
    except:
        return None


def apply_context_corrections(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gender correction
    gender_cols = [c for c in df.columns if "gender" in c.lower()]
    for col in gender_cols:
        df[col] = df[col].astype(str).apply(fix_gender)

    # Age correction
    age_cols = [c for c in df.columns if "age" in c.lower()]
    for col in age_cols:
        df[col] = df[col].apply(fix_age)

    # Date normalization
    date_cols = [c for c in df.columns if "date" in c.lower()]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
