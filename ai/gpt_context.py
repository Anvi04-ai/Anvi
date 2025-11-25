import pandas as pd


def ai_context_enhance(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Optional GPT-style enhancement layer.
    If no model provided, returns df unchanged.
    """

    if model is None:
        return df  # safe fallback

    df = df.copy()
    enhanced_rows = []

    for idx, row in df.iterrows():
        text = row.to_dict()

        try:
            # The model should accept JSON-like inputs
            response = model(text)
            enhanced_rows.append(response)

        except Exception:
            enhanced_rows.append(text)

    return pd.DataFrame(enhanced_rows)
