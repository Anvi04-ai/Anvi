import pandas as pd

from fixes.apply_fixes import apply_basic_fixes
from autocorrect_hybrid import hybrid_autocorrect
from context_ai_correct import apply_context_corrections


def full_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master cleaning pipeline combining:
    - Basic fixes
    - Hybrid autocorrect
    - Context-aware rule engine
    """

    df = df.copy()

    # 1. Trim, fix spaces, normalize text
    df = apply_basic_fixes(df)

    # 2. Hybrid autocorrect (dictionary + fuzzy)
    df = hybrid_autocorrect(df)

    # 3. Context-based corrections (domain rules, patterns)
    df = apply_context_corrections(df)

    return df
