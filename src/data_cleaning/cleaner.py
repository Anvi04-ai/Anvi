import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from rapidfuzz import fuzz  # For fuzzy dedup
from utils.logger import logger

def apply_cleaning(df: pd.DataFrame, suggestions: list[str]) -> pd.DataFrame:
    """Apply selected cleanings with validation."""
    cleaned = df.copy()
    for sug in suggestions:
        try:
            if "missing" in sug.lower():
                num_cols = cleaned.select_dtypes(include=[np.number]).columns
                if len(num_cols) > 0:
                    imputer = SimpleImputer(strategy='median')
                    cleaned[num_cols] = imputer.fit_transform(cleaned[num_cols])
            elif "duplicates" in sug.lower():
                # Fuzzy dedup example for strings
                str_cols = cleaned.select_dtypes(include=[object]).columns
                for col in str_cols:
                    indices = []
                    for i in range(len(cleaned)):
                        for j in range(i+1, len(cleaned)):
                            if fuzz.ratio(str(cleaned.iloc[i][col]), str(cleaned.iloc[j][col])) > 90:
                                indices.append(j)
                    cleaned = cleaned.drop(indices).reset_index(drop=True)
                cleaned = cleaned.drop_duplicates()
            logger.info(f"Applied: {sug}")
        except Exception as e:
            logger.warning(f"Partial fail on {sug}: {str(e)}")
            continue
    return cleaned