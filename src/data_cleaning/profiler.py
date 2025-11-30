import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest  # For outliers

def profile_data(df: pd.DataFrame) -> dict:
    """Generate comprehensive data profile with error handling."""
    try:
        missing = df.isnull().sum().to_dict()
        duplicates = df.duplicated().sum()
        dtypes = df.dtypes.value_counts().to_dict()
        # Outlier detection example
        if len(df.select_dtypes(include=[np.number]).columns) > 0:
            iso = IsolationForest(contamination=0.1)
            outliers = iso.fit_predict(df.select_dtypes(include=[np.number]))
            outlier_count = sum(outliers == -1)
        else:
            outlier_count = 0
        return {
            'shape': df.shape,
            'missing': missing,
            'duplicates': duplicates,
            'dtypes': dtypes,
            'outliers': outlier_count
        }
    except Exception as e:
        raise ValueError(f"Profiling failed: {str(e)}")