# apply_fixes.py
import pandas as pd

def apply_fixes(df):
    """
    Function to clean and fix DataFrame issues safely.
    """
    # Check if input is a DataFrame
    if df is None:
        raise ValueError("Input DataFrame is None. Please provide a valid DataFrame.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df)} instead.")

    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Example fixes (you can modify as needed)
    # 1. Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Fill NaN values with empty strings (optional)
    df = df.fillna("")

    # 3. Drop duplicate rows (optional)
    df = df.drop_duplicates()

    # Add any other fixes here
    # e.g., df['column_name'] = df['column_name'].str.lower()

    return df
