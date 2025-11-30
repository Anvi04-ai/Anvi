import sys
import os
# Add src to path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import pandas as pd
import numpy as np
from data_cleaning.cleaner import apply_cleaning  # Now works!


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, np.nan, 3, 4],
        'B': ['x', 'x', 'y', 'z'],
        'C': [10, 20, 20, 30]
    })


def test_apply_cleaning_missing(sample_df):
    suggestions = ["handle missing values"]
    cleaned = apply_cleaning(sample_df, suggestions)
    assert not cleaned['A'].isnull().any()


def test_apply_cleaning_duplicates(sample_df):
    df_with_dups = pd.concat([sample_df, sample_df.iloc[[0]]])
    suggestions = ["remove duplicates"]
    cleaned = apply_cleaning(df_with_dups, suggestions)
    assert len(cleaned) == len(sample_df)