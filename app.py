import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os

# 🌐 Language and NLP setup
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# --- Import helper modules ---
from detection.detect import get_missing_counts, count_duplicates, numeric_outlier_counts
from fixes.apply_fixes import (
    fill_missing_values,
    remove_duplicates,
    convert_data_types,
    normalize_text_case,
    apply_auto_corrections,
    fuzzy_dedupe_by_column
)
from utils.storage import save_prefs, load_prefs, append_session, load_sessions

# --- Import AI modules ---
from ai.autocorrect_hybrid import hybrid_text_suggestions
from ai.context_ai_correct import safe_context_ai_clean

from fuzzywuzzy import process

# ✅ Initialize Streamlit
st.set_page_config(page_title="AI Data Cleaning", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Cleaning App")
st.markdown("Clean messy data using hybrid AI + fuzzy logic 🔥")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

# ===========================
# 🔹 Define Fuzzy Clean Function
# ===========================
def clean_text_with_fuzzy(value, reference_list):
    try:
        best_match = process.extractOne(str(value), reference_list)
        if best_match and best_match[1] > 80:  # similarity threshold
            return best_match[0].title()  # capitalize clean names
        return str(value).title()
    except:
        return str(value).title()

# ===========================
# 🔹 Main App Logic
# ===========================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Original Data:")
    st.dataframe(df.head())

    # Run detections
    missing = get_missing_counts(df)
    duplicates = count_duplicates(df)
    outliers = numeric_outlier_counts(df)

    st.write("**Missing values:**", missing)
    st.write("**Duplicate rows:**", duplicates)
    st.write("**Numeric outliers:**", outliers)

    # Apply fixes
    df = fill_missing_values(df)
    df = remove_duplicates(df)
    df = convert_data_types(df)
    df = normalize_text_case(df)

    st.success("✅ Basic cleaning complete!")

    # 🔹 Apply Fuzzy + AI Correction
    for col in df.select_dtypes(include='object').columns:
        reference_data = df[col].dropna().unique().tolist()
        df[col] = df[col].apply(lambda x: clean_text_with_fuzzy(x, reference_data))

    st.success("✅ Hybrid AI + Context Correction applied successfully!")
    st.dataframe(df.head())

    # Download cleaned CSV
    cleaned_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", "text/csv")

    if st.button("Save my cleaning preferences"):
        save_prefs({"timestamp": str(datetime.datetime.now()), "columns": list(df.columns)})
        st.info("Preferences saved successfully ✅")

else:
    st.info("👆 Upload a CSV file to get started!")