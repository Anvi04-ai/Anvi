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

def protect_identifier_columns(df):
    """Ensure identifier columns like Roll No, ID, etc. remain as strings."""
    protected_keywords = ["roll", "id", "number", "code", "reg", "student"]
    for col in df.columns:
        if any(keyword in col.lower() for keyword in protected_keywords):
            df[col] = df[col].astype(str)
    return df

# ✅ Initialize Streamlit
st.set_page_config(page_title="AI Data Cleaning", page_icon="🤖", layout="wide")

st.title("🤖 AI Data Cleaning App")
st.markdown("Clean messy data using hybrid AI + fuzzy logic 🔥")

# Allow CSV and Excel upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

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
    # --- Handle CSV or Excel ---
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, dtype=str)
    else:
        df = pd.read_excel(uploaded_file, dtype=str)

    # 🔒 Protect important columns
    df = protect_identifier_columns(df)

    st.write("### Original Data:")
    st.dataframe(df.head())

    # Run detections
    missing = get_missing_counts(df)
    duplicates = count_duplicates(df)
    outliers = numeric_outlier_counts(df)

    st.write("**Missing values:**", missing)
    st.write("**Duplicate rows:**", duplicates)
    st.write("**Numeric outliers:**", outliers)

    # Apply basic cleaning
    df = fill_missing_values(df)
    df = remove_duplicates(df)
    df = convert_data_types(df)
    df = normalize_text_case(df)

    st.success("✅ Basic cleaning complete!")

    # 🔹 Apply Fuzzy + AI Correction
    for col in df.select_dtypes(include='object').columns:
        reference_data = df[col].dropna().unique().tolist()
        df[col] = df[col].apply(lambda x: clean_text_with_fuzzy(x, reference_data))

    # 🔹 Optional: Context-based AI correction
    try:
        df = safe_context_ai_clean(df)
    except Exception as e:
        st.warning(f"⚠️ Context correction skipped: {e}")

    st.success("✅ Hybrid AI + Context Correction applied successfully!")
    st.dataframe(df.head())

    # --- Download cleaned file options ---
    cleaned_csv = df.to_csv(index=False).encode("utf-8")
    cleaned_excel_path = "cleaned_data.xlsx"
    df.to_excel(cleaned_excel_path, index=False)

    st.download_button("📥 Download Cleaned CSV", cleaned_csv, "cleaned_data.csv", "text/csv")
    with open(cleaned_excel_path, "rb") as f:
        st.download_button("📘 Download Cleaned Excel", f, file_name="cleaned_data.xlsx")

    # --- Save preferences ---
    if st.button("Save my cleaning preferences"):
        save_prefs({"timestamp": str(datetime.datetime.now()), "columns": list(df.columns)})
        st.info("Preferences saved successfully ✅")

else:
    st.info("👆 Upload a CSV or Excel file to get started!")