import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os

# 🌐 Language and NLP setup
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

import spacy
nlp = spacy.load("en_core_web_sm")

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


# ---------------- UI Layout ----------------
st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)


from utils.storage import save_prefs, load_prefs, append_session, load_sessions

st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='color:#0f172a'>🧠 AI Data Cleaning Console</h1>", unsafe_allow_html=True)
st.markdown("Upload a CSV or Excel file and let the AI detect issues, clean data, and suggest fixes automatically.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
    st.markdown("---")
    st.write("💡 Tips:")
    st.write("- Files should include headers.")
    st.write("- Try samples/messy_sample.csv for demo.")
    st.markdown("---")

    with st.expander("🕒 Session History"):
        sessions = load_sessions().get("sessions", [])
        if not sessions:
            st.write("No sessions yet.")
        else:
            for s in reversed(sessions[-10:]):
                st.markdown(f"*{s.get('timestamp','-')}* — {s.get('file_name','-')}")
                st.markdown(f"Rows after: {s.get('rows_after','-')}  \nActions: {', '.join(s.get('actions',[]))}")
                st.markdown("---")

        if st.button("Download session log"):
            st.download_button("Download JSON", json.dumps(load_sessions(), indent=2), "sessions.json", "application/json")

if not uploaded_file:
    st.info("⬆️ Please upload a CSV or Excel file to begin.")
else:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            import openpyxl
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read the file: {e}")
        st.stop()

    original_rows = df.shape[0]
    left, right = st.columns((2, 1))

    with left:
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("🔍 Detected Issues")
        missing = get_missing_counts(df)
        if missing.sum() == 0:
            st.write("No missing values detected.")
        else:
            st.write("Missing values (per column):")
            st.table(missing[missing > 0])

        duplicates = count_duplicates(df)
        st.write(f"Duplicate rows count: *{duplicates}*")

        outliers = numeric_outlier_counts(df)
        if outliers:
            st.write("Numeric outliers (approx counts):")
            st.write(outliers)
        else:
            st.write("No major numeric outliers detected.")

    if 'text_cols' not in locals():
     text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    with right:
        st.subheader("🤖 AI Text Cleanup")

        enable_context_ai = st.checkbox("✨ Enable Deep Context-Aware Correction (GPT)", value=False)

        if st.button("Run Hybrid AI Text Correction"):
            try:
                for col in text_cols:
                    new_values = []
                    for val in df[col].astype(str):
                        # 🌍 Detect language safely
                        try:
                            lang = detect(val)
                        except Exception:
                            lang = "unknown"

                        # 🧠 Skip non-English texts or numeric-only
                        if lang != "en" or val.isdigit():
                            new_values.append(val)
                            continue

                        # 🧾 Use spaCy to check if it contains named entities (proper nouns)
                        doc = nlp(val)
                        if any(ent.label_ in ["PERSON", "GPE", "ORG"] for ent in doc.ents):
                            new_values.append(val)  # skip name/place/org to protect it
                            continue

                        # 🚀 Hybrid AI + Context correction
                        cleaned_val = hybrid_text_suggestions(val)
                        if enable_context_ai:
                            cleaned_val = safe_context_ai_clean(pd.DataFrame({col: [cleaned_val]}))[col].iloc[0]
                        new_values.append(cleaned_val)

                    df[col] = new_values

                st.success("✅ Advanced Hybrid AI Text Correction applied successfully — names & languages protected!")
                st.dataframe(df.head())

            except Exception as e:
                st.error(f"AI text correction failed: {e}")

        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")