import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
from langdetect import detect, DetectorFactory
from textblob import TextBlob
from spellchecker import SpellChecker

# Ensure consistent language detection
DetectorFactory.seed = 0

# 🧠 Lightweight NLP-based text correction
spell = SpellChecker()

def simple_language_check(text):
    try:
        return detect(text)
    except:
        return "unknown"

def get_grammar_score(text):
    blob = TextBlob(text)
    return round(1 - len(blob.correct().split()) / len(text.split()), 2) if text.strip() else 1


# --- Lightweight context-aware correction ---
def gpt_context_correction(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()

    replacements = {
        "counrty": "country",
        "adress": "address",
        "imndfia": "India",
        "rooll": "roll",
        "rool": "roll",
        "hte": "the",
        "teh": "the",
        "recieve": "receive",
        "adresss": "address",
        "studnt": "student",
        "collge": "college",
        "technlogy": "technology",
        "enviroment": "environment",
        "reserch": "research"
    }

    def correct_word(word):
        clean_word = word.lower()
        if clean_word in replacements:
            corrected = replacements[clean_word]
        else:
            # fallback to spellchecker if word looks English
            corrected = spell.correction(word) if word.isalpha() else word

        if word.istitle():
            corrected = corrected.capitalize()
        elif word.isupper():
            corrected = corrected.upper()
        return corrected

    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    corrected_text = " ".join(corrected_words)
    corrected_text = " ".join(corrected_text.split())
    return corrected_text


def safe_context_ai_clean(df):
    try:
        for col in df.select_dtypes(include=['object', 'string']).columns:
            # Skip numeric-like columns
            if pd.to_numeric(df[col], errors='coerce').notnull().mean() > 0.5:
                continue
            df[col] = df[col].astype(str).fillna("")
            df[col] = df[col].apply(gpt_context_correction)
        return df
    except Exception as e:
        print(f"❌ Context AI correction failed: {e}")
        return df


# ---------------------- Streamlit App ----------------------

st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

from ai.autocorrect_hybrid import hybrid_text_suggestions

try:
    from suggestions.suggest import suggest_issues
except Exception:
    def suggest_issues(df: pd.DataFrame, top_n=5):
        suggestions = []
        missing = df.isnull().mean()
        high_missing = missing[missing > 0.3]
        for col, frac in high_missing.items():
            suggestions.append({
                "type": "missing_high",
                "column": col,
                "message": f"{col} has {int(frac*100)}% missing values. Consider imputing or dropping."
            })
        for col in df.columns:
            if df[col].dtype == object:
                sample = df[col].dropna().astype(str).head(50).tolist()
                date_like = sum(1 for v in sample if any(sep in v for sep in ['/', '-', '.']) and any(ch.isdigit() for ch in v))
                if date_like > 10:
                    suggestions.append({
                        "type": "parse_dates",
                        "column": col,
                        "message": f"{col} contains many date-like strings. Consider parsing to datetime."
                    })
        return suggestions[:top_n]


# Header
st.markdown("<h1 style='color:#0f172a'>🧠 AI Data Cleaning Console</h1>", unsafe_allow_html=True)
st.markdown("Upload a CSV or Excel file and let the AI detect issues, clean data, and suggest fixes automatically.")

# Sidebar
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

    # ---------- Left Panel ----------
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

    # ---------- Right Panel ----------
    with right:
        st.subheader("💡 AI Suggestions")
        suggestions = suggest_issues(df, top_n=5)
        if not suggestions:
            st.write("✅ No major issues detected.")
        else:
            for i, s in enumerate(suggestions):
                st.info(f"{s['message']}")
                key = f"apply_{i}{s.get('column','')}{s.get('type','')}"
                if st.button(f"Apply suggestion for {s.get('column','')}", key=key):
                    if s['type'] == 'missing_high':
                        df[s['column']] = df[s['column']].fillna("Unknown")
                        st.success(f"Filled missing in {s['column']} with 'Unknown'.")
                    elif s['type'] == 'parse_dates':
                        try:
                            df[s['column']] = pd.to_datetime(df[s['column']], errors='coerce')
                            st.success(f"Parsed {s['column']} to datetime.")
                        except Exception as e:
                            st.error(f"Failed to parse: {e}")
                    else:
                        st.warning("Custom handling required for this issue.")

        st.markdown("---")
        st.subheader("🧩 Auto Cleaning")
        strategy = st.selectbox("Missing value strategy", ["mean", "median", "zero"], index=0)

        if st.button("🚀 Apply Auto Cleaning"):
            try:
                df_before = df.copy()
                df = apply_auto_corrections(df, strategy=strategy, normalize_text=True, parse_dates_flag=True)
                st.success("✅ Data cleaned successfully!")
                st.dataframe(df.head())
                append_session({
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    "file_name": uploaded_file.name,
                    "rows_before": df_before.shape[0],
                    "rows_after": df.shape[0],
                    "actions": ["apply_auto_corrections"],
                    "strategy": strategy
                })
            except Exception as e:
                st.error(f"Cleaning failed: {e}")

        # ---------- Text Fixes ----------
        st.markdown("---")
        st.subheader("🪄 Text Formatting Fixes")

        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if text_cols:
            selected_cols = st.multiselect("Select columns to capitalize (or leave empty for all):", text_cols)
            if st.button("Fix Capitalization"):
                try:
                    df = normalize_text_case(df, columns=selected_cols if selected_cols else None)
                    st.success("Capitalization fixed successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Capitalization fix failed: {e}")
        else:
            st.info("No text columns found for capitalization.")

        # ---------- Hybrid + Context Correction ----------
        st.markdown("---")
        st.subheader("🤖 AI Text Cleanup")

        enable_context_ai = st.checkbox("✨ Enable Deep Context-Aware Correction (Light NLP Mode)", value=True)

        if st.button("Run Hybrid AI Text Correction"):
            try:
                for col in text_cols:
                    df[col] = df[col].astype(str).apply(hybrid_text_suggestions)
                if enable_context_ai:
                    df = safe_context_ai_clean(df)
                st.success("✅ Hybrid AI + Context Correction applied successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"AI text correction failed: {e}")

        # ---------- Download ----------
        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

        if st.button("Save my cleaning preferences"):
            save_prefs({"missing_strategy": strategy})
            st.success("Preferences saved.")

    # ---------- Summary ----------
    st.markdown("---")
    if st.button("Save current session to history"):
        session = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "file_name": uploaded_file.name,
            "rows_before": original_rows,
            "rows_after": df.shape[0],
            "actions": ["manual_changes"],
            "strategy": strategy
        }
        append_session(session)
        st.success("Session saved to history.")

    st.markdown("### 📈 Summary")
    st.write(f"Rows before: *{original_rows}* — Rows now: *{df.shape[0]}*")
    st.write(f"Columns: *{df.shape[1]}*")