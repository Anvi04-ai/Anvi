import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import os
import re
import spacy
from spellchecker import SpellChecker
from langdetect import detect, DetectorFactory

# ---------------- AI UTILITIES ----------------
from fuzzywuzzy import fuzz

# Stable language detection
DetectorFactory.seed = 0

# Load multilingual NER model (fallback to English)
try:
    nlp = spacy.load("xx_ent_wiki_sm")
except:
    nlp = spacy.load("en_core_web_sm")

# Multilingual spellcheckers
SPELL_CHECKERS = {
    "en": SpellChecker(language='en'),
    "fr": SpellChecker(language='fr'),
    "es": SpellChecker(language='es'),
    "de": SpellChecker(language='de'),
    "it": SpellChecker(language='it')
}

# Common typo fixes
COMMON_REPLACEMENTS = {
    "counrty": "country",
    "adress": "address",
    "hte": "the",
    "teh": "the",
    "recieve": "receive",
    "collge": "college",
    "technlogy": "technology",
    "enviroment": "environment",
    "reserch": "research",
    "inndia": "India",
    "imndfia": "India"
}

# Custom words protection file
CUSTOM_WORDS_PATH = "custom_words.txt"

def load_custom_words():
    if os.path.exists(CUSTOM_WORDS_PATH):
        with open(CUSTOM_WORDS_PATH, "r") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())
    return set()

def save_custom_word(word):
    with open(CUSTOM_WORDS_PATH, "a") as f:
        f.write(f"{word.lower()}\n")

# ---------------- HYBRID + CONTEXT AI CLEANING ----------------
def detect_language_safe(text):
    try:
        return detect(text)
    except:
        return "en"

def is_named_entity(word):
    if not isinstance(word, str) or len(word.strip()) == 0:
        return False
    doc = nlp(word)
    return any(ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"] for ent in doc.ents)

def hybrid_text_suggestions(text):
    spell = SpellChecker()
    custom_words = load_custom_words()
    suggestions = []

    if not isinstance(text, str):
        text = str(text)
    words = text.split()

    for word in words:
        clean = word.strip(".,!?;:").lower()
        if clean.isdigit() or "@" in clean or clean in custom_words:
            suggestions.append((word, word, 1.0))
            continue
        if clean in spell:
            suggestions.append((word, word, 1.0))
            continue
        suggestion = spell.correction(clean)
        score = fuzz.ratio(clean, suggestion) / 100 if suggestion else 0
        if suggestion and score >= 0.7:
            suggestions.append((word, suggestion, score))
        else:
            suggestions.append((word, word, 0.5))
    return suggestions

def hybrid_text_clean(text):
    suggestions = hybrid_text_suggestions(text)
    corrected_words = [s[1] for s in suggestions]
    return " ".join(corrected_words)

def correct_word(word, lang):
    clean_word = re.sub(r'[^\w\s]', '', word.lower())
    if is_named_entity(word):
        return word
    if clean_word in COMMON_REPLACEMENTS:
        corrected = COMMON_REPLACEMENTS[clean_word]
        if word.istitle():
            corrected = corrected.capitalize()
        elif word.isupper():
            corrected = corrected.upper()
        return corrected
    spell = SPELL_CHECKERS.get(lang, SPELL_CHECKERS["en"])
    if clean_word and clean_word not in spell:
        suggestion = spell.correction(clean_word)
        if suggestion and suggestion != clean_word:
            if word.istitle():
                suggestion = suggestion.capitalize()
            elif word.isupper():
                suggestion = suggestion.upper()
            return suggestion
    return word

def gpt_context_correction(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    if text.isdigit() or len(text) <= 2:
        return text
    lang = detect_language_safe(text)
    words = text.split()
    corrected_words = [correct_word(w, lang) for w in words]
    corrected_text = " ".join(corrected_words)
    return re.sub(r'\s+', ' ', corrected_text).strip()

def safe_context_ai_clean(df):
    try:
        for col in df.columns:
            if str(df[col].dtype) not in ['object', 'string']:
                continue
            df[col] = df[col].astype(str).fillna("")
            df[col] = df[col].apply(lambda x: gpt_context_correction(hybrid_text_clean(x)))
        return df
    except Exception as e:
        print(f"❌ Context AI correction failed: {e}")
        return df


# ---------------- ORIGINAL APP BELOW ----------------
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

    with right:
        st.subheader("🤖 AI Text Cleanup")
        enable_context_ai = st.checkbox("✨ Enable Deep Context-Aware Correction (GPT)", value=True)

        if st.button("Run Hybrid AI Text Correction"):
            try:
                df = safe_context_ai_clean(df)
                st.success("✅ Multilingual AI Text Correction applied successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"AI text correction failed: {e}")

        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")