import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import logging
from pathlib import Path


LOG_PATH = Path.cwd() / "logs"
LOG_PATH.mkdir(exist_ok=True)
log_file = LOG_PATH / "app.log"

logging.basicConfig(
    filename=str(log_file),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("ai_data_cleaner")


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


st.set_page_config(
    page_title="AI Data Cleaner",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<div style="background:#f8fafc;padding:12px;border-radius:8px;margin-bottom:15px">
<h3 style="margin:0">🧠 Welcome — AI Data Cleaner</h3>
<p style="margin:4px 0 0">Upload a CSV, click <b>Apply Auto Cleaning</b>, optionally normalize text, then download the cleaned file.</p>
<ul style="margin:6px 0 0;padding-left:20px">
<li>Handles: missing values, duplicates, wrong types, fuzzy text, capitalization</li>
<li>Try a sample: <code>samples/messy_sample.csv</code></li>
</ul>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "xls"])
    st.markdown("---")
    st.write("💡 Tips:")
    st.write("- Supported formats: CSV, XLSX, XLS")
    st.write("- File must include headers")
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
            st.download_button("Download JSON", json.dumps(load_sessions(), indent=2),
                               "sessions.json", "application/json")


if not uploaded_file:
    st.info("⬆️ Please upload a CSV or Excel file to begin.")
else:
    file_name = uploaded_file.name.lower()
    try:
        if file_name.endswith(".csv"):
           df = pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file to begin.")
            st.stop()

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
        st.subheader("💡 AI Suggestions")
        suggestions = suggest_issues(df, top_n=5)
        if not suggestions:
            st.write("✅ No major issues detected.")
        else:
            for i, s in enumerate(suggestions):
                st.info(f"{s['message']}")
                key = f"apply_{i}{s.get('column','')}{s.get('type','')}"
                if st.button(f"Apply suggestion for {s.get('column','')}", key=key):
                    try:
                        if s['type'] == 'missing_high':
                            df[s['column']] = df[s['column']].fillna("Unknown")
                            st.success(f"Filled missing in {s['column']} with 'Unknown'.")
                        elif s['type'] == 'parse_dates':
                            df[s['column']] = pd.to_datetime(df[s['column']], errors='coerce')
                            st.success(f"Parsed {s['column']} to datetime.")
                        logger.info(f"Applied suggestion: {s}")
                    except Exception as e:
                        st.error(f"Failed to apply suggestion: {e}")
                        logger.exception("Suggestion failed")


        st.markdown("---")
        st.subheader("🧩 Auto Cleaning")
        strategy = st.selectbox(
            "Missing value strategy",
            ["mean", "median", "zero"],
            index=0,
            help="Choose how numeric missing values are filled."
        )

        if st.button("🚀 Apply Auto Cleaning"):
            start_ts = datetime.datetime.utcnow()
            with st.spinner("Cleaning data — please wait..."):
                try:
                    df_before = df.copy()
                    df = apply_auto_corrections(df, strategy=strategy,
                                                normalize_text=True, parse_dates_flag=True)
                    duration = (datetime.datetime.utcnow() - start_ts).total_seconds()
                    st.success("✅ Data cleaned successfully!")


                    st.write("Before / After sample")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Before")
                        st.dataframe(df_before.head(5))
                    with col2:
                        st.write("After")
                        st.dataframe(df.head(5))

                    append_session({
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        "file_name": uploaded_file.name,
                        "rows_before": df_before.shape[0],
                        "rows_after": df.shape[0],
                        "actions": ["apply_auto_corrections"],
                        "strategy": strategy,
                        "duration_seconds": duration
                    })
                    logger.info("Auto cleaning completed successfully.")
                except Exception as e:
                    st.error(f"Cleaning failed: {e}")
                    logger.exception("Auto cleaning failed")


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
                    logger.exception("Capitalization fix failed")
        else:
            st.info("No text columns found for capitalization.")


        st.markdown("---")
        st.subheader("🧠 Fuzzy Normalization")
        if text_cols:
            col_to_dedupe = st.selectbox(
                "Select a column to fuzzy-normalize",
                options=["(none)"] + text_cols,
                key="fuzzy_select"
            )
            if col_to_dedupe != "(none)":
                if st.button(f"Apply fuzzy dedupe to {col_to_dedupe}", key="fuzzy_single"):
                    df = fuzzy_dedupe_by_column(df, col_to_dedupe, threshold=88)
                    df = normalize_text_case(df, [col_to_dedupe])
                    st.success(f"Fuzzy normalization + capitalization applied on '{col_to_dedupe}'.")
            if st.checkbox("Apply fuzzy normalization to ALL text columns", key="fuzzy_all_checkbox"):
                if st.button("Run fuzzy normalization (All Columns)", key="fuzzy_all_button"):
                    for col in text_cols:
                        df = fuzzy_dedupe_by_column(df, col, threshold=88)
                    df = normalize_text_case(df, text_cols)
                    st.success("Fuzzy normalization + capitalization applied to all text columns.")
        else:
            st.info("No text columns found for fuzzy normalization.")


        st.markdown("---")
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned CSV", csv_bytes,
                           file_name="cleaned_data.csv", mime="text/csv")

        if st.button("Save my cleaning preferences"):
            save_prefs({"missing_strategy": strategy})
            st.success("Preferences saved.")
            logger.info("User preferences saved.")


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
        logger.info("Session saved manually.")

    st.markdown("### 📈 Summary")
    st.write(f"Rows before: *{original_rows}* — Rows now: *{df.shape[0]}*")
    st.write(f"Columns: *{df.shape[1]}*")