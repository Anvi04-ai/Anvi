import streamlit as st
import pandas as pd
import json
from fixes.apply_fixes import apply_fixes
from detection.detect import fuzzy_duplicate_pairs
from utils.save_log import save_log


# --------------------------
# Helper functions
# --------------------------

def download_button_for_df(df, filename="output.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download File",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


# --- Upload Section ---
st.title("AI Data Cleaner (Advanced Mode)")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

df_raw = None
cleaned_df = None

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()

    try:
        if file_ext == "csv":
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df_raw = None

    if df_raw is not None:
        st.subheader("Raw Data (first 200 rows)")
        st.dataframe(df_raw.head(200))

        # =======================
        # ADVANCED CLEANING
        # =======================
        if st.button("Run Advanced AI Cleaning"):
            with st.spinner("Running AI-powered cleaning..."):
                try:
                    result = apply_fixes(df_raw)
                    cleaned_df = result['cleaned_df']
                    dup_df_internal = result['duplicates_df']

                    st.success("AI Cleaning Complete!")
                    st.subheader("Cleaned Data (First 200 Rows)")
                    st.dataframe(cleaned_df.head(200))

                    download_button_for_df(cleaned_df, filename="cleaned_data.csv")

                    if not dup_df_internal.empty:
                        st.subheader("Possible Internal Duplicate Entries")
                        st.dataframe(dup_df_internal.head(200))

                except Exception as e:
                    st.error(f"Error during cleaning: {e}")

        # =======================
        # EXTERNAL DUPLICATE DETECTION
        # =======================
        if st.button("Detect Fuzzy Duplicates (Global Scan)"):
            source_df = cleaned_df if cleaned_df is not None else df_raw
            with st.spinner("Detecting fuzzy duplicates..."):
                try:
                    dup_df = fuzzy_duplicate_pairs(source_df, threshold=85, sample_limit=1500)
                    if dup_df.empty:
                        st.info("No fuzzy duplicates found.")
                    else:
                        st.subheader("Fuzzy Duplicate Pairs")
                        st.dataframe(dup_df)

                except Exception as e:
                    st.error(f"Fuzzy detection error: {e}")
else:
    st.warning("Upload a file to begin.")
