import os
import io
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from rapidfuzz import fuzz, process

# Import your local modules (these are the files we created earlier)
from fixes.apply_fixes import apply_basic_fixes, apply_corrections
from suggestions.suggest import suggest_column_fixes, suggest_row_fixes
from utils.save_log import save_log

# Path to the screenshot you uploaded (developer-provided local path)
EXAMPLE_IMAGE_PATH = "/mnt/data/2025-11-25T16-12-00.812Z.png"


# ---------------------------
# Helper functions
# ---------------------------
def read_uploaded_file(uploaded) -> pd.DataFrame:
    """Read CSV or Excel uploaded file into a DataFrame."""
    file_bytes = uploaded.read()
    try:
        if uploaded.name.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes))
        else:
            return pd.read_excel(io.BytesIO(file_bytes))
    except Exception as exc:
        raise RuntimeError(f"Could not read uploaded file: {exc}")


def fuzzy_duplicate_pairs(df: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """
    Detect fuzzy duplicate pairs using rapidfuzz.
    Returns a DataFrame with columns: row_i, row_j, score
    Note: This is O(n^2) and intended for small-to-medium datasets (n < ~1000).
    """
    records = []
    n = len(df)
    if n < 2:
        return pd.DataFrame(records)

    # Convert each row to a single string for comparison
    row_strings = [" ".join(map(str, df.iloc[i].astype(str).values)) for i in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            score = fuzz.WRatio(row_strings[i], row_strings[j])
            if score >= threshold:
                records.append({"row_i": int(i), "row_j": int(j), "score": int(score)})

    return pd.DataFrame(records)


def download_button_for_df(df: pd.DataFrame, filename: str = "cleaned_data.csv"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label="Download cleaned CSV", data=csv_bytes, file_name=filename, mime="text/csv")


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Data Cleaner", layout="wide")
st.title("🧹 AI Data Cleaner — PyCharm / Streamlit Ready")
st.write("Upload CSV/XLSX, preview, run cleaning, see suggestions, detect fuzzy duplicates, and download results.")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**Example Screenshot**")
    if os.path.exists(EXAMPLE_IMAGE_PATH):
        st.image(EXAMPLE_IMAGE_PATH, use_column_width=True)
    else:
        st.info("Example image not found at expected path.")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df_raw = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Raw Data (first 200 rows)")
    st.dataframe(df_raw.head(200))

    # Action buttons
    st.markdown("---")
    run_clean = st.button("Run Automated Cleaning")
    show_suggestions = st.button("Show Suggestions")
    detect_duplicates_btn = st.button("Detect Fuzzy Duplicates")

    # Optional manual corrections: user can paste a small JSON mapping
    st.markdown("### Optional: Manual corrections (JSON format)")
    corrections_input = st.text_area(
        "Example: {\"city\": {\"mumbay\": \"Mumbai\"}, \"gender\": {\"femlae\": \"female\"}}",
        value="",
        height=80
    )

    cleaned_df = None

    if run_clean:
        with st.spinner("Running cleaning pipeline..."):
            cleaned_df = apply_basic_fixes(df_raw)
            st.success("Cleaning complete")
            st.subheader("Cleaned Data (first 200 rows)")
            st.dataframe(cleaned_df.head(200))

            # Save audit/log
            try:
                details = {
                    "action": "apply_basic_fixes",
                    "rows_before": int(len(df_raw)),
                    "rows_after": int(len(cleaned_df))
                }
                save_path = save_log("clean", details)
                st.write(f"Saved log: `{save_path}`")
            except Exception:
                # don't break UI on logging failure
                st.info("Logging failed (check logs directory permissions).")

            download_button_for_df(cleaned_df, filename="cleaned_data.csv")

    # If user provided corrections JSON, try to apply them
    if corrections_input.strip():
        try:
            import json
            corrections = json.loads(corrections_input)
            if cleaned_df is None:
                # Apply corrections on raw if cleaning not yet run
                target_df = df_raw.copy()
            else:
                target_df = cleaned_df.copy()

            target_df = apply_corrections(target_df, corrections=corrections)
            st.success("Manual corrections applied")
            st.dataframe(target_df.head(200))
            download_button_for_df(target_df, filename="corrected_data.csv")
        except Exception as e:
            st.error(f"Could not parse/apply corrections JSON: {e}")

    if show_suggestions:
        with st.spinner("Generating suggestions..."):
            col_sug = suggest_column_fixes(df_raw)
            row_sug_df = suggest_row_fixes(df_raw)
            st.subheader("Column-level Suggestions")
            st.json(col_sug)
            st.subheader("Row-level Suggestions (first 200 rows of issues)")
            if not row_sug_df.empty:
                st.dataframe(row_sug_df.head(200))
            else:
                st.write("No row-level issues detected")

    if detect_duplicates_btn:
        with st.spinner("Detecting fuzzy duplicates..."):
            # Use cleaned data if available, otherwise raw
            source_df = cleaned_df if cleaned_df is not None else df_raw
            dup_df = fuzzy_duplicate_pairs(source_df, threshold=85)
            st.subheader("Fuzzy Duplicate Pairs")
            if dup_df.empty:
                st.write("No fuzzy duplicate pairs detected with the current threshold.")
            else:
                st.dataframe(dup_df)
                # Offer to show the actual row pairs
                if st.checkbox("Show sample duplicate row pairs"):
                    for _, r in dup_df.head(50).iterrows():
                        i, j = int(r["row_i"]), int(r["row_j"])
                        st.markdown(f"**Pair ({i}, {j}) — score: {r['score']}**")
                        st.write("Row i:")
                        st.write(source_df.iloc[i].to_dict())
                        st.write("Row j:")
                        st.write(source_df.iloc[j].to_dict())
                        st.markdown("---")

    st.markdown("## Quick stats")
    st.write(f"Rows: {len(df_raw)} • Columns: {len(df_raw.columns)}")
    st.write("Column types:")
    st.table(pd.DataFrame(df_raw.dtypes, columns=["dtype"]))
else:
    st.info("Upload a CSV or Excel file to begin cleaning.")
