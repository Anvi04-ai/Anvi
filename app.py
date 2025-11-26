import streamlit as st
import pandas as pd
import json
from fixes.apply_fixes import apply_fixes
from detection.detect import fuzzy_duplicate_pairs
from utils.save_log import save_log

# --------------------------
# Helper functions (ADD HERE)
# --------------------------

def download_button_for_df(df, filename="output.csv"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download File",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def run_clean():
    return st.button("Run Cleaning", key="run_clean")

def detect_duplicates_btn():
    return st.button("Detect Duplicates", key="dup_btn")

# --- Upload Section ---
st.title("AI Data Cleaner ")

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

        # Run cleaning button
        if st.button("Run Automated Cleaning"):
            if df_raw.empty:
                st.warning("Uploaded file is empty. Cannot run cleaning.")
            else:
                try:
                    # Safe apply_fixes call
                    cleaned_df = apply_fixes(df_raw)
                    st.success("Cleaning complete")
                    st.dataframe(cleaned_df.head(200))
                    download_button_for_df(cleaned_df, filename="cleaned_data.csv")

                    # Optional: save log
                    try:
                        save_path = save_log("clean", {"rows_before": len(df_raw), "rows_after": len(cleaned_df)})
                        st.write(f"Saved log: `{save_path}`")
                    except Exception:
                        st.info("Logging failed (check logs dir).")
                except Exception as e:
                    st.error(f"Error during cleaning: {e}")

        # Fuzzy duplicate button
        if st.button("Detect Fuzzy Duplicates"):
            source_df = cleaned_df if cleaned_df is not None else df_raw
            with st.spinner("Detecting fuzzy duplicates..."):
                try:
                    dup_df = fuzzy_duplicate_pairs(source_df, threshold=85, sample_limit=1500)
                    if dup_df.empty:
                        st.info("No fuzzy duplicates found with the current threshold.")
                    else:
                        st.subheader("Fuzzy Duplicate Pairs")
                        st.dataframe(dup_df)
                        if st.checkbox("Show sample duplicate row pairs"):
                            for _, r in dup_df.head(50).iterrows():
                                i, j = int(r['row_i']), int(r['row_j'])
                                st.markdown(f"**Pair ({i}, {j}) — score: {r['score']}**")
                                st.write("Row i:")
                                st.write(source_df.iloc[i].to_dict())
                                st.write("Row j:")
                                st.write(source_df.iloc[j].to_dict())
                                st.markdown("---")
                except Exception as e:
                    st.error(f"Fuzzy detection error: {e}")
else:
    st.warning("Upload a file to begin.")

