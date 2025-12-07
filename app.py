import streamlit as st
import pandas as pd
from io import BytesIO

# 🧠 Import helper functions from data_sources.py
from data_sources import (
    get_all_countries,
    get_all_cities,
    get_sample_companies,
    ai_correct_name
)

# ----------------- 🌟 Streamlit UI -----------------
st.set_page_config(page_title="CleanChain AI", page_icon="✨")
st.title("✨ CleanChain AI — Smart Global Data Cleaner")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your data (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Detect file type automatically
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🧾 Original Data")
    st.dataframe(df.head())

    if st.button("✨ Clean and Correct My Data"):
        with st.spinner("Cleaning and AI-correcting your data..."):
            # 🧼 Step 1: Basic cleaning
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # 🧠 Step 2: Load reference data
            countries = get_all_countries()
            cities = get_all_cities()
            companies = get_sample_companies()

            correction_log = []

            # 🧠 Step 3: Define correction function
            def correct_with_log(x, ref_list, col_name):
                if not isinstance(x, str) or not x.strip():
                    return x
                corrected, confidence = ai_correct_name(x, ref_list)
                if corrected != x:
                    correction_log.append({
                        "Column": col_name,
                        "Original": x,
                        "Corrected": corrected,
                        "Confidence": round(confidence * 100, 2)
                    })
                return corrected

            # Apply AI correction where relevant
            if "country" in df.columns:
                df["country"] = df["country"].apply(lambda x: correct_with_log(x, countries, "Country"))
            if "city" in df.columns:
                df["city"] = df["city"].apply(lambda x: correct_with_log(x, cities, "City"))
            if "company" in df.columns:
                df["company"] = df["company"].apply(lambda x: correct_with_log(x, companies, "Company"))

            # ✅ Step 4: Display results
            st.success("✅ Data cleaned and AI-corrected successfully!")

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head())

            if correction_log:
                st.write("### 🤖 Corrections Applied")
                corrections_df = pd.DataFrame(correction_log)
                st.dataframe(corrections_df.style.background_gradient(
                    subset=["Confidence"], cmap="YlGn"
                ))
            else:
                st.info("No AI corrections were required — all names already valid!")

            # 💾 Step 5: Download buttons
            csv = df.to_csv(index=False).encode("utf-8")
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = buffer.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
            with col2:
                st.download_button(
                    "📊 Download Cleaned Excel",
                    excel_data,
                    "cleaned_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
