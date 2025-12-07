import streamlit as st
import pandas as pd
from io import BytesIO
from ai_services import correct_text_with_ai
from data_sources import get_all_countries, get_all_cities, get_sample_companies

# Streamlit config
st.set_page_config(page_title="CleanChain AI", page_icon="✨")
st.title("✨ CleanChain AI — Smart Global Data Cleaner")

# File upload
uploaded_file = st.file_uploader(
    "📤 Upload your data (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    # Detect file type
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🧾 Original Data")
    st.dataframe(df.head())

    if st.button("✨ Clean and Correct My Data"):
        with st.spinner("Cleaning and correcting your data ..."):

            # 🧼 Basic cleaning
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # Load reference lists
            countries = get_all_countries()
            cities = get_all_cities()
            companies = get_sample_companies()

            # Correction log
            correction_log = []

            def correct_with_log(x, ref_list, col_name):
                if not isinstance(x, str) or not x.strip():
                    return x
                original = x
                corrected = correct_text_with_ai(x)
                # If correction matches a known reference (extra check)
                if corrected.title() in ref_list:
                    result = corrected.title()
                else:
                    result = corrected
                if result != original:
                    correction_log.append({
                        "Column": col_name,
                        "Original": original,
                        "Corrected": result
                    })
                return result

            # Apply corrections
            if "country" in df.columns:
                df["country"] = df["country"].apply(lambda x: correct_with_log(x, countries, "Country"))
            if "city" in df.columns:
                df["city"] = df["city"].apply(lambda x: correct_with_log(x, cities, "City"))
            if "company" in df.columns:
                df["company"] = df["company"].apply(lambda x: correct_with_log(x, companies, "Company"))

            st.success("✅ Data cleaned and AI-corrected successfully!")

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head())

            # Corrections table
            if correction_log:
                st.write("### 🤖 Corrections Applied")
                corrections_df = pd.DataFrame(correction_log)
                st.dataframe(corrections_df)
            else:
                st.info("No AI corrections were required — all names already valid!")

            # 💾 Download buttons
            csv = df.to_csv(index=False).encode("utf-8")
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = buffer.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download CSV", csv, "cleaned_data.csv", "text/csv")
            with col2:
                st.download_button(
                    "📊 Download Excel",
                    excel_data,
                    "cleaned_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
