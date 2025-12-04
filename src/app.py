import streamlit as st
import pandas as pd
from io import BytesIO
from rapidfuzz import process, fuzz  # <-- fuzzy matching for corrections

# ✅ Set page settings
st.set_page_config(page_title="CleanChain AI", page_icon="✨")

st.title("✨ CleanChain AI — Smart B2B Data Cleaner")

# 🧠 Reference data for fuzzy correction (you can expand this anytime)
COUNTRIES = ["India", "China", "Japan", "Dubai", "USA", "Germany", "France", "UK"]
CITIES = ["Pune", "Mumbai", "Delhi", "Bangalore", "Tokyo", "Beijing", "Dubai", "New York"]
COMPANIES = [
    "Microsoft", "Google", "Amazon", "Apple", "IBM", "Intel", "Tesla", "Meta", "Netflix"
]

# 🧩 Correction helper
def correct_with_reference(value, reference_list, threshold=80):
    """Corrects a string using fuzzy matching."""
    if not isinstance(value, str) or value.strip() == "":
        return value
    match, score, _ = process.extractOne(value, reference_list, scorer=fuzz.ratio)
    if score >= threshold:
        return match
    return value

# 📤 File uploader (CSV + Excel)
uploaded_file = st.file_uploader(
    "📤 Upload your data file (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    # Detect file type automatically
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### 🧾 Original Data")
    st.dataframe(df.head())

    if st.button("✨ Clean My Data"):
        with st.spinner("Cleaning and correcting your data..."):
            # 🧽 Basic cleaning
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # 🧠 Smart corrections
            if "country" in df.columns:
                df["country"] = df["country"].apply(lambda x: correct_with_reference(x, COUNTRIES))
            if "city" in df.columns:
                df["city"] = df["city"].apply(lambda x: correct_with_reference(x, CITIES))
            if "company" in df.columns:
                df["company"] = df["company"].apply(lambda x: correct_with_reference(x, COMPANIES))

            st.success("✅ Data cleaned and corrected successfully!")
            st.write("### 🧼 Cleaned Data")
            st.dataframe(df.head())

            # 💾 Prepare downloads
            csv = df.to_csv(index=False).encode("utf-8")

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = buffer.getvalue()

            # ⬇️ Download buttons
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Cleaned CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            with col2:
                st.download_button(
                    "📊 Download Cleaned Excel",
                    data=excel_data,
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
