import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="CleanChain AI", page_icon="✨")

st.title("✨ CleanChain AI — Smart B2B Data Cleaner")

# Allow both CSV and Excel
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
        with st.spinner("Cleaning your data..."):
            # Basic cleaning
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            st.success("✅ Data cleaned successfully!")
            st.write("### 🧼 Cleaned Data")
            st.dataframe(df.head())

            # Allow download as both CSV and Excel
            csv = df.to_csv(index=False).encode("utf-8")

            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = buffer.getvalue()

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
