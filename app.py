import streamlit as st
import pandas as pd
from io import BytesIO
from ai_correction_engine import correct_entity

# 🎨 Page setup
st.set_page_config(page_title="CleanChain AI", page_icon="✨")
st.title("✨ CleanChain AI — Smart Global Data Cleaner")

# 📤 File upload
uploaded_file = st.file_uploader(
    "📤 Upload your data (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

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
        with st.spinner("AI is cleaning and correcting your data... ⏳"):

            # 1️⃣ Basic text cleanup
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # 2️⃣ AI-powered correction
            correction_log = []

            def correct_with_log(value, entity_type):
                if not isinstance(value, str) or not value.strip():
                    return value
                corrected, confidence = correct_entity(value, entity_type)
                if corrected != value:
                    correction_log.append({
                        "Type": entity_type.title(),
                        "Original": value,
                        "Corrected": corrected,
                        "Confidence": round(confidence * 100, 2)
                    })
                return corrected

            # Apply AI correction based on column names
            if "country" in df.columns:
                df["country"] = df["country"].apply(lambda x: correct_with_log(x, "country"))

            if "city" in df.columns:
                df["city"] = df["city"].apply(lambda x: correct_with_log(x, "city"))

            if "name" in df.columns:
                df["name"] = df["name"].apply(lambda x: correct_with_log(x, "name"))

            st.success("✅ Data cleaned and AI-corrected successfully!")

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head())

            # Show corrections log
            if correction_log:
                st.write("### 🤖 AI Corrections Applied")
                corrections_df = pd.DataFrame(correction_log)
                st.dataframe(corrections_df)
            else:
                st.info("No major corrections needed — your data was already clean!")

            # 3️⃣ Download options
            csv_data = df.to_csv(index=False).encode("utf-8")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = excel_buffer.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download CSV", csv_data, "cleaned_data.csv", "text/csv")
            with col2:
                st.download_button("📊 Download Excel", excel_data,
                                   "cleaned_data.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
