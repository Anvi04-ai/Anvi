import streamlit as st
import pandas as pd
from io import BytesIO
from ai_correction_engine import correct_entity

# ==================== 🌟 PAGE CONFIG ====================
st.set_page_config(
    page_title="CleanChain AI",
    page_icon="✨",
    layout="wide",
)

# ==================== 🎨 CUSTOM STYLES ====================
st.markdown("""
    <style>
    body {
        background-color: #f8f9fa;
        color: #212529;
    }
    .main {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 25px 40px;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.05);
    }
    h1 {
        color: #5c4dff;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    .stButton button {
        background: linear-gradient(90deg, #6a11cb, #2575fc);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2575fc, #6a11cb);
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #43cea2, #185a9d);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: 0.3s;
    }
    .stDownloadButton button:hover {
        transform: scale(1.03);
        background: linear-gradient(90deg, #185a9d, #43cea2);
    }
    </style>
""", unsafe_allow_html=True)

# ==================== 🌍 HEADER ====================
st.title("✨ CleanChain AI — Fast Global Data Cleaner")
st.caption("🚀 Instantly clean, correct & format your data using AI")

# ==================== 📤 FILE UPLOAD ====================
uploaded_file = st.file_uploader(
    "📤 Upload your data (CSV or Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:
    file_name = uploaded_file.name.lower()
    df = pd.read_csv(uploaded_file) if file_name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.write("### 🧾 Original Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("✨ Clean and Correct My Data"):
        progress = st.progress(0)
        with st.spinner("AI is cleaning your data... ⏳"):

            # Step 1️⃣ Basic Cleanup
            progress.progress(25)
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # Step 2️⃣ AI-Powered Correction (cached for speed)
            progress.progress(60)
            correction_cache = {}

            def correct_fast(value, entity_type):
                if not isinstance(value, str) or not value.strip():
                    return value
                key = (entity_type, value.lower().strip())
                if key in correction_cache:
                    return correction_cache[key]
                corrected, _ = correct_entity(value, entity_type)
                correction_cache[key] = corrected
                return corrected

            for col_name, entity_type in [("country", "country"), ("city", "city"), ("name", "name")]:
                if col_name in df.columns:
                    unique_values = df[col_name].unique()
                    mapping = {v: correct_fast(v, entity_type) for v in unique_values}
                    df[col_name] = df[col_name].map(mapping)

            progress.progress(90)

            # Step 3️⃣ Display Results
            st.success("✅ Data cleaned and AI-corrected successfully!")
            st.balloons()

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Step 4️⃣ Download Options
            progress.progress(100)
            csv_data = df.to_csv(index=False).encode("utf-8")
            excel_buffer = BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="CleanedData")
            excel_data = excel_buffer.getvalue()

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("⬇️ Download CSV", csv_data, "cleaned_data.csv", "text/csv")
            with col2:
                st.download_button(
                    "📊 Download Excel",
                    excel_data,
                    "cleaned_data.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
