import streamlit as st
import pandas as pd
from io import BytesIO
import requests

# ==================== ☁️ CLOUDFLARE AI CALL ====================
def correct_entity_cloudflare(name):
    try:
        response = requests.post(
            "https://ai-name-corrector.YOUR-NAME.workers.dev/clean",  # 🔁 replace YOUR-NAME
            json={"name": name},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("cleaned_name", name)
        else:
            return name
    except Exception as e:
        print("Error:", e)
        return name


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
}
.stDownloadButton button {
    background: linear-gradient(90deg, #43cea2, #185a9d);
    color: white;
    border-radius: 6px;
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

            # ==================== 1️⃣ BASIC CLEANUP ====================
            progress.progress(25)
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # ==================== 2️⃣ AI NAME CORRECTION ====================
            progress.progress(60)
            correction_cache = {}

            def correct_fast(value):
                if not isinstance(value, str) or not value.strip():
                    return value

                key = value.lower().strip()
                if key in correction_cache:
                    return correction_cache[key]

                corrected = correct_entity_cloudflare(value)
                correction_cache[key] = corrected
                return corrected

            # Apply AI correction ONLY to name column
            if "name" in df.columns:
                unique_names = df["name"].unique()
                mapping = {v: correct_fast(v) for v in unique_names}
                df["name"] = df["name"].map(mapping)

            progress.progress(90)

            # ==================== 3️⃣ RESULTS ====================
            st.success("✅ Data cleaned and AI-corrected successfully!")
            st.balloons()

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # ==================== 4️⃣ DOWNLOAD ====================
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
