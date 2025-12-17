import streamlit as st
import pandas as pd
from io import BytesIO
import requests

# ==================== ☁️ CLOUDFLARE AI CLEANER ====================
CLOUDFLARE_API = "https://ai-name-corrector.anvi-ai.workers.dev/"  # 🔁 Replace YOUR-NAME

def correct_entity_cloudflare(name: str):
    """
    Sends the name to your Cloudflare Worker for cleaning/correction.
    Falls back gracefully if the Worker is unreachable.
    """
    if not isinstance(name, str) or not name.strip():
        return name

    try:
        response = requests.post(
            CLOUDFLARE_API,
            json={"name": name},
            timeout=8
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("cleaned_name", name)
    except Exception as e:
        print("⚠️ Cloudflare Worker error:", e)
    return name  # fallback


# ==================== 🌟 PAGE CONFIG ====================
st.set_page_config(
    page_title="CleanChain AI",
    page_icon="✨",
    layout="wide",
)

# ==================== 🎨 STYLES ====================
st.markdown("""
<style>
body {background-color: #f8f9fa; color: #212529;}
.main {background-color: #ffffff; border-radius: 12px; padding: 25px 40px; box-shadow: 0px 4px 20px rgba(0,0,0,0.05);}
h1 {color: #5c4dff; text-align: center; font-family: 'Poppins', sans-serif; font-weight: 700;}
.stButton button {
    background: linear-gradient(90deg, #6a11cb, #2575fc);
    color: white; font-weight: 600; border: none; border-radius: 8px;
    padding: 0.6rem 1.2rem; transition: all 0.3s ease-in-out;
}
.stButton button:hover {transform: scale(1.05);}
.stDownloadButton button {
    background: linear-gradient(90deg, #43cea2, #185a9d);
    color: white; border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ==================== 🧠 HEADER ====================
st.title("✨ CleanChain AI — Smart Data Cleaner")
st.caption("🚀 Instantly clean, correct & format your data using Cloudflare AI")

# ==================== 📤 FILE UPLOAD ====================
uploaded_file = st.file_uploader("📤 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    file_name = uploaded_file.name.lower()
    df = pd.read_csv(uploaded_file) if file_name.endswith(".csv") else pd.read_excel(uploaded_file)

    st.write("### 🧾 Original Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("✨ Clean & Correct Data"):
        progress = st.progress(0)
        with st.spinner("AI is cleaning your data... ⏳"):

            # Step 1️⃣ Normalize Data
            progress.progress(20)
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # Step 2️⃣ AI Correction via Cloudflare Worker
            progress.progress(60)
            if "name" in df.columns:
                cache = {}
                def correct_name(n):
                    n_stripped = n.strip().lower()
                    if n_stripped in cache:
                        return cache[n_stripped]
                    cleaned = correct_entity_cloudflare(n)
                    cache[n_stripped] = cleaned
                    return cleaned

                df["name"] = df["name"].apply(correct_name)

            # Step 3️⃣ Finalize
            progress.progress(90)
            st.success("✅ AI Cleaning Complete!")
            st.balloons()

            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            # Step 4️⃣ Downloads
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
