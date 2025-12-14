import streamlit as st
import pandas as pd
from io import BytesIO
from ai_correction_engine import correct_entity
from global_cleaning import (
    detect_and_translate,
    normalize_time,
    convert_to_usd,
)

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

    # 🌍 Global Cleaning Options
    st.write("### 🌍 Global Cleaning Options")
    apply_translation = st.checkbox("🌐 Detect language & translate to English", value=True)
    apply_currency = st.checkbox("💱 Convert all amounts to USD (if currency column exists)", value=False)
    apply_timezone = st.checkbox("🕒 Normalize time columns to UTC", value=False)
    fast_mode = st.checkbox("⚡ Fast Mode (Skip slow AI translations & repeated corrections)", value=True)

    if st.button("✨ Clean and Correct My Data"):
        with st.spinner("AI is cleaning and correcting your data... ⏳"):

            # 🧹 1️⃣ Basic text cleanup
            df.columns = df.columns.str.lower().str.strip()
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            # 🧠 2️⃣ AI-powered correction (cached)
            correction_log = []
            correction_cache = {}

            def correct_with_log(value, entity_type):
                if not isinstance(value, str) or not value.strip():
                    return value

                key = (entity_type, value.lower().strip())
                if key in correction_cache:
                    return correction_cache[key]

                corrected, confidence = correct_entity(value, entity_type)
                correction_cache[key] = corrected

                if corrected != value:
                    correction_log.append({
                        "Type": entity_type.title(),
                        "Original": value,
                        "Corrected": corrected,
                        "Confidence": round(confidence * 100, 2)
                    })
                return corrected

            # Apply AI correction (only once per unique)
            for col_name, entity_type in [("country", "country"), ("city", "city"), ("name", "name")]:
                if col_name in df.columns:
                    unique_values = df[col_name].unique()
                    mapping = {v: correct_with_log(v, entity_type) for v in unique_values}
                    df[col_name] = df[col_name].map(mapping)

            # 🌐 3️⃣ Global cleaning with caching
            if apply_translation and not fast_mode:
                st.info("Translating non-English text to English (may take longer)...")
                translation_cache = {}

                def cached_translate(text):
                    if not isinstance(text, str) or not text.strip():
                        return text
                    if text in translation_cache:
                        return translation_cache[text]
                    translated = detect_and_translate(text)
                    translation_cache[text] = translated
                    return translated

                df = df.applymap(cached_translate)
            elif apply_translation:
                st.info("⚡ Fast mode enabled — skipping translation for faster performance")

            # 💱 Currency conversion
            if apply_currency and "amount" in df.columns and "currency" in df.columns:
                df["amount_usd"] = df.apply(
                    lambda x: convert_to_usd(x["amount"], x["currency"]), axis=1
                )

            # 🕒 Time normalization
            if apply_timezone:
                datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
                for col in datetime_cols:
                    df[col] = df[col].apply(normalize_time)

            st.success("✅ Data cleaned and AI-corrected successfully!")

            # 🧼 4️⃣ Display results
            st.write("### 🧼 Cleaned Data Preview")
            st.dataframe(df.head())

            if correction_log:
                st.write("### 🤖 AI Corrections Applied")
                corrections_df = pd.DataFrame(correction_log)
                st.dataframe(corrections_df)
            else:
                st.info("No major corrections detected — your data was already clean!")

            # 💾 5️⃣ Download options
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
