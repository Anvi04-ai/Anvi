import streamlit as st
import pandas as pd
from io import BytesIO

# Import the real function you have
from cleaning import clean_record


# Load city & country lists
@st.cache_data
def load_city_list():
    try:
        df = pd.read_csv("world_cities.csv")
        return df["city"].dropna().unique().tolist()
    except:
        return []

@st.cache_data
def load_country_list():
    try:
        df = pd.read_csv("countries.csv")
        return df["country"].dropna().unique().tolist()
    except:
        return []


city_list = load_city_list()
country_list = load_country_list()


# -----------------------------------------
# Streamlit UI
# -----------------------------------------
st.set_page_config(page_title="AI Data Cleaning", page_icon="🧹", layout="wide")

st.title("🧹 AI Data Cleaning App")
st.write("Upload your dataset and get AI-based name & spelling corrections instantly!")


uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, dtype=str)
        else:
            df = pd.read_excel(uploaded_file, dtype=str)

        st.success("📁 File uploaded successfully!")
        st.subheader("🔍 Original Data Preview")
        st.dataframe(df.head())

        # Clean record-by-record
        cleaned_rows = []
        for _, row in df.iterrows():
            cleaned_rows.append(clean_record(row.to_dict(), city_list, country_list))

        cleaned_df = pd.DataFrame(cleaned_rows)

        st.subheader("✨ Cleaned Data Preview")
        st.dataframe(cleaned_df.head())

        # Download cleaned CSV
        csv_data = cleaned_df.to_csv(index=False).encode()
        st.download_button("📥 Download Cleaned CSV", csv_data, "cleaned_data.csv")

        # Download cleaned Excel
        excel_buffer = BytesIO()
        cleaned_df.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)

        st.download_button(
            "📘 Download Cleaned Excel",
            excel_buffer,
            file_name="cleaned_data.xlsx"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")