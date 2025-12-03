import streamlit as st
import pandas as pd

st.set_page_config(page_title="CleanChain AI", page_icon="✨")

st.title("✨ CleanChain AI — Smart B2B Data Cleaner")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Original Data")
    st.dataframe(df.head())

    if st.button("Clean My Data"):
        with st.spinner("Cleaning your data..."):
            # Simple cleaning logic
            df = df.applymap(lambda x: x.strip().title() if isinstance(x, str) else x)
            df = df.drop_duplicates()

            st.success("✅ Data cleaned successfully!")
            st.write("### Cleaned Data")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Cleaned CSV", data=csv, file_name="cleaned_data.csv")
