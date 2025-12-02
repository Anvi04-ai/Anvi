import streamlit as st
import pandas as pd
import plotly.express as px
from utils.config import OPENAI_API_KEY
from utils.logger import logger
from data_cleaning.profiler import profile_data
from data_cleaning.ai_suggestor import get_ai_suggestions
from data_cleaning.cleaner import apply_cleaning

st.set_page_config(page_title="AI Data Cleaning Platform", layout="wide")
st.title("🚀 Advanced AI Data Cleaning Platform")


# Cache data load
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, low_memory=False)  # Handle large files
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            st.error("Only CSV/Excel supported.")
            st.stop()
    except pd.errors.EmptyDataError:
        st.error("File is empty.")
        st.stop()
    except Exception as e:
        logger.error(f"Load error: {e}")
        st.error(f"Upload failed: {str(e)}")
        st.stop()
        return None


# Sidebar for batch upload (advanced)
st.sidebar.header("Batch Options")
batch_mode = st.sidebar.checkbox("Enable Batch Processing")

uploaded_files = st.file_uploader("Upload files", type=['csv', 'xlsx'], accept_multiple_files=batch_mode)
if uploaded_files:
    dfs = [load_data(f) for f in uploaded_files if (df := load_data(f)) is not None]
    if dfs:
        df = pd.concat(dfs, ignore_index=True) if batch_mode else dfs[0]

        # Profile
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Raw Data")
            st.dataframe(df.head(10))
        with col2:
            profile = profile_data(df)
            st.subheader("Data Profile")
        # Display profile safely
        st.subheader("Data Profile")
        safe_profile = {}
        for k, v in profile.items():
            if hasattr(v, 'item'):  # numpy int64, float64, etc.
                safe_profile[k] = v.item()
            elif isinstance(v, (list, dict, tuple)) and len(v) > 10:
                safe_profile[k] = f"{type(v).__name__} with {len(v)} items (truncated)"
            else:
                safe_profile[k] = v

        st.json(safe_profile)
        # AI Suggestions
        if st.button("🔮 Get AI Cleaning Suggestions"):
            with st.spinner("Analyzing with AI..."):
                suggestions = get_ai_suggestions(df, OPENAI_API_KEY)
                st.session_state.suggestions = suggestions

        if 'suggestions' in st.session_state:
            st.subheader("Select Fixes")
            selected = []
            for i, sug in enumerate(st.session_state.suggestions):
                if st.checkbox(f"☑️ {sug}", key=f"sug_{i}"):
                    selected.append(sug)

            if st.button("🧹 Apply & Visualize") and selected:
                with st.spinner("Cleaning..."):
                    cleaned_df = apply_cleaning(df, selected)

                # Viz
                if len(cleaned_df.columns) >= 2:
                    fig = px.scatter(cleaned_df, x=cleaned_df.columns[0], y=cleaned_df.columns[1],
                                     title="Before/After Scatter")
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("Cleaned Data")
                st.dataframe(cleaned_df.head())

                # Export
                csv = cleaned_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
                logger.info("Export successful")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built for scale | v1.0")