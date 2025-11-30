# AI Data Cleaning Platform
Scalable Streamlit app for AI-powered data prep.

## Setup
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. Add `.env`: OPENAI_API_KEY=your_key
4. `streamlit run src/app.py`

## Features
- Upload & profile data
- AI suggestions via PandasAI
- Apply fixes with viz
- Error logs in utils/logger.py

## Scaling
Docker: `docker build -t ai-clean . && docker run -p 8501:8501 ai-clean`