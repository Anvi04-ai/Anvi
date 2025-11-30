import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global exception hook (Streamlit hack)
import sys
import streamlit as st
def handle_uncaught(e):
    logger.error(f"Uncaught: {str(e)}")
    st.error("An unexpected error occurred. Check logs.")
    sys.__excepthook__(type(e), e, e.__traceback__)
sys.excepthook = handle_uncaught