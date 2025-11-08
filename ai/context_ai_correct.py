import pandas as pd
from fuzzywuzzy import process
import re

# ==========================================
# 🔹 Smart AI + Fuzzy Context Correction
# ==========================================

# 🧠 Name protection list (won’t be corrected)
PROTECTED_NAMES = {
    "Anvi", "Drishti", "Rahul", "Aarav", "Priya", "Neha",
    "Riya", "Saanvi", "Aditya", "Rohan", "Sarthak", "Krishna",
    "Amit", "Arjun", "Ishita", "Sneha", "Raj", "Rakesh",
    "Kiran", "Pooja", "Tanvi", "Manish", "Nisha",
    "India", "USA", "Google", "Microsoft", "Apple", "Amazon"
}


def is_protected_name(value):
    """Detect if a word should be protected (e.g., name or brand)."""
    if not isinstance(value, str):
        return False

    clean_val = re.sub(r'[^A-Za-z]', '', value).strip().title()
    return clean_val in PROTECTED_NAMES


def clean_text_with_fuzzy(value, reference_list):
    """Clean text values using fuzzy matching without breaking protected names."""
    try:
        # Skip non-string values
        if not isinstance(value, str):
            return str(value)

        text = value.strip()
        if not text:
            return value

        # Protect short words, numbers, and known names
        if text.isdigit() or len(text) <= 2 or is_protected_name(text):
            return value

        # Skip if it's in proper name format (Title Case)
        if text.istitle() or text.isupper():
            if is_protected_name(text):
                return text

        # Try fuzzy correction
        best_match = process.extractOne(text, reference_list)
        if best_match and best_match[1] > 85:
            corrected = best_match[0].title()
            return corrected

        return text.title()
    except Exception:
        return value


def safe_context_ai_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply hybrid fuzzy + rule-based correction to all text/object columns.
    Avoids numeric/date columns automatically and protects known names.
    """
    df_clean = df.copy()

    for col in df_clean.columns:
        # Process only text/object columns
        if df_clean[col].dtype == object:
            reference_data = df_clean[col].dropna().unique().tolist()
            df_clean[col] = df_clean[col].apply(lambda x: clean_text_with_fuzzy(x, reference_data))
        else:
            continue  # skip numeric/date columns like roll numbers

    return df_clean