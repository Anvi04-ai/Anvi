import pandas as pd
from fuzzywuzzy import process
import re

# ==========================================
# 🔹 Smart AI + Fuzzy Context Correction
# ==========================================

PROTECTED_NAMES = {
    "Anvi", "Drishti", "Rahul", "Aarav", "Priya", "Neha", "Riya",
    "Saanvi", "Aditya", "Rohan", "Sarthak", "Krishna", "Amit",
    "Arjun", "Ishita", "Sneha", "Raj", "Rakesh", "Kiran", "Pooja",
    "Tanvi", "Manish", "Nisha", "India", "USA", "Google", "Microsoft",
    "Apple", "Amazon"
}

COMMON_FIXES = {
    "counrty": "Country",
    "adress": "Address",
    "roll no": "Roll Number",
    "rollno": "Roll Number",
    "naem": "Name",
    "studnt": "Student",
    "collge": "College",
    "brnach": "Branch",
    "technlogy": "Technology",
    "departmnt": "Department"
}


def is_protected_name(value):
    if not isinstance(value, str):
        return False
    clean_val = re.sub(r'[^A-Za-z]', '', value).strip().title()
    return clean_val in PROTECTED_NAMES


def apply_common_fixes(text):
    lower = text.lower().strip()
    if lower in COMMON_FIXES:
        return COMMON_FIXES[lower]
    return text


def clean_text_with_fuzzy(value, reference_list):
    try:
        if not isinstance(value, str):
            return str(value)
        text = value.strip()
        if not text:
            return value

        # Skip numbers or short codes
        if text.isdigit() or len(text) <= 2:
            return value

        # Protect proper names
        if is_protected_name(text):
            return text

        # Apply known corrections
        fixed = apply_common_fixes(text)
        if fixed != text:
            return fixed

        # Fuzzy match correction
        best_match = process.extractOne(text, reference_list)
        if best_match and best_match[1] > 85:
            return best_match[0].title()

        return text.title()
    except Exception:
        return value


def safe_context_ai_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Applies hybrid AI correction on all text columns, skipping numeric/date ones."""
    df_clean = df.copy()

    for col in df_clean.columns:
        if df_clean[col].dtype == object:
            ref = df_clean[col].dropna().unique().tolist()
            df_clean[col] = df_clean[col].apply(lambda x: clean_text_with_fuzzy(x, ref))
        else:
            continue
    return df_clean