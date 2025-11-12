import os
import pandas as pd
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz
from langdetect import detect

# ===============================
# 🔹 Load Custom & Global Data
# ===============================
CUSTOM_WORDS_PATH = "custom_words.txt"
DATA_PATH = "data"

def load_custom_words():
    """Load words that should never be corrected (user-defined)."""
    if os.path.exists(CUSTOM_WORDS_PATH):
        with open(CUSTOM_WORDS_PATH, "r") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())
    return set()

def load_global_dictionaries():
    """Load known names, cities, and countries from data folder."""
    names, cities, countries = set(), set(), set()
    try:
        names = set(pd.read_csv(os.path.join(DATA_PATH, "name_gender.csv"))["name"].str.lower())
    except Exception:
        pass
    try:
        cities = set(pd.read_csv(os.path.join(DATA_PATH, "world_cities.csv"))["city"].str.lower())
    except Exception:
        pass
    try:
        countries = set(pd.read_csv(os.path.join(DATA_PATH, "countries.csv"))["country"].str.lower())
    except Exception:
        pass
    return names, cities, countries

# Load datasets globally
KNOWN_NAMES, KNOWN_CITIES, KNOWN_COUNTRIES = load_global_dictionaries()

# ===============================
# 🔹 Save Custom Word
# ===============================
def save_custom_word(word):
    """Save a word to custom dictionary for future use."""
    with open(CUSTOM_WORDS_PATH, "a") as f:
        f.write(f"{word.lower()}\n")

# ===============================
# 🔹 Core Hybrid Logic
# ===============================
def hybrid_text_suggestions(text):
    """
    Suggest text corrections using hybrid AI + fuzzy logic + language detection.
    Returns a list of (original_word, suggestion, confidence).
    """
    spell = SpellChecker()
    custom_words = load_custom_words()
    suggestions = []

    # Handle None or non-string safely
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Try to detect language
    try:
        lang = detect(text)
    except:
        lang = "en"

    words = text.split()

    for word in words:
        clean = word.strip(".,!?;:").lower()

        # Skip special/custom/known/global words
        if (
            clean.isdigit()
            or "@" in clean
            or clean in custom_words
            or clean in KNOWN_NAMES
            or clean in KNOWN_CITIES
            or clean in KNOWN_COUNTRIES
            or word.istitle()
            or word.isupper()
        ):
            suggestions.append((word, word, 1.0))
            continue

        # Already valid word
        if clean in spell:
            suggestions.append((word, word, 1.0))
            continue

        # AI Spell Correction
        suggestion = spell.correction(clean)
        score = fuzz.ratio(clean, suggestion) / 100 if suggestion else 0

        # Fuzzy fallback (if low confidence)
        if suggestion is None or score < 0.7:
            candidates = spell.candidates(clean)
            if candidates:
                suggestion = max(candidates, key=lambda c: fuzz.ratio(clean, c))
                score = fuzz.ratio(clean, suggestion) / 100
            else:
                suggestion = word
                score = 0.5

        suggestions.append((word, suggestion, score))

    return suggestions

# ===============================
# 🔹 Return Corrected Text
# ===============================
def hybrid_text_clean(text):
    """Return directly corrected text for app display."""
    suggestions = hybrid_text_suggestions(text)
    corrected_words = [s[1] for s in suggestions]
    return " ".join(corrected_words)