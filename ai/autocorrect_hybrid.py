import os
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz

# Path for custom words file
CUSTOM_WORDS_PATH = "custom_words.txt"

def load_custom_words():
    """Load custom dictionary words that should never be corrected."""
    if os.path.exists(CUSTOM_WORDS_PATH):
        with open(CUSTOM_WORDS_PATH, "r") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())
    return set()

def hybrid_text_clean(text):
    """
    Correct text using a hybrid approach:
      1. Skip domain-specific / custom words
      2. SpellChecker for simple typos
      3. Fuzzy matching for tougher errors
    """
    spell = SpellChecker()
    custom_words = load_custom_words()
    words = text.split()
    corrected = []

    for word in words:
        clean = word.strip(".,!?;:").lower()

        # Skip numbers, URLs, and protected words
        if clean.isdigit() or "@" in clean or clean in custom_words:
            corrected.append(word)
            continue

        if clean in spell:
            corrected.append(word)
            continue

        # SpellChecker suggestion
        suggestion = spell.correction(clean)

        # Fuzzy similarity fallback
        if suggestion and fuzz.ratio(clean, suggestion) < 70:
            candidates = spell.candidates(clean)
            if candidates:
                suggestion = max(candidates, key=lambda c: fuzz.ratio(clean, c))

        if suggestion and fuzz.ratio(clean, suggestion) >= 70:
            new_word = suggestion.capitalize() if word[0].isupper() else suggestion
            corrected.append(new_word)
        else:
            corrected.append(word)

    return " ".join(corrected)