import os
from spellchecker import SpellChecker
from fuzzywuzzy import fuzz

# Path for custom dictionary file
CUSTOM_WORDS_PATH = "custom_words.txt"

def load_custom_words():
    """Load custom dictionary words that should never be corrected."""
    if os.path.exists(CUSTOM_WORDS_PATH):
        with open(CUSTOM_WORDS_PATH, "r") as f:
            return set(w.strip().lower() for w in f.readlines() if w.strip())
    return set()

def save_custom_word(word):
    """Save a word to the custom dictionary (persistent memory)."""
    with open(CUSTOM_WORDS_PATH, "a") as f:
        f.write(f"{word.lower()}\n")

def hybrid_text_suggestions(text):
    """
    Suggest text corrections using hybrid AI logic.
    Returns a list of (original_word, suggestion, confidence)
    """
    spell = SpellChecker()
    custom_words = load_custom_words()
    suggestions = []

    # Handle non-string input safely
    if not isinstance(text, str):
        text = str(text)

    words = text.split()

    for word in words:
        clean = word.strip(".,!?;:").lower()

        # Skip protected words (emails, numbers, or custom names)
        if clean.isdigit() or "@" in clean or clean in custom_words:
            suggestions.append((word, word, 1.0))
            continue

        # If word is already valid
        if clean in spell:
            suggestions.append((word, word, 1.0))
            continue

        # Use SpellChecker suggestion
        suggestion = spell.correction(clean)
        score = fuzz.ratio(clean, suggestion) / 100 if suggestion else 0

        # Only correct if high confidence
        if suggestion and score >= 0.7:
            suggestions.append((word, suggestion, score))
        else:
            suggestions.append((word, word, 0.5))

    return suggestions


def hybrid_text_clean(text):
    """
    Return corrected text (not suggestions), for direct app use.
    """
    suggestions = hybrid_text_suggestions(text)
    corrected_words = [s[1] for s in suggestions]
    return " ".join(corrected_words)