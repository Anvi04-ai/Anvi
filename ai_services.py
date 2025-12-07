# ai_services.py
import pkg_resources
from symspellpy import SymSpell, Verbosity

# Load the SymSpell model
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

# Add an English frequency dictionary (bundled with the package)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

def correct_text_with_ai(text):
    """
    Offline AI-like text correction using SymSpell.
    Example: 'Imndfia' → 'India'
    """
    if not isinstance(text, str) or text.strip() == "":
        return text

    text = text.strip().title()
    suggestions = sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2)

    if suggestions:
        return suggestions[0].term.title()
    return text
