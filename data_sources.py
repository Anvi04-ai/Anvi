import pycountry
import geonamescache
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache
from fuzzywuzzy import process


# ✅ 1. Countries
def get_all_countries():
    """Return a sorted list of all country names."""
    countries = [country.name for country in pycountry.countries]
    return sorted(set(countries))


# ✅ 2. Cities
def get_all_cities():
    """Return a sorted list of world cities using GeoNames."""
    gc = geonamescache.GeonamesCache()
    cities = [city_data["name"] for city_data in gc.get_cities().values()]
    return sorted(set(cities))


# ✅ 3. Companies (sample for now)
def get_sample_companies():
    """Return a small list of known global companies."""
    return [
        "Google", "Microsoft", "Amazon", "Apple", "IBM", "Intel",
        "Tesla", "Meta", "Netflix", "Samsung", "Adobe", "Oracle"
    ]


# ✅ 4. Cached Model Loader
@lru_cache(maxsize=1)
def get_model():
    """Load the sentence transformer model once and reuse it."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ✅ 5. Advanced AI Name Correction
def ai_correct_name(name: str, reference_list: list, min_confidence: float = 0.45):
    """
    AI-based correction for names, cities, and countries.
    Uses sentence embeddings first, and fuzzy matching as a backup.
    Example:
        'Imndfia' → 'India'
        'Untied States' → 'United States'
    """
    if not name or not isinstance(name, str):
        return name, 0.0

    model = get_model()
    name_embedding = model.encode(name, convert_to_tensor=True)
    reference_embeddings = model.encode(reference_list, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(name_embedding, reference_embeddings)[0]
    best_match_index = torch.argmax(similarities).item()
    confidence = similarities[best_match_index].item()
    corrected_name = reference_list[best_match_index]

    # 🧠 Fallback to fuzzy match if confidence too low
    if confidence < min_confidence:
        match, score = process.extractOne(name, reference_list)
        if score > 80:
            return match, score / 100
        return name, confidence

    return corrected_name, confidence


# ✅ Quick self-test
if __name__ == "__main__":
    print("✅ Loaded", len(get_all_countries()), "countries")
    print("✅ Loaded", len(get_all_cities()), "cities")
    print("✅ Loaded", len(get_sample_companies()), "companies")

    tests = ["Imndfia", "Untied States", "Mmbai", "Gogle", "Dubia"]
    for t in tests:
        corrected, score = ai_correct_name(t, get_all_countries() + get_all_cities() + get_sample_companies())
        print(f"🔍 {t} → {corrected} ({round(score*100,2)}%)")
