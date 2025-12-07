import pycountry
import geonamescache
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from functools import lru_cache


# ✅ 1. Countries
def get_all_countries():
    countries = [country.name for country in pycountry.countries]
    return sorted(set(countries))


# ✅ 2. Cities
def get_all_cities():
    gc = geonamescache.GeonamesCache()
    cities = [city_data["name"] for city_data in gc.get_cities().values()]
    return sorted(set(cities))


# ✅ 3. Companies (sample for now)
def get_sample_companies():
    return [
        "Google", "Microsoft", "Amazon", "Apple", "IBM", "Intel",
        "Tesla", "Meta", "Netflix", "Samsung", "Adobe", "Oracle"
    ]


# ✅ 4. Cached Model Loader
@lru_cache(maxsize=1)
def get_model():
    """Load the sentence transformer model once."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ✅ 5. AI correction function
def ai_correct_name(name: str, reference_list: list):
    """
    Use sentence embeddings to find the closest match to the given name
    in the reference list (city, country, or company).
    Returns (corrected_name, confidence_score)
    """
    if not name or not isinstance(name, str):
        return name, 0.0

    model = get_model()

    # Encode inputs
    name_embedding = model.encode(name, convert_to_tensor=True)
    reference_embeddings = model.encode(reference_list, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(name_embedding, reference_embeddings)[0]
    best_match_index = torch.argmax(similarities).item()
    confidence = similarities[best_match_index].item()

    corrected_name = reference_list[best_match_index]
    return corrected_name, confidence


# ✅ For quick test
if __name__ == "__main__":
    print("✅ Loaded", len(get_all_countries()), "countries")
    print("✅ Loaded", len(get_all_cities()), "cities")
    print("✅ Loaded", len(get_sample_companies()), "companies")

    example = "Untied States"
    corrected, score = ai_correct_name(example, get_all_countries())
    print(f"🔍 {example} → {corrected} ({round(score*100,2)}%)")
