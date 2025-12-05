import pycountry
import geonamescache
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# ============================
# 🌍 GLOBAL REFERENCE LOADERS
# ============================

# 🌍 COUNTRIES
def get_all_countries():
    countries = [country.name for country in pycountry.countries]
    return sorted(countries)

# 🏙 CITIES
def get_all_cities():
    gc = geonamescache.GeonamesCache()
    cities = [city_data["name"] for city_data in gc.get_cities().values()]
    return sorted(set(cities))

# 🏢 COMPANIES (sample list; can later link OpenCorporates API)
def get_sample_companies():
    return [
        "Google", "Microsoft", "Amazon", "Apple", "IBM", "Intel",
        "Tesla", "Meta", "Netflix", "Samsung"
    ]

# =================================
# 🤖 AI NAME CORRECTION ENGINE
# =================================

# Load a small, fast embedding model (works on CPU)
model = SentenceTransformer("all-MiniLM-L6-v2")

def ai_correct_name(name, reference_list, threshold=0.75):
    """
    Correct a given name using semantic similarity (embeddings).
    name: the input string to correct
    reference_list: list of valid names (countries, cities, etc.)
    threshold: minimum similarity (0–1) for correction
    """
    if not isinstance(name, str) or not name.strip():
        return name

    # Encode input and references into embeddings
    emb_name = model.encode(name, convert_to_tensor=True)
    emb_refs = model.encode(reference_list, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(emb_name, emb_refs)[0]
    best_idx = torch.argmax(similarities).item()
    best_score = similarities[best_idx].item()

    # Return best match if similarity is strong enough
    if best_score >= threshold:
        return reference_list[best_idx]
    return name


# ============================
# 🧠 COMBINED TEST (optional)
# ============================

if __name__ == "__main__":
    countries = get_all_countries()
    cities = get_all_cities()
    companies = get_sample_companies()

    print(f"✅ Loaded {len(countries)} countries")
    print(f"✅ Loaded {len(cities)} cities")
    print(f"✅ Loaded {len(companies)} companies")

    # Save as CSVs (optional, useful for offline reference)
    pd.DataFrame(countries, columns=["Country"]).to_csv("countries.csv", index=False)
    pd.DataFrame(cities, columns=["City"]).to_csv("cities.csv", index=False)
    pd.DataFrame(companies, columns=["Company"]).to_csv("companies.csv", index=False)

    # Test the AI correction engine
    print("\n🔍 AI Name Correction Tests:")
    print("Inda ➜", ai_correct_name("Inda", countries))
    print("Garmany ➜", ai_correct_name("Garmany", countries))
    print("Unted Sttes ➜", ai_correct_name("Unted Sttes", countries))
    print("Nwe Yrok ➜", ai_correct_name("Nwe Yrok", cities))
    print("Microsfot ➜", ai_correct_name("Microsfot", companies))
