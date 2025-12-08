import pycountry
import geonamescache
from sentence_transformers import SentenceTransformer, util
import torch
from fuzzywuzzy import process
from functools import lru_cache

# ---------------------------
# STEP 1 — Load global data
# ---------------------------

def get_all_countries():
    """Returns list of all countries worldwide."""
    return sorted({c.name for c in pycountry.countries})


def get_all_cities():
    """Returns list of major global cities."""
    gc = geonamescache.GeonamesCache()
    cities = [city["name"] for city in gc.get_cities().values()]
    return sorted(set(cities))


def get_all_names():
    """Returns list of common male/female first names."""
    male_names = [
        "John", "Michael", "David", "James", "Robert", "Daniel", "William",
        "Joseph", "Charles", "Thomas", "George", "Henry", "Richard"
    ]
    female_names = [
        "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Susan",
        "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Anvi", "Gayatri"
    ]
    return sorted(set(male_names + female_names))


# ---------------------------
# STEP 2 — Load AI Model
# ---------------------------

@lru_cache(maxsize=1)
def get_model():
    """Load and cache SentenceTransformer model."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


# ---------------------------
# STEP 3 — AI Correction Logic
# ---------------------------

def ai_correct_name(name: str, ref_list: list, min_confidence: float = 0.45):
    """
    Use transformer embeddings + fuzzy matching to correct misspelled names.
    """
    if not name or not isinstance(name, str):
        return name, 0.0

    name = name.strip().title()
    model = get_model()

    try:
        # Vector-based matching
        name_emb = model.encode(name, convert_to_tensor=True)
        ref_embs = model.encode(ref_list, convert_to_tensor=True)
        sims = util.cos_sim(name_emb, ref_embs)[0]
        best_idx = torch.argmax(sims).item()
        confidence = sims[best_idx].item()
        corrected = ref_list[best_idx]
    except Exception:
        # Fallback to fuzzy if AI model fails
        corrected, confidence = process.extractOne(name, ref_list)

    if confidence < min_confidence:
        return name, confidence
    return corrected, confidence


# ---------------------------
# STEP 4 — Main Correction Function
# ---------------------------

def correct_entity(name: str, entity_type: str):
    """
    Entity-aware correction: 'city', 'country', or 'name'.
    """
    entity_type = entity_type.lower()

    if entity_type == "country":
        ref_list = get_all_countries()
    elif entity_type == "city":
        ref_list = get_all_cities()
    elif entity_type == "name":
        ref_list = get_all_names()
    else:
        return name, 0.0

    return ai_correct_name(name, ref_list)


# ---------------------------
# STEP 5 — Quick Test
# ---------------------------

if __name__ == "__main__":
    print("✅ Countries loaded:", len(get_all_countries()))
    print("✅ Cities loaded:", len(get_all_cities()))
    print("✅ Names loaded:", len(get_all_names()))

    samples = [
        ("Imndfia", "country"),
        ("Untied Stats", "country"),
        ("Mubmai", "city"),
        ("Jonh", "name"),
        ("Lndon", "city"),
    ]

    for text, typ in samples:
        corrected, score = correct_entity(text, typ)
        print(f"🔍 {typ.title()} — {text} → {corrected} ({round(score*100,2)}%)")
