from sentence_transformers import SentenceTransformer, util
import pycountry
import geonamescache

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load data
gc = geonamescache.GeonamesCache()
ALL_COUNTRIES = [c.name for c in pycountry.countries]
ALL_CITIES = [c["name"] for c in gc.get_cities().values()]

def ai_correct_name(name: str, ref_list, threshold=0.7):
    """
    Correct a string using semantic similarity within its domain (cities, countries, etc.)
    """
    if not isinstance(name, str) or not name.strip():
        return name, 1.0

    name = name.strip().title()
    name_emb = model.encode(name, convert_to_tensor=True)
    ref_embs = model.encode(ref_list, convert_to_tensor=True)

    # Compute similarity
    cosine_scores = util.cos_sim(name_emb, ref_embs)[0]
    best_idx = cosine_scores.argmax().item()
    best_score = cosine_scores[best_idx].item()

    if best_score > threshold:
        return ref_list[best_idx], best_score
    else:
        return name, best_score
