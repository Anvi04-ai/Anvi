import os, json, time, requests
import pandas as pd
from rapidfuzz import process, fuzz

# ------------------------------------------------------------
# 📂 PATHS AND GLOBALS
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

MAPPINGS_PATH = os.path.join(DATA_DIR, "user_mappings.json")
WHITELIST_PATH = os.path.join(DATA_DIR, "whitelist.txt")
CHANGELOG_PATH = os.path.join(DATA_DIR, "change_log.json")

PDL_API_KEY = os.environ.get("PDL_API_KEY", "<PUT_YOUR_KEY_HERE>")
PDL_URL = "https://api.peopledatalabs.com/cleaner"

# ------------------------------------------------------------
# 📚 LOAD GLOBAL REFERENCE DATASETS
# ------------------------------------------------------------
def load_reference_data():
    names, cities, countries = [], [], []
    try:
        names = pd.read_csv(os.path.join(DATA_DIR, "name_gender.csv"))["name"].dropna().astype(str).tolist()
    except: pass
    try:
        cities = pd.read_csv(os.path.join(DATA_DIR, "world_cities.csv"))["city"].dropna().astype(str).tolist()
    except: pass
    try:
        countries = pd.read_csv(os.path.join(DATA_DIR, "countries.csv"))["country"].dropna().astype(str).tolist()
    except: pass
    return names, cities, countries

KNOWN_NAMES, KNOWN_CITIES, KNOWN_COUNTRIES = load_reference_data()

# ------------------------------------------------------------
# ⚙️ JSON & STORAGE HELPERS
# ------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            return json.load(open(path, "r", encoding="utf-8"))
        except:
            return default
    return default

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

def append_changelog(entry):
    log = load_json(CHANGELOG_PATH, [])
    log.append(entry)
    save_json(CHANGELOG_PATH, log)

# ------------------------------------------------------------
# 🧾 USER MAPPINGS & WHITELIST
# ------------------------------------------------------------
def load_whitelist():
    if not os.path.exists(WHITELIST_PATH):
        return set()
    with open(WHITELIST_PATH, "r", encoding="utf-8") as f:
        return set(l.strip() for l in f.readlines() if l.strip())

def save_whitelist_items(items):
    os.makedirs(os.path.dirname(WHITELIST_PATH), exist_ok=True)
    with open(WHITELIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(items)))

def load_user_mappings():
    return load_json(MAPPINGS_PATH, {})

def save_user_mapping(src, target):
    m = load_user_mappings()
    m[src.lower()] = target
    save_json(MAPPINGS_PATH, m)

# ------------------------------------------------------------
# 🌍 PEOPLE DATA LABS (AI NORMALIZER)
# ------------------------------------------------------------
def call_pdl_clean(value, field_hint=None):
    if value is None:
        return None
    params = {"api_key": PDL_API_KEY, "value": str(value)}
    if field_hint:
        params["type"] = field_hint
    try:
        r = requests.get(PDL_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        normalized = data.get("cleaned") or data.get("result") or data.get("value")
        if isinstance(normalized, dict):
            return normalized.get("name") or normalized.get("value")
        return normalized
    except:
        return None

# ------------------------------------------------------------
# 🔍 FUZZY MATCHING
# ------------------------------------------------------------
def fuzzy_fallback(value, candidates, threshold=88):
    if not value or not candidates:
        return None
    best = process.extractOne(str(value), candidates, scorer=fuzz.WRatio)
    if best and best[1] >= threshold:
        return best[0]
    return None

# ------------------------------------------------------------
# 🧠 IDENTIFIER DETECTION (protect roll no, phone, etc.)
# ------------------------------------------------------------
def is_identifier_column(df, col):
    name = col.lower()
    keywords = ["roll", "id", "emp", "phone", "mobile", "adhar", "aadhar", "ssn"]
    if any(k in name for k in keywords):
        return True
    s = df[col].dropna().astype(str)
    numeric_frac = (s.str.isnumeric()).mean() if len(s)>0 else 0
    avg_len = s.str.len().mean() if len(s)>0 else 0
    return numeric_frac > 0.8 and avg_len < 12

# ------------------------------------------------------------
# 🧩 HYBRID AI + CONTEXT CORRECTION PIPELINE
# ------------------------------------------------------------
def suggest_corrections_for_value(value, col_hint=None):
    v = "" if value is None else str(value).strip()
    if v == "":
        return None, None

    # 1️⃣ User-defined mapping
    mappings = load_user_mappings()
    if v.lower() in mappings:
        return mappings[v.lower()], "user"

    # 2️⃣ PDL normalization
    if col_hint in ("country", "city", "company", "organization"):
        p = call_pdl_clean(v, field_hint=col_hint)
        if p and p.lower() != v.lower():
            return p, "pdl"

    # 3️⃣ Fuzzy fallback
    ref_list = []
    if col_hint in ("name", "person", "student"):
        ref_list = KNOWN_NAMES
    elif col_hint == "city":
        ref_list = KNOWN_CITIES
    elif col_hint == "country":
        ref_list = KNOWN_COUNTRIES

    f = fuzzy_fallback(v, ref_list)
    if f and f.lower() != v.lower():
        return f, "fuzzy"

    return None, None

# ------------------------------------------------------------
# 🧼 MAIN CLEANING FUNCTION
# ------------------------------------------------------------
def clean_dataframe(df):
    df = df.copy()
    whitelist = load_whitelist()

    for col in df.select_dtypes(include="object").columns:
        if is_identifier_column(df, col):
            continue

        corrected = []
        for val in df[col]:
            val_str = str(val).strip() if val is not None else ""
            if val_str.lower() in whitelist:
                corrected.append(val)
                continue

            suggestion, source = suggest_corrections_for_value(val_str, col_hint=col.lower())

            if suggestion and suggestion != val_str:
                append_changelog({
                    "column": col,
                    "original": val_str,
                    "corrected": suggestion,
                    "method": source,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                corrected.append(suggestion)
            else:
                corrected.append(val_str)

        df[col] = corrected

    return df