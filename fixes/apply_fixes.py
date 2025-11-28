# fixes/apply_fixes.py
import os
import pandas as pd
from typing import Optional, Dict, List
from nameparser import HumanName
from rapidfuzz import process, fuzz

# ---------- Config: paths to your CSVs (relative to package root) ----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project root / fixes/..
COUNTRIES_PATH = os.path.join(BASE_DIR, "data", "countries.csv")
NAMES_PATH = os.path.join(BASE_DIR, "data", "name_gender.csv")
CITIES_PATH = os.path.join(BASE_DIR, "data", "world_cities.csv")

# ---------- Helpers to load reference lists robustly ----------
def _load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8", engine="python")
        except Exception:
            return None

def _choose_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Choose first column name in df that matches any candidate (case-insensitive contains or exact).
    """
    if df is None:
        return None
    cols = list(df.columns)
    cols_lower = [c.lower() for c in cols]
    for cand in candidates:
        cand_l = cand.lower()
        # exact
        for i, c in enumerate(cols_lower):
            if c == cand_l:
                return cols[i]
        # contains
        for i, c in enumerate(cols_lower):
            if cand_l in c:
                return cols[i]
    return None

def build_reference_sets():
    countries_df = _load_csv_safe(COUNTRIES_PATH)
    names_df = _load_csv_safe(NAMES_PATH)
    cities_df = _load_csv_safe(CITIES_PATH)

    # Countries: common header names
    country_col = _choose_column(countries_df, ["name", "country", "country_name", "C1", "Name"]) if countries_df is not None else None
    country_list = []
    if country_col and countries_df is not None:
        country_list = countries_df[country_col].dropna().astype(str).str.strip().unique().tolist()

    # Names: header often 'name'
    name_col = _choose_column(names_df, ["name", "full_name"]) if names_df is not None else None
    name_list = []
    if name_col and names_df is not None:
        name_list = names_df[name_col].dropna().astype(str).str.strip().unique().tolist()

    # Cities: many possible headers: 'city', 'city_ascii', 'name'
    city_col = _choose_column(cities_df, ["city", "city_ascii", "name", "place"]) if cities_df is not None else None
    city_list = []
    if city_col and cities_df is not None:
        city_list = cities_df[city_col].dropna().astype(str).str.strip().unique().tolist()

    # Normalized lower-case sets for fast membership checks
    country_set = set([c.lower() for c in country_list])
    name_set = set([n.lower() for n in name_list])
    city_set = set([c.lower() for c in city_list])

    return {
        "country_list": country_list,
        "country_set": country_set,
        "name_list": name_list,
        "name_set": name_set,
        "city_list": city_list,
        "city_set": city_set,
    }

# Build once (module import)
REFS = build_reference_sets()

# ---------- Utility functions ----------
def safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame is None. Please upload a file first.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df)}")
    return df.copy(deep=True)

def normalize_whitespace(s):
    return " ".join(str(s).split()).strip()

def smart_titlecase_name(name: str) -> str:
    if name is None or str(name).strip() == "":
        return ""
    try:
        hn = HumanName(str(name))
        # HumanName returns nicely-formatted name
        return normalize_whitespace(str(hn))
    except Exception:
        return normalize_whitespace(str(name)).title()

def titlecase_text(s: str) -> str:
    if s is None:
        return ""
    return normalize_whitespace(str(s)).title()

def fuzzy_match_one(query: str, choices: List[str], scorer=fuzz.WRatio, score_cutoff: int = 88):
    """Return (best_match, score) or (None, 0). Choices should be original-case strings."""
    if not query or not choices:
        return None, 0
    try:
        res = process.extractOne(query, choices, scorer=scorer)
        if res:
            match, score, _ = res
            if score >= score_cutoff:
                return match, score
            return match, score
    except Exception:
        return None, 0
    return None, 0

# ---------- Main function ----------
def apply_fixes(df: pd.DataFrame,
                columns_config: Optional[Dict[str, str]] = None,
                manual_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                fuzzy_threshold: int = 88) -> Dict:
    """
    Smart cleaning using your data/countries.csv, data/name_gender.csv, data/world_cities.csv.
    Returns dict: {'cleaned_df', 'duplicates_df', 'log'}.
    - columns_config (optional): {'name_col': 'FullName', 'country_col': 'Country', 'city_col': 'City'}
    - manual_mappings: per-column mapping old->new (applied before fuzzy correction)
    - fuzzy_threshold: 0-100 threshold for accepting fuzzy match (recommended 85-92)
    """
    log = []
    df = safe_copy_df(df)
    n_before = len(df)
    log.append(f"Rows before: {n_before}")

    # Infer columns if not provided
    if columns_config is None:
        columns_config = {}
        def guess_col(keywords):
            for c in df.columns:
                cl = c.lower()
                if any(k in cl for k in keywords):
                    return c
            return None
        columns_config['name_col'] = guess_col(['name', 'full_name', 'person'])
        columns_config['country_col'] = guess_col(['country', 'nation'])
        columns_config['city_col'] = guess_col(['city', 'town', 'place', 'location'])
        columns_config['email_col'] = guess_col(['email', 'e-mail'])
    log.append(f"Columns config: {columns_config}")

    # Basic cleaning for string (object) columns: strip and normalize whitespace
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").apply(normalize_whitespace)

    # Apply manual mappings if provided (highest priority)
    if manual_mappings:
        for col, mapping in manual_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).replace(mapping)
                log.append(f"Applied manual mapping on column {col} ({len(mapping)} entries)")

    # Helper to correct a value (uses refs)
    country_choices = REFS.get("country_list", []) or []
    city_choices = REFS.get("city_list", []) or []
    name_choices = REFS.get("name_list", []) or []
    country_set = REFS.get("country_set", set())
    city_set = REFS.get("city_set", set())
    name_set = REFS.get("name_set", set())

    def correct_value_by_type(val: str, col_type: str) -> str:
        """col_type: 'name' | 'city' | 'country' | 'generic'"""
        if val is None:
            return ""
        s = str(val).strip()
        if s == "":
            return ""

        low = s.lower()

        # Exact membership check first (fast)
        if col_type == "country":
            if low in country_set:
                return titlecase_text(s)  # keep consistent title-case form
            # try fuzzy
            match, score = fuzzy_match_one(s, country_choices, score_cutoff=fuzzy_threshold)
            if match and score >= fuzzy_threshold:
                log.append(f"Country fuzzy-correct: '{s}' → '{match}' (score={score})")
                return titlecase_text(match)
            return titlecase_text(s)

        if col_type == "city":
            if low in city_set:
                return titlecase_text(s)
            match, score = fuzzy_match_one(s, city_choices, score_cutoff=fuzzy_threshold)
            if match and score >= fuzzy_threshold:
                log.append(f"City fuzzy-correct: '{s}' → '{match}' (score={score})")
                return titlecase_text(match)
            return titlecase_text(s)

        if col_type == "name":
            # first exact lookup in name set
            if low in name_set:
                # smart titlecase using nameparser
                fixed = smart_titlecase_name(s)
                return fixed
            # try fuzzy against name choices (if reasonably sized)
            if len(name_choices) > 0:
                match, score = fuzzy_match_one(s, name_choices, score_cutoff=92)
                if match and score >= 92:
                    log.append(f"Name fuzzy-correct: '{s}' → '{match}' (score={score})")
                    return smart_titlecase_name(match)
            # fallback: use smart titlecase
            return smart_titlecase_name(s)

        # generic: title-case each word
        return titlecase_text(s)

    # Apply typed corrections
    name_col = columns_config.get('name_col')
    country_col = columns_config.get('country_col')
    city_col = columns_config.get('city_col')
    email_col = columns_config.get('email_col')

    # 1) Names
    if name_col and name_col in df.columns:
        df[name_col] = df[name_col].apply(lambda v: correct_value_by_type(v, "name"))
        log.append(f"Normalized names in column '{name_col}'")

    # 2) Countries
    if country_col and country_col in df.columns:
        df[country_col + "_canonical"] = df[country_col].apply(lambda v: correct_value_by_type(v, "country"))
        log.append(f"Canonicalized country column into '{country_col}_canonical'")

    # 3) Cities
    if city_col and city_col in df.columns:
        df[city_col + "_canonical"] = df[city_col].apply(lambda v: correct_value_by_type(v, "city"))
        log.append(f"Canonicalized city column into '{city_col}_canonical'")

    # 4) Emails -> lowercase + strip
    if email_col and email_col in df.columns:
        df[email_col] = df[email_col].astype(str).str.strip().str.lower()
        log.append(f"Normalized emails in '{email_col}'")

    # 5) Title-case remaining object columns that were not handled
    handled = {c for c in [name_col, country_col, city_col, email_col] if c}
    for col in df.columns:
        if df[col].dtype == object and col not in handled and not col.endswith("_canonical"):
            df[col] = df[col].apply(titlecase_text)

    # 6) Duplicate detection (blocking)
    dup_pairs = []
    try:
        # create simple block key (first char of name + first 3 of city canonical if exists)
        block_keys = []
        if name_col in df.columns:
            block_keys.append(df[name_col].fillna("").str[:1].str.lower())
        if city_col and (city_col + "_canonical") in df.columns:
            block_keys.append(df[city_col + "_canonical"].fillna("").str[:3].str.lower())

        if block_keys:
            if len(block_keys) > 1:
                block_series = pd.Series(["|".join(parts) for parts in zip(*block_keys)])
            else:
                block_series = block_keys[0]
            df['_block_key'] = block_series.fillna("")
        else:
            df['_block_key'] = ""

        blocks = df.groupby('_block_key').indices
        for block, indices in blocks.items():
            if len(indices) < 2:
                continue
            idx_list = list(indices)
            for i_pos in range(len(idx_list)):
                for j_pos in range(i_pos + 1, len(idx_list)):
                    i = idx_list[i_pos]
                    j = idx_list[j_pos]
                    base_score = 0
                    if name_col in df.columns:
                        n1 = str(df.at[i, name_col]).lower()
                        n2 = str(df.at[j, name_col]).lower()
                        if n1 and n2:
                            base_score = fuzz.WRatio(n1, n2)
                    if (city_col + "_canonical") in df.columns:
                        c1 = str(df.at[i, city_col + "_canonical"]).lower()
                        c2 = str(df.at[j, city_col + "_canonical"]).lower()
                        if c1 and c2:
                            city_score = fuzz.WRatio(c1, c2)
                            combined = 0.75 * base_score + 0.25 * city_score
                        else:
                            combined = base_score
                    else:
                        combined = base_score
                    if combined >= fuzzy_threshold:
                        dup_pairs.append({'row_i': i, 'row_j': j, 'score': combined})
        duplicates_df = pd.DataFrame(dup_pairs)
        log.append(f"Found {len(dup_pairs)} duplicate candidate pairs (threshold={fuzzy_threshold})")
    except Exception as e:
        duplicates_df = pd.DataFrame()
        log.append(f"Duplicate detection error: {e}")

    # 7) cleanup helper columns
    if '_block_key' in df.columns:
        df.drop(columns=['_block_key'], inplace=True)

    # 8) Final log and return
    log.append(f"Rows after: {len(df)}")

    return {
        "cleaned_df": df,
        "duplicates_df": duplicates_df,
        "log": log
    }
