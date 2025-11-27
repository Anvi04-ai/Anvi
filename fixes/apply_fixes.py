# fixes/apply_fixes.py
import pandas as pd
import pycountry
from rapidfuzz import process, fuzz
from nameparser import HumanName
import logging
from typing import Optional, Dict, List
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # root folder

# Load all dictionaries
countries_df = pd.read_csv(os.path.join(BASE_DIR, "data/countries.csv"))
names_df = pd.read_csv(os.path.join(BASE_DIR, "data/name_gender.csv"))
cities_df = pd.read_csv(os.path.join(BASE_DIR, "data/world_cities.csv"))

# Convert to Python dictionaries for fast lookup
COUNTRY_LIST = set(countries_df["C1"].str.lower().str.strip())

NAME_LIST = set(names_df["name"].str.lower().str.strip())

CITY_LIST = set(cities_df["city"].str.lower().str.strip())

def apply_fixes(df):
    df = df.copy()

    log_messages = []

    def clean_text(value):
        if not isinstance(value, str):
            return value

        original = value.strip()
        low = original.lower()

        # Capitalization
        capitalized = original.title()

        # Country correction
        if low in COUNTRY_LIST:
            log_messages.append(f"Country corrected: {original} → {capitalized}")
            return capitalized

        # Name correction
        if low in NAME_LIST:
            log_messages.append(f"Name corrected: {original} → {capitalized}")
            return capitalized

        # City correction
        if low in CITY_LIST:
            log_messages.append(f"City corrected: {original} → {capitalized}")
            return capitalized

        # If unknown → keep title case
        if original != capitalized:
            log_messages.append(f"Capitalized: {original} → {capitalized}")

        return capitalized

    # Apply to whole DF
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).apply(clean_text)

    # Detect duplicates
    duplicates = df[df.duplicated(keep=False)]

    return {
        "cleaned_df": df,
        "duplicates_df": duplicates,
        "log": log_messages
    }


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------
# Helper utilities
# -------------------------
def safe_copy_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("Input DataFrame is None. Please upload a file first.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df)}")
    return df.copy(deep=True)


def normalize_whitespace(s: str) -> str:
    return " ".join(str(s).split()).strip()


def smart_titlecase_name(name: str) -> str:
    """
    Use nameparser to handle capitalization better than str.title()
    """
    if pd.isna(name) or str(name).strip() == "":
        return ""
    try:
        hn = HumanName(str(name))
        # nameparser lowercases and capitalizes properly, preserving Mc/Mac isn't perfect but much better.
        return normalize_whitespace(str(hn))
    except Exception:
        return normalize_whitespace(str(name)).title()


def normalize_text_for_matching(s: str) -> str:
    return normalize_whitespace(str(s)).strip().lower()


# -------------------------
# Reference matching helpers
# -------------------------
def build_reference_index(values: List[str]) -> List[str]:
    """Return cleaned unique list for matching"""
    cleaned = []
    for v in values:
        if pd.isna(v):
            continue
        s = normalize_text_for_matching(v)
        if s:
            cleaned.append(v)  # keep original form for return
    # preserve original distinct values
    return list(dict.fromkeys(cleaned))


def fuzzy_match_one(query: str, choices: List[str], scorer=fuzz.WRatio, score_cutoff: int = 85):
    """Return (best_match, score) or (None, 0)"""
    if not query or not choices:
        return None, 0
    result = process.extractOne(query, choices, scorer=scorer)
    if result is None:
        return None, 0
    match, score, _ = result
    if score >= score_cutoff:
        return match, score
    return None, score


def canonicalize_country(name: str, country_list: Optional[List[str]] = None, score_cutoff: int = 85):
    """
    Try pycountry first, then fuzzy-match against provided country_list (if any).
    Returns canonical country name or original.
    """
    if not name or str(name).strip() == "":
        return ""
    n = str(name).strip()
    # Try pycountry
    try:
        c = pycountry.countries.lookup(n)
        return c.name
    except Exception:
        pass
    # try fuzzy match against provided list if present
    if country_list:
        match, score = fuzzy_match_one(n, country_list, score_cutoff=score_cutoff)
        if match:
            return match
    # fallback: return original but titlecased
    return smart_titlecase_name(n)


# -------------------------
# Main apply_fixes function
# -------------------------
def apply_fixes(df: pd.DataFrame,
                columns_config: Optional[Dict[str, Dict]] = None,
                country_ref: Optional[List[str]] = None,
                city_ref: Optional[List[str]] = None,
                manual_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                fuzzy_threshold: int = 88) -> Dict:
    """
    df: input dataframe
    columns_config: optional config telling which columns are 'name','country','city','email', etc.
      Example:
        {'name_col': 'full_name', 'country_col': 'country', 'city_col': 'city', 'email_col': 'email'}
    country_ref / city_ref: lists of canonical country and city names (strings). Use your CSVs here.
    manual_mappings: explicit corrections mapping: {'column_name': {'mispelled':'Correct'}}
    fuzzy_threshold: score threshold for fuzzy matches (0-100)

    Returns dict with:
      - 'cleaned_df': cleaned DataFrame
      - 'duplicates_df': DataFrame of detected fuzzy duplicate pairs (may be empty)
      - 'log': list of strings (actions performed)
    """
    log = []
    df = safe_copy_df(df)
    n_before = len(df)
    log.append(f"Rows before: {n_before}")

    # default column mapping inference if not provided
    if columns_config is None:
        # try to guess common names
        columns_config = {}
        cols = [c.lower() for c in df.columns]

        def col_like(key_words):
            for c in df.columns:
                if any(k in c.lower() for k in key_words):
                    return c
            return None

        columns_config['name_col'] = col_like(['name', 'full_name', 'person'])
        columns_config['country_col'] = col_like(['country', 'nation'])
        columns_config['city_col'] = col_like(['city', 'town'])
        columns_config['email_col'] = col_like(['email', 'e-mail'])
    log.append(f"Columns config: {columns_config}")

    # 1) Basic cleaning for all string columns: strip spaces, normalize unicode
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].fillna("").apply(normalize_whitespace)

    # 2) Apply manual mappings if provided (top-priority)
    if manual_mappings:
        for col, mapping in manual_mappings.items():
            if col in df.columns:
                df[col] = df[col].astype(str).replace(mapping)
                log.append(f"Applied manual mapping on column {col} ({len(mapping)} entries)")

    # 3) Name capitalization & normalization
    name_col = columns_config.get('name_col')
    if name_col and name_col in df.columns:
        df[name_col] = df[name_col].apply(lambda x: smart_titlecase_name(x))
        log.append(f"Normalized names in column {name_col}")

    # 4) Country canonicalization
    country_col = columns_config.get('country_col')
    country_choices = None
    if country_ref:
        country_choices = list(dict.fromkeys(country_ref))
    if country_col and country_col in df.columns:
        df[country_col + "_canonical"] = df[country_col].apply(
            lambda v: canonicalize_country(v, country_list=country_choices, score_cutoff=fuzzy_threshold))
        log.append(f"Canonicalized country column into {country_col + '_canonical'}")

    # 5) City canonicalization using provided city_ref (best-effort fuzzy)
    city_col = columns_config.get('city_col')
    city_choices = None
    if city_ref:
        city_choices = list(dict.fromkeys(city_ref))
    if city_col and city_col in df.columns and city_choices:
        # Use process.extractOne for each row, but do it in a loop to avoid memory explosion for huge lists.
        def _match_city(val):
            if not val:
                return ""
            match, score = fuzzy_match_one(val, city_choices, score_cutoff=fuzzy_threshold)
            return match if match else smart_titlecase_name(val)

        df[city_col + "_canonical"] = df[city_col].apply(_match_city)
        log.append(f"Canonicalized city column into {city_col + '_canonical'}")

    # 6) Email lowercasing and whitespace trimming
    email_col = columns_config.get('email_col')
    if email_col and email_col in df.columns:
        df[email_col] = df[email_col].astype(str).str.strip().str.lower()
        log.append(f"Normalized emails in {email_col}")

    # 7) Duplicate detection (fast blocking + fuzzy check)
    # We'll create a simple signature to block on: first letter of name + first 3 letters of city (if present)
    dup_pairs = []
    try:
        block_keys = []
        if name_col in df.columns:
            block_keys.append(df[name_col].fillna("").str[:1].str.lower())
        if city_col in df.columns:
            block_keys.append(df[city_col].fillna("").str[:3].str.lower())
        if block_keys:
            block_series = pd.Series(["|".join(parts) for parts in zip(*block_keys)]) if len(block_keys) > 1 else \
            block_keys[0]
            df['_block_key'] = block_series.fillna("")
        else:
            df['_block_key'] = ""

        # build index mapping block -> indices
        blocks = df.groupby('_block_key').indices
        # For each block with more than 1 row, compare pairwise using fuzzy ratio on name (or all columns)
        for block, indices in blocks.items():
            if len(indices) < 2:
                continue
            idx_list = list(indices)
            for i_pos in range(len(idx_list)):
                for j_pos in range(i_pos + 1, len(idx_list)):
                    i = idx_list[i_pos]
                    j = idx_list[j_pos]
                    base_score = 0
                    # compute name similarity if exists
                    if name_col in df.columns:
                        n1 = normalize_text_for_matching(df.at[i, name_col])
                        n2 = normalize_text_for_matching(df.at[j, name_col])
                        if n1 and n2:
                            base_score = fuzz.WRatio(n1, n2)
                    # combine with city similarity if available
                    if city_col in df.columns:
                        c1 = normalize_text_for_matching(df.at[i, city_col]) if pd.notna(df.at[i, city_col]) else ""
                        c2 = normalize_text_for_matching(df.at[j, city_col]) if pd.notna(df.at[j, city_col]) else ""
                        if c1 and c2:
                            city_score = fuzz.WRatio(c1, c2)
                            # weighted average: names more important
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
        logger.exception("Error during duplicate detection")
        duplicates_df = pd.DataFrame()
        log.append(f"Duplicate detection error: {e}")

    # 8) Final touches: drop temporary helper columns
    if '_block_key' in df.columns:
        df.drop(columns=['_block_key'], inplace=True)

    # 9) Return results
    results = {
        'cleaned_df': df,
        'duplicates_df': duplicates_df,
        'log': log
    }
    return results
