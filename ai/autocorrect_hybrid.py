import pandas as pd
from rapidfuzz import process


COMMON_DICTIONARY = {
    "mumbay": "mumbai",
    "punee": "pune",
    "femlae": "female",
    "mal": "male",
    "adn": "and",
    "teh": "the",
}


def correct_word(word: str, dictionary: dict) -> str:
    """
    Correct single word using hybrid dictionary + fuzzy nearest match.
    """
    if word.lower() in dictionary:
        return dictionary[word.lower()]

    # Fuzzy match to nearest dictionary key
    keys = list(dictionary.keys())
    match, score, _ = process.extractOne(word.lower(), keys)

    if score >= 85:
        return dictionary[match]

    return word


def hybrid_autocorrect(df: pd.DataFrame, dictionary: dict = COMMON_DICTIONARY) -> pd.DataFrame:
    """
    Autocorrect text columns using a hybrid approach.
    """
    df = df.copy()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).apply(
                lambda x: " ".join(correct_word(w, dictionary) for w in x.split())
            )

    return df
