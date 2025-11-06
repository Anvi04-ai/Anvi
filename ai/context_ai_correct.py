import re
import pandas as pd
import os
from ai.autocorrect_hybrid import hybrid_text_clean  # ✅ Hybrid correction layer import


def gpt_context_correction(text):
    """
    Context-based AI text correction (light version).
    Detects simple spelling and domain-level issues and fixes them.
    Safe conversion for numeric and non-string inputs included.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.strip()

    # Common spelling fixes — customizable
    replacements = {
        "counrty": "country",
        "adress": "address",
        "imndfia": "India",
        "rooll": "roll",
        "rool": "roll",
        "correctionn": "correction",
        "hte": "the",
        "teh": "the",
        "recieve": "receive",
        "adresss": "address",
        "studnt": "student",
        "collge": "college",
        "technlogy": "technology",
        "enviroment": "environment",
        "reserch": "research",
        "drishti": "Drishti"  # protect proper noun
    }

    def correct_word(word):
        clean_word = re.sub(r'[^\w\s]', '', word.lower())
        if clean_word in replacements:
            corrected = replacements[clean_word]
            if word.istitle():
                corrected = corrected.capitalize()
            elif word.isupper():
                corrected = corrected.upper()
            return corrected
        return word

    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    corrected_text = " ".join(corrected_words)
    corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()

    return corrected_text


def safe_context_ai_clean(df, output_path="cleaned_output.csv", verbose=True):
    """
    🔹 Runs hybrid + context-based AI cleaning safely on a DataFrame.
    🔹 Automatically skips numeric and datetime-like columns for better speed.
    🔹 Converts non-string cells to strings when necessary.
    🔹 Automatically saves cleaned CSV file.
    """
    try:
        for col in df.columns:
            col_type = str(df[col].dtype)

            # ✅ Skip numeric and datetime columns automatically
            if pd.api.types.is_numeric_dtype(df[col]):
                if verbose:
                    print(f"⏩ Skipping numeric column: {col}")
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                if verbose:
                    print(f"⏩ Skipping datetime column: {col}")
                continue

            # ✅ Convert to string and fill NaN safely
            df[col] = df[col].astype(str).fillna("")

            # ✅ Apply both hybrid and contextual correction
            df[col] = df[col].apply(lambda x: gpt_context_correction(hybrid_text_clean(x)))

        # ✅ Save output to CSV automatically
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            df.to_csv(output_path, index=False)
            if verbose:
                print(f"📁 Cleaned file saved to: {output_path}")

        if verbose:
            print("✅ Context + Hybrid AI text cleaning completed successfully!")

        return df

    except Exception as e:
        print(f"❌ Context AI cleaning failed: {e}")
        return df