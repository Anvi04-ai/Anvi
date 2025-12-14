import pandas as pd
import re
from datetime import datetime
from forex_python.converter import CurrencyRates

# 🌍 Try importing optional libraries
try:
    from postal.parser import parse_address
    from postal.expand import expand_address
    POSTAL_AVAILABLE = True
except ImportError:
    POSTAL_AVAILABLE = False

try:
    from langdetect import detect
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False


# -----------------------------------------
# 🌐 1️⃣ Language detection and translation
# -----------------------------------------
def detect_and_translate(text, target_lang="en"):
    """Detect the language and translate to English (if needed)."""
    if not isinstance(text, str) or not text.strip():
        return text

    if not TRANSLATOR_AVAILABLE:
        return text  # fallback if missing dependencies

    try:
        detected_lang = detect(text)
        if detected_lang != target_lang:
            translated = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
            return translated
        return text
    except Exception:
        return text


# -----------------------------------------
# ⏰ 2️⃣ Time normalization
# -----------------------------------------
def normalize_time(value):
    """Normalize time strings like '3rd Dec 2025' → '2025-12-03'."""
    if not isinstance(value, str):
        return value

    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%b %d, %Y", "%d %b %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


# -----------------------------------------
# 💰 3️⃣ Currency conversion
# -----------------------------------------
def convert_to_usd(value):
    """
    Detects numeric + currency patterns and converts to USD using forex_python.
    Example: '€120' → 130.45 (USD)
    """
    if not isinstance(value, str):
        return value

    c = CurrencyRates()

    # Extract currency symbol and amount
    match = re.match(r"([€£₹$¥])\s*([\d,.]+)", value)
    if not match:
        return value

    symbol, amount_str = match.groups()
    amount = float(amount_str.replace(",", ""))

    currency_map = {"€": "EUR", "£": "GBP", "₹": "INR", "$": "USD", "¥": "JPY"}
    currency = currency_map.get(symbol, "USD")

    try:
        if currency == "USD":
            return round(amount, 2)
        usd_value = c.convert(currency, "USD", amount)
        return round(usd_value, 2)
    except Exception:
        return amount


# -----------------------------------------
# 🏠 4️⃣ Address standardization
# -----------------------------------------
def standardize_address(address):
    """
    Uses libpostal if available, otherwise returns address unchanged.
    Example: '221B Baker St, London' → '221 Baker Street, London'
    """
    if not isinstance(address, str) or not address.strip():
        return address

    if not POSTAL_AVAILABLE:
        return address  # graceful fallback

    try:
        expanded = expand_address(address)
        if expanded:
            return expanded[0]
        parsed = parse_address(address)
        normalized = ", ".join([p[0] for p in parsed])
        return normalized
    except Exception:
        return address


# -----------------------------------------
# 🧪 Quick test
# -----------------------------------------
if __name__ == "__main__":
    print("🌍 postal installed:", POSTAL_AVAILABLE)
    print("🌎 translator installed:", TRANSLATOR_AVAILABLE)

    print(detect_and_translate("Bonjour le monde"))
    print(normalize_time("3rd Dec 2025"))
    print(convert_to_usd("€120"))
    print(standardize_address("221B Baker St, London"))
