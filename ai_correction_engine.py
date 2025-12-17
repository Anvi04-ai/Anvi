import requests

# 🌐 Your Cloudflare Worker URL
WORKER_URL = "https://ai-name-corrector.YOUR-NAME.workers.dev/clean"  # <-- REPLACE this with your actual URL

def correct_entity(name: str, entity_type: str = "name"):
    """
    Sends the name to the Cloudflare AI Worker for correction.
    """
    if not isinstance(name, str) or not name.strip():
        return name, 1.0  # skip empty values

    try:
        response = requests.post(WORKER_URL, json={"name": name})
        if response.status_code == 200:
            data = response.json()
            corrected = data.get("cleaned_name", name)
            return corrected, 0.98  # assume high confidence
        else:
            print("⚠️ Worker Error:", response.text)
            return name, 0.5
    except Exception as e:
        print("❌ Connection error:", e)
        return name, 0.5
