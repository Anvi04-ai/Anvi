import json
import os
from datetime import datetime


LOG_DIR = "logs"


def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def save_log(event_type: str, details: dict):
    """
    Save logs of all actions performed.
    """
    ensure_log_dir()

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": event_type,
        "details": details
    }

    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
    filepath = os.path.join(LOG_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=4)

    return filepath

