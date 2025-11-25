import os
import json
import pandas as pd
from datetime import datetime

STORAGE_DIR = "stored_datasets"


def ensure_storage_dir():
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)


def save_dataset(df: pd.DataFrame, name: str = None) -> str:
    """
    Save a cleaned dataset with timestamp.
    Returns filepath.
    """
    ensure_storage_dir()

    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"dataset_{timestamp}.csv"

    filepath = os.path.join(STORAGE_DIR, name)
    df.to_csv(filepath, index=False)

    return filepath


def list_saved_datasets() -> list:
    """Return list of stored files with path and size."""
    ensure_storage_dir()

    files = []
    for f in os.listdir(STORAGE_DIR):
        path = os.path.join(STORAGE_DIR, f)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            files.append({"name": f, "path": path, "size_bytes": size})

    return files


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load a dataset from storage by filename.
    """
    ensure_storage_dir()

    filepath = os.path.join(STORAGE_DIR, name)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No dataset found: {name}")

    if filepath.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.lower().endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        raise ValueError("Unsupported file type")
