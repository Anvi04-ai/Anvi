# utils/storage.py
import json
from pathlib import Path
from typing import Any, Dict, Optional

BASE = Path.cwd()  # project root when running from project folder
PREFS_PATH = BASE / "user_prefs.json"
SESSIONS_PATH = BASE / "session_history.json"

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# --- Preferences ---
def load_prefs() -> Dict[str, Any]:
    """Return saved user preferences (or {} if none)."""
    return _read_json(PREFS_PATH)

def save_prefs(prefs: Dict[str, Any]) -> None:
    """Save user preferences (overwrites file)."""
    _write_json(PREFS_PATH, prefs)

# --- Session history (appendable) ---
def load_sessions() -> Dict[str, Any]:
    """Return session history dict (or {})."""
    return _read_json(SESSIONS_PATH)

def append_session(session: Dict[str, Any]) -> None:
    """
    Append a session record to session_history.json.
    Session could be like: {"timestamp": "...", "actions": [...], "file_name": "..."}
    """
    data = load_sessions()
    # store as list under "sessions"
    sessions = data.get("sessions", [])
    sessions.append(session)
    data["sessions"] = sessions
    _write_json(SESSIONS_PATH, data)