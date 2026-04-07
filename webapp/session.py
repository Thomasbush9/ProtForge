"""
Session management for ProtForge multi-session support.

Each session lives in .sessions/<uuid>/ with its own config.yaml,
run metadata, and log file. A central registry tracks all sessions.
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = REPO_ROOT / ".sessions"
REGISTRY_FILE = SESSIONS_DIR / "registry.json"


class Session:
    """Represents a single pipeline session with its own config and state."""

    def __init__(self, session_id: str):
        self.id = session_id
        self.dir = SESSIONS_DIR / session_id
        self.config_path = self.dir / "config.yaml"
        self.run_meta_file = self.dir / ".snakemake_run.json"
        self.log_file = self.dir / "snakemake_run.log"
        self.backup_dir = self.dir / ".config_backups"


def load_registry() -> dict:
    """Load the session registry, or return an empty one."""
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    return {"sessions": [], "active_session_id": None}


def save_registry(registry: dict):
    """Write the registry to disk."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_FILE.write_text(json.dumps(registry, indent=2))


def create_session(name: str, base_config: dict | None = None) -> Session:
    """Create a new session. Optionally seed it with a config dict."""
    session_id = uuid.uuid4().hex[:12]
    session = Session(session_id)
    session.dir.mkdir(parents=True, exist_ok=True)

    config = base_config if base_config is not None else {}
    session.config_path.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )

    registry = load_registry()
    registry["sessions"].append({
        "id": session_id,
        "name": name,
        "created": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
    })
    # If this is the first session, make it active
    if registry["active_session_id"] is None:
        registry["active_session_id"] = session_id
    save_registry(registry)
    return session


def delete_session(session_id: str):
    """Remove a session's directory and registry entry."""
    registry = load_registry()
    registry["sessions"] = [s for s in registry["sessions"] if s["id"] != session_id]

    # If we deleted the active session, switch to another
    if registry["active_session_id"] == session_id:
        if registry["sessions"]:
            registry["active_session_id"] = registry["sessions"][0]["id"]
        else:
            registry["active_session_id"] = None

    save_registry(registry)

    session_dir = SESSIONS_DIR / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)


def rename_session(session_id: str, new_name: str):
    """Rename a session in the registry."""
    registry = load_registry()
    for s in registry["sessions"]:
        if s["id"] == session_id:
            s["name"] = new_name
            s["last_modified"] = datetime.now().isoformat()
            break
    save_registry(registry)


def touch_session(session_id: str):
    """Update last_modified timestamp for a session."""
    registry = load_registry()
    for s in registry["sessions"]:
        if s["id"] == session_id:
            s["last_modified"] = datetime.now().isoformat()
            break
    save_registry(registry)


def list_sessions() -> list[dict]:
    """Return all sessions from the registry."""
    return load_registry().get("sessions", [])


def get_session(session_id: str) -> Session:
    """Get a Session object by ID."""
    return Session(session_id)


def get_active_session_id() -> str | None:
    """Return the active session ID from the registry."""
    return load_registry().get("active_session_id")


def set_active_session(session_id: str):
    """Update the active session in the registry."""
    registry = load_registry()
    registry["active_session_id"] = session_id
    save_registry(registry)


def migrate_legacy():
    """One-time migration: create a Default session from existing repo-root files.

    Only runs if .sessions/ doesn't exist yet. Copies (not moves) existing
    config.yaml, .snakemake_run.json, and snakemake_run.log so nothing is lost.
    """
    if SESSIONS_DIR.exists() and REGISTRY_FILE.exists():
        return  # Already migrated

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing config if present
    legacy_config_path = REPO_ROOT / "config.yaml"
    base_config = None
    if legacy_config_path.exists():
        try:
            base_config = yaml.safe_load(legacy_config_path.read_text()) or {}
        except yaml.YAMLError:
            base_config = {}

    session = create_session("Default", base_config=base_config)

    # Copy (not move) legacy run metadata
    legacy_meta = REPO_ROOT / ".snakemake_run.json"
    if legacy_meta.exists():
        shutil.copy2(legacy_meta, session.run_meta_file)

    legacy_log = REPO_ROOT / "snakemake_run.log"
    if legacy_log.exists():
        shutil.copy2(legacy_log, session.log_file)

    # Ensure this session is active
    set_active_session(session.id)
