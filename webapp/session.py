"""
Session management for ProtForge multi-session support.

Each session lives in .sessions/<uuid>/ with its own config.yaml,
run metadata, and log file. A central registry tracks all sessions.
"""

import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSIONS_DIR = REPO_ROOT / ".sessions"
REGISTRY_FILE = SESSIONS_DIR / "registry.json"

# The cluster-correct config a new user starts from. Everything that is a
# property of Kempner (shared databases, partitions, container layout) is
# already filled in; only the user's own fields are blanked out below.
KEMPNER_TEMPLATE = REPO_ROOT / "config.kempner.template.yaml"

# `${VAR}` expansion is shared with the Snakefile so the webapp and the workflow
# agree on what a template path resolves to.
_WORKFLOW_SCRIPTS = REPO_ROOT / "workflow" / "scripts"
if str(_WORKFLOW_SCRIPTS) not in sys.path:
    sys.path.append(str(_WORKFLOW_SCRIPTS))

from snake_helpers import expand_config  # noqa: E402

# Template fields the user must supply, written as `<YOUR_EMAIL>`-style markers.
# Seeding blanks them: a placeholder that reaches the config tab looks like a
# real value, and one that reaches sbatch mails the wrong person.
_PLACEHOLDER = re.compile(r"^<[^<>]+>$")


class Session:
    """Represents a single pipeline session with its own config and state."""

    def __init__(self, session_id: str):
        self.id = session_id
        self.dir = SESSIONS_DIR / session_id
        self.config_path = self.dir / "config.yaml"
        self.run_meta_file = self.dir / ".snakemake_run.json"
        self.log_file = self.dir / "snakemake_run.log"
        self.backup_dir = self.dir / ".config_backups"


def current_user() -> str:
    """The OS user running the app — recorded on each session it creates."""
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


def blank_placeholders(node):
    """Replace `<YOUR_EMAIL>`-style template markers with empty strings."""
    if isinstance(node, dict):
        return {k: blank_placeholders(v) for k, v in node.items()}
    if isinstance(node, list):
        return [blank_placeholders(v) for v in node]
    if isinstance(node, str) and _PLACEHOLDER.match(node.strip()):
        return ""
    return node


def default_seed_config() -> dict:
    """The config a brand-new session starts from.

    A repo-root config.yaml wins when present — that is the user's own file,
    written by the documented `cp config.kempner.template.yaml config.yaml`
    setup step. Otherwise fall back to the shipped Kempner template so a fresh
    clone opens on working cluster defaults instead of an empty form.

    `${PROTFORGE_ROOT}` is expanded when it is exported, and left literal when
    it is not (non-strict) so the field still shows what belongs there. The
    strict pass in check_launch_inputs() blocks a launch on the leftovers.
    """
    for source in (REPO_ROOT / "config.yaml", KEMPNER_TEMPLATE):
        if not source.is_file():
            continue
        try:
            cfg = yaml.safe_load(source.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(cfg, dict):
            continue
        return blank_placeholders(expand_config(cfg, strict=False))
    return {}


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
    """Create a new session.

    `base_config` of None means "start from the cluster defaults"
    (default_seed_config()); pass an explicit `{}` for a genuinely blank config.
    """
    session_id = uuid.uuid4().hex[:12]
    session = Session(session_id)
    session.dir.mkdir(parents=True, exist_ok=True)

    config = default_seed_config() if base_config is None else base_config
    session.config_path.write_text(
        yaml.dump(config, default_flow_style=False, sort_keys=False)
    )

    registry = load_registry()
    registry["sessions"].append({
        "id": session_id,
        "name": name,
        "created": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
        # Who owns this session. The repo often lives in a shared lab directory,
        # so a session created by someone else must be flagged rather than
        # silently adopted along with their account, email and paths.
        "created_by": current_user(),
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


def foreign_session_owner(session_id: str) -> str | None:
    """The user who created `session_id`, if that is not the user running now.

    Returns None when the session is ours, or when it predates `created_by`
    (an older registry) — an absent owner is unknown, not foreign.
    """
    for s in load_registry().get("sessions", []):
        if s["id"] == session_id:
            owner = s.get("created_by")
            return owner if owner and owner != current_user() else None
    return None


def migrate_legacy():
    """One-time bootstrap: create the Default session.

    Only runs if .sessions/ doesn't exist yet. Seeds from repo-root config.yaml
    when the user has already written one, else from the shipped Kempner
    template (see default_seed_config), and copies (not moves) existing
    .snakemake_run.json and snakemake_run.log so nothing is lost.
    """
    if SESSIONS_DIR.exists() and REGISTRY_FILE.exists():
        return  # Already migrated

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session = create_session("Default")

    # Copy (not move) legacy run metadata
    legacy_meta = REPO_ROOT / ".snakemake_run.json"
    if legacy_meta.exists():
        shutil.copy2(legacy_meta, session.run_meta_file)

    legacy_log = REPO_ROOT / "snakemake_run.log"
    if legacy_log.exists():
        shutil.copy2(legacy_log, session.log_file)

    # Ensure this session is active
    set_active_session(session.id)
