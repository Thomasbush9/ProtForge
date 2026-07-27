"""
Pure-Python helpers used by Snakefile / .smk rules.

Lifted out of Snakefile so the logic is independently importable and testable.
The Snakefile thin-wraps these to keep the rule-side call signatures terse
(rules don't pass `slurm_cfg` themselves — they call `stage_resource("boltz",
"mem_mb", 16000)` and the Snakefile-level wrapper supplies SLURM_CFG).
"""

from __future__ import annotations

import os
import re

# `${VAR}` / `$VAR` left behind by os.path.expandvars means the variable was not
# set in the environment. We refuse to run rather than silently build a path with
# a literal "${PROTFORGE_ROOT}" segment in it.
_UNEXPANDED = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def expand_config(node, _path: str = "", strict: bool = True):
    """Recursively expand ``${VAR}`` / ``$VAR`` / ``~`` in every config string.

    Lets a shipped template refer to the caller's workspace (``${PROTFORGE_ROOT}``)
    instead of hardcoding one user's absolute paths, so the only thing a new user
    edits is their SLURM account/email. Non-string leaves pass through untouched,
    so configs with plain absolute paths behave exactly as before.

    Raises KeyError naming the config key and the missing variable if a
    placeholder survives expansion — a wrong path should fail at parse time, not
    as a confusing mid-run "no such file".

    With ``strict=False`` an unset variable leaves the raw value in place instead
    of raising. The webapp seeds a new session from the shipped template before
    the user has necessarily exported ``PROTFORGE_ROOT``; keeping the literal
    ``${PROTFORGE_ROOT}`` there shows them what to fill in, and the strict pass
    at launch time still refuses to run on it.
    """
    if isinstance(node, dict):
        return {k: expand_config(v, f"{_path}.{k}" if _path else str(k), strict)
                for k, v in node.items()}
    if isinstance(node, list):
        return [expand_config(v, f"{_path}[{i}]", strict) for i, v in enumerate(node)]
    if not isinstance(node, str):
        return node

    expanded = os.path.expanduser(os.path.expandvars(node))
    leftover = _UNEXPANDED.search(expanded)
    if leftover:
        if not strict:
            return node
        raise KeyError(
            f"config key '{_path}' references environment variable "
            f"'{leftover.group(1)}', which is not set. Export it before running "
            f"(e.g. `export {leftover.group(1)}=/n/holylfs06/LABS/<lab>/Everyone/<you>`) "
            f"or replace the placeholder in config.yaml with a literal path. "
            f"Raw value: {node!r}"
        )
    return expanded


def unexpanded_var(value) -> str | None:
    """Name of the first environment variable `value` still depends on, if any.

    Lets a caller tell "this path is wrong" apart from "this path was never
    resolved" — a distinction that matters, because the fix for the second is to
    export a variable, not to spend an hour building container images.
    """
    if not isinstance(value, str):
        return None
    match = _UNEXPANDED.search(os.path.expandvars(value))
    return match.group(1) if match else None


def slurm_identity_errors(cfg: dict, require_account: bool = True) -> list[str]:
    """Problems with the SLURM identity a run would submit under.

    ``slurm.account`` reaches every rule's ``slurm_account`` and ``slurm.email``
    reaches sbatch's ``--mail-user``, so a config carrying the shipped template's
    placeholders — or another user's details — must not reach the cluster.

    ``require_account=False`` treats a missing account as acceptable (a local,
    non-SLURM run has no account to give) while still rejecting a placeholder.
    """
    errors: list[str] = []
    slurm = cfg.get("slurm", {}) or {}

    account = str(slurm.get("account", "") or "").strip()
    if not account:
        if require_account:
            errors.append(
                "slurm.account is not set — set it in your config "
                "(e.g. kempner_yourpi_lab) before launching."
            )
    elif account.startswith("<"):
        errors.append(
            f"slurm.account is still the template placeholder ({account}). "
            "Replace it with your own SLURM account."
        )

    email = str(slurm.get("email", "") or "").strip()
    if email.startswith("<"):
        errors.append(
            f"slurm.email is still the template placeholder ({email}). "
            "Replace it with your own address, or clear it to disable "
            "job notifications."
        )
    elif email and "@" not in email:
        errors.append(f"slurm.email does not look like an address: {email}")

    return errors


def stage_resource(slurm_cfg: dict, stage: str, key: str, default):
    """Read slurm.resources.<stage>.<key> from the SLURM config block.

    `slurm_cfg` is what Snakefile sees as `config.get("slurm", {})`. Falls back
    to `default` if the override is missing.
    """
    return (
        slurm_cfg.get("resources", {})
        .get(stage, {})
        .get(key, default)
    )


def stage_uses_gpu(slurm_cfg: dict, stage: str, default: bool) -> bool:
    """Whether this stage should request a GPU.

    Reads `slurm.resources.<stage>.gpus` if set (0 = no GPU); otherwise falls
    back to the rule's hardcoded default. Used by Snakefile's slurm_extra().
    """
    gpus = stage_resource(slurm_cfg, stage, "gpus", None)
    if gpus is None:
        return default
    return int(gpus) > 0
