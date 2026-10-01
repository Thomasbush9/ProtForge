"""
Pure-Python helpers used by Snakefile / .smk rules.

Lifted out of Snakefile so the logic is independently importable and testable.
The Snakefile thin-wraps these to keep the rule-side call signatures terse
(rules don't pass `slurm_cfg` themselves — they call `stage_resource("boltz",
"mem_mb", 16000)` and the Snakefile-level wrapper supplies SLURM_CFG).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

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


# Enabled stage -> (containers.<key>, weight-cache config path). The MSA
# databases and the Boltz checkpoint are shared cluster paths, so only the
# per-user HF/OpenFold caches are checked here.
_STAGE_ASSETS = {
    "msa": ("colabfold", None),
    "boltz": ("boltz", None),
    "esmc": ("esmc", ("esmc", "cache_dir")),
    "esmfold": ("esmfold", ("esmfold", "cache_dir")),
    "openfold": ("openfold", ("openfold", "cache_dir")),
}


def check_stage_assets(cfg: dict, placeholder_hint: str = "") -> list[str]:
    """Fatal missing container images or model caches for the enabled stages.

    A seeded config points at `$PROTFORGE_ASSETS/sifs/*.sif` and
    `$PROTFORGE_ASSETS/models/hf`, which only exist once containers/build.sh and
    scripts/download_models.py have been run. Without this the config validates,
    every job is submitted, and each one dies on a missing image.

    Both the Snakefile and the webapp's preflight call this, so a dry run and
    the UI agree on whether an install is actually runnable.

    ``placeholder_hint`` appends caller-specific remediation to the unresolved-
    ``${VAR}`` message. The webapp needs it (its sessions store an already-seeded
    config, so exporting the variable is not enough on its own); the Snakefile
    does not, and never reaches that branch anyway because ``expand_config``
    raises on an unresolved placeholder first.
    """
    errors: list[str] = []
    pipeline = cfg.get("pipeline", {}) or {}
    containers = cfg.get("containers", {}) or {}
    shared_gpu = containers.get("gpu", "")
    sae_enabled = ((cfg.get("esmc", {}) or {}).get("sae", {}) or {}).get("enabled", False)

    def _placeholder_error(key: str, raw: str, var: str) -> str:
        # An unresolved ${VAR} is not a missing asset. Saying "build it" here
        # would send the user off to rebuild images that already exist.
        msg = (
            f"{key} still contains an unresolved ${{{var}}}: {raw}. "
            f"Export {var} before running."
        )
        return f"{msg} {placeholder_hint}".rstrip() if placeholder_hint else msg

    for stage, (container_key, cache_key) in _STAGE_ASSETS.items():
        if not pipeline.get(stage) and not (stage == "esmc" and sae_enabled):
            continue

        sif = containers.get(container_key, "") or shared_gpu
        var = unexpanded_var(sif)
        if not sif:
            errors.append(
                f"{stage} is enabled but containers.{container_key} is not set."
            )
        elif var:
            errors.append(
                _placeholder_error(f"containers.{container_key}", sif, var)
            )
        elif not Path(os.path.expanduser(os.path.expandvars(sif))).is_file():
            # Report where we actually looked, not the raw ${VAR} form.
            errors.append(
                f"{stage} container image not found: {os.path.expandvars(sif)} — "
                f"build it with `bash containers/build.sh` "
                f"(see docs/CLUSTER_SETUP.md)."
            )

        if cache_key:
            cache = (cfg.get(cache_key[0], {}) or {}).get(cache_key[1], "")
            cache_var = unexpanded_var(cache)
            key_name = f"{cache_key[0]}.{cache_key[1]}"
            if not cache:
                errors.append(f"{stage} is enabled but {key_name} is not set.")
            elif cache_var:
                errors.append(_placeholder_error(key_name, cache, cache_var))
            elif not Path(os.path.expanduser(os.path.expandvars(cache))).is_dir():
                resolved = os.path.expanduser(os.path.expandvars(cache))
                errors.append(
                    f"{stage} model cache not found: {resolved} — download "
                    f"weights with `python scripts/download_models.py "
                    f"--cache-dir {resolved}`."
                )

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


_ESMC_DEPTHS = {"300M": 30, "600M": 36, "6B": 80}
_SAE_REPOS = {
    "all-layers": "biohub/ESMC-{size}-sae-k64-codebook16384",
    "mlp": "biohub/ESMC-{size}-sae-mlp-k64-codebook131072",
}


def parse_esmc_layer_selection(value, size) -> list[int] | None:
    """Parse a layer selection against the model's inclusive hidden-state range."""
    if not isinstance(size, str) or size not in _ESMC_DEPTHS:
        raise ValueError(f"Unknown ESM-C size {size!r}; choose 300M, 600M, or 6B.")
    text = str(value).strip()
    if text.lower() == "all":
        return None
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(re.fullmatch(r"[0-9]+", part) is None for part in parts):
        raise ValueError(
            f"Layers must be 'all' or comma-separated non-negative integers, got {value!r}."
        )
    layers = [int(part) for part in parts]
    if len(set(layers)) != len(layers):
        raise ValueError(f"Layers must be distinct, got {value!r}.")
    depth = _ESMC_DEPTHS[size]
    if any(layer > depth for layer in layers):
        raise ValueError(f"Layers for {size} must be in 0..{depth}, got {value!r}.")
    return layers


def _sae_snapshot(cache: Path, repo: str) -> Path:
    """Resolve one local revision, never combining files from different snapshots."""
    root = cache / "hub" / f"models--{repo.replace('/', '--')}"
    ref = root / "refs" / "main"
    if ref.exists() or ref.is_symlink():
        revision = ref.read_text().strip()
        if not revision or Path(revision).name != revision:
            raise ValueError(f"Invalid cached main revision in {ref}.")
        snapshot = root / "snapshots" / revision
        if not snapshot.is_dir():
            raise ValueError(f"Cached main snapshot not found: {snapshot}.")
        return snapshot
    snapshots = sorted(p for p in (root / "snapshots").glob("*") if p.is_dir())
    if len(snapshots) != 1:
        raise ValueError(
            f"Cannot select cached SAE snapshot for {repo}: expected refs/main "
            f"or one snapshot in {root / 'snapshots'}. Download the selected SAE repo."
        )
    return snapshots[0]


def esmc_activation_errors(cfg, *, check_cache: bool = True) -> list[str]:
    """Validate active outputs; off-cluster DAG planning may skip cache checks."""
    errors: list[str] = []
    esmc = cfg.get("esmc", {}) or {}
    sae = esmc.get("sae", {}) or {}
    embeddings_enabled = (cfg.get("pipeline", {}) or {}).get("esmc", False)
    sae_enabled = sae.get("enabled", False)
    if not embeddings_enabled and not sae_enabled:
        return errors

    def sizes_for(value, key):
        if not isinstance(value, (list, tuple)) or not value:
            errors.append(f"{key} must select at least one ESM-C size: 300M, 600M, or 6B.")
            return []
        sizes = []
        for size in value:
            if not isinstance(size, str) or size not in _ESMC_DEPTHS:
                errors.append(f"{key} has unsupported size {size!r}; choose 300M, 600M, or 6B.")
            elif size not in sizes:
                sizes.append(size)
        return sizes

    models = esmc.get("models", ["6B"])
    if embeddings_enabled:
        model_sizes = sizes_for(models, "esmc.models")
        if esmc.get("save_hidden_states", False):
            for key, allowed, default in (
                ("hidden_pool", ("none", "mean"), "none"),
                ("hidden_dtype", ("float16", "bfloat16", "float32"), "float16"),
            ):
                if esmc.get(key, default) not in allowed:
                    errors.append(f"esmc.{key} must be {'|'.join(allowed)}.")
            for size in model_sizes:
                try:
                    parse_esmc_layer_selection(esmc.get("hidden_layers", "all"), size)
                except ValueError as exc:
                    errors.append(f"esmc.hidden_layers: {exc}")

    if not sae_enabled:
        return errors
    # An omitted/empty SAE subset inherits embedding sizes, matching the rule.
    sae_sizes = sizes_for(sae.get("sizes") or models, "esmc.sae.sizes / esmc.models")
    sae_type = sae.get("sae_type", "all-layers")
    if not isinstance(sae_type, str) or sae_type not in _SAE_REPOS:
        errors.append("esmc.sae.sae_type must be all-layers|mlp.")
        return errors
    selections = {}
    for size in sae_sizes:
        try:
            selections[size] = parse_esmc_layer_selection(sae.get("layers", "all"), size)
        except ValueError as exc:
            errors.append(f"esmc.sae.layers: {exc}")
    if not check_cache:
        return errors

    cache_value = esmc.get("cache_dir", "")
    if not cache_value:
        errors.append("esmc.cache_dir must be set to the HF cache containing the selected SAE repo.")
        return errors
    var = unexpanded_var(cache_value)
    if var:
        errors.append(f"esmc.cache_dir contains unresolved ${{{var}}}; export it before launching.")
        return errors
    cache = Path(os.path.expanduser(os.path.expandvars(cache_value)))
    for size, selected in selections.items():
        repo = _SAE_REPOS[sae_type].format(size=size)
        try:
            snapshot = _sae_snapshot(cache, repo)
            config_path = snapshot / "config.json"
            if not config_path.is_file():
                raise ValueError(f"SAE config missing: {config_path}.")
            metadata = json.loads(config_path.read_text())
            available = metadata.get("available_layers") if isinstance(metadata, dict) else None
            if (
                not isinstance(available, list) or not available
                or any(type(layer) is not int for layer in available)
            ):
                raise ValueError(f"{config_path} must advertise a nonempty available_layers integer list.")
            parse_esmc_layer_selection(",".join(map(str, available)), size)
            layers = available if selected is None else selected
            unavailable = sorted(set(layers) - set(available))
            if unavailable:
                raise ValueError(f"{repo} does not advertise SAE layers {unavailable} in {config_path}.")
            missing = [
                f"layer_{layer}.safetensors" for layer in layers
                if not (snapshot / f"layer_{layer}.safetensors").is_file()
            ]
            if missing:
                raise ValueError(
                    f"SAE shards missing in {snapshot}: {', '.join(missing)}. "
                    f"Download these files from {repo} before launching."
                )
        except (OSError, ValueError) as exc:
            errors.append(f"esmc.sae ({size}, {sae_type}): {exc}")
    return errors
