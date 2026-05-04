"""
Pure-Python helpers used by Snakefile / .smk rules.

Lifted out of Snakefile so the logic is independently importable and testable.
The Snakefile thin-wraps these to keep the rule-side call signatures terse
(rules don't pass `slurm_cfg` themselves — they call `stage_resource("boltz",
"mem_mb", 16000)` and the Snakefile-level wrapper supplies SLURM_CFG).
"""

from __future__ import annotations


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
