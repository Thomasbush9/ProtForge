import os
from argparse import ArgumentParser

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--cache",
        type=str,
        required=True,
        help="HF_HOME root (bind-mounted cache, e.g. /models/hf)",
    )
    args = parser.parse_args()

    hub_cache = _enforce_offline(args.cache)

    sequence = (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    )

    print(f"Loading {ESMFOLD2_REPO} offline from {hub_cache}...", flush=True)
    model = ESMFold2Model.from_pretrained(
        ESMFOLD2_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()

    output = model.infer_protein(sequence, num_loops=3, num_sampling_steps=50)
    print(
        f"pLDDT mean: {float(output['plddt'].mean()):.3f}, "
        f"pTM: {float(output['ptm'].mean()):.3f}"
    )
