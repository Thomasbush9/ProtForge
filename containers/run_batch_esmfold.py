import os
import time
from argparse import ArgumentParser

from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


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


