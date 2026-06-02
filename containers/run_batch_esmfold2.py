import os
import time
from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer
from transformers.models.esmc.modeling_esmc import ESMCModel

ESMC_REPO = "biohub/ESMC-6B"


def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str, required=True)
    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)

    print("Loading the model...")
    tokenizer = AutoTokenizer.from_pretrained(
        ESMC_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    )
    model = ESMCModel.from_pretrained(
        ESMC_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()

    sequences = [
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "MQIFVKTTSDTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    ]

    print("Launching tokenizer...")
    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    print("Input tokenized")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    print("Starting Inference")
    start = time.time()
    with torch.inference_mode():
        output = model(**inputs)
    print(f"Inference Completed in:{time.time()-start}")
    print(output)
