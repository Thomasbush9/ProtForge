import os
import time
from argparse import ArgumentParser

import torch
from esm.utils.forge_context_manager import ForgeBatchExecutor
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


def embed_sequence(sequence: str, model: ESMCModel, tokenizer) -> torch.Tensor:
    """Mean-pool last hidden state. HF ESMCModel has no encode/logits (those are SDK-only)."""
    inputs = tokenizer(sequence, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1)
    return (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _report_batch_failures(outputs: list) -> None:
    for i, result in enumerate(outputs):
        if isinstance(result, BaseException):
            print(f"task {i} failed: {type(result).__name__}: {result}", flush=True)


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

    print("Executing Batch operation")
    start = time.time()
    # Local GPU: one worker — batch_executor defaults to many threads on one CUDA model.
    with ForgeBatchExecutor(max_workers=1, max_attempts=1) as executor:
        outputs = executor.execute_batch(
            embed_sequence,
            model=model,
            tokenizer=tokenizer,
            sequence=sequences,
        )
    _report_batch_failures(outputs)
    ok = [o for o in outputs if not isinstance(o, BaseException)]
    if ok:
        print(f"Embeddings: {len(ok)} x shape {ok[0].shape}", flush=True)
    print(f"Embeddings generated in {time.time() - start:.3f}s")
