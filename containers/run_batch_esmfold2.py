from functools import cache
import os, time 
from argparse import ArgumentParser
from esm.sdk import batch_executor
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from transformers.models.esmc.modeling_esmc import ESMCModel
from esm.sdk.api import ESMProtein, LogitsConfig
from containers.run_esmfold2 import hub_cache

ESMFOLD2_REPO = "biohub/ESMFold2-Fast"
ESMC_REPO = "biohub/ESMC-6B"
#config for pooling 
EMBEDDING_CONFIG = LogitsConfig(
    sequence=True, return_hidden_states=False, return_mean_hidden_states=True
)
MEAN_POOLED_EMBEDDING_CONFIG = LogitsConfig(return_hidden_states=True, return_mean_hidden_states=True)

def _enforce_offline(cache_dir: str | None) -> str | None:
    if not cache_dir:
        return None
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return f"{cache_dir}/hub"

#TODO:enforce cuda device for encoding 
def embed_sequence(sequence:str, model:ESMCModel):
    protein = ESMProtein.from_sequence(sequence=sequence)
    protein_tensor = model.encode(protein)
    output = model.logits(protein_tensor, config=MEAN_POOLED_EMBEDDING_CONFIG)
    return output

def parse_inputs(input:str):
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache", type=str)
    parser.add_argument("--input", type=str)

    args = parser.parse_args()
    hub_cache = _enforce_offline(args.cache)


    print("Loading the model...")
    model = ESMCModel.from_pretrained(
        ESMC_REPO,
        cache_dir=hub_cache,
        local_files_only=True,
    ).cuda().eval()
    # get batch of input: dir[.yaml|.fasta]->list[seq]
    sequences = [
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "MQIFVKTTSDTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG", 
    ]

    print("Executing Batch operation")
    start = time.time()
    with batch_executor() as executor: 
        outputs = executor.execute_batch(
            user_func=embed_sequence, model=model,
            sequence=sequences)
    end = time.time()
    print(f"Embeddings generated in {end-start}")
    


