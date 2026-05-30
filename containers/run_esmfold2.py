from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from argparse import ArgumentParser




if __name__ == "__main__":


    parser = ArgumentParser()
    parser.add_argument('--cache', type=str)
    args = parser.parse_args()
    cache_dir = args.cache


    sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

    model = ESMFold2Model.from_pretrained(
        "biohub/ESMFold2-Fast",
        cache_dir=f"{cache_dir}/hub",
        local_files_only=True,
    ).cuda().eval()
    output = model.infer_protein(sequence, num_loops=3, num_sampling_steps=50)

    print(f"pLDDT mean: {float(output['plddt'].mean()):.3f}, pTM: {float(output['ptm'].mean()):.3f}")


