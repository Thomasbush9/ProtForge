from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from argparse import ArgumentParser
# Ubiquitin (PDB 1UBQ)
sequence = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# optionally use "biohub/ESMFold2"
#TODO:specify where the hub will be
model = ESMFold2Model.from_pretrained("biohub/ESMFold2-Fast").cuda().eval()
output = model.infer_protein(sequence, num_loops=3, num_sampling_steps=50)

print(f"pLDDT mean: {float(output['plddt'].mean()):.3f}, pTM: {float(output['ptm'].mean()):.3f}")

