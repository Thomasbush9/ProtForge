"""ESMFold worker: fold sequences listed in a chunk file using facebook/esmfold_v1.

Input paths can be .yaml (Boltz-style) or .fasta/.a3m — load_seq_ handles both.
Saves per-sequence {output_dir}/{stem}/esmfold/{structure.pdb,plddt.npy}.

Requires HF cache populated by scripts/download_esmfold.py before first run.
Pass --cache_dir to force offline load (recommended on compute nodes without internet).
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb

from utils.utils import load_seq_


def convert_outputs_to_pdb(outputs) -> list[str]:
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        pred = OFProtein(
            aatype=outputs["aatype"][i],
            atom_positions=final_atom_positions[i],
            atom_mask=final_atom_mask[i],
            residue_index=outputs["residue_index"][i] + 1,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--fasta_list", type=str, required=True,
                        help="Path to .txt file where each line is a YAML or FASTA path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs, one subdir per sequence")
    parser.add_argument("--processed_paths_file", type=str, required=True,
                        help="Path to the processed_paths.txt file")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF cache root (same path used by download_esmfold.py). "
                             "Enables offline load from compute nodes.")
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="Folding-trunk chunk size. Enable for long sequences or on OOM.")
    parser.add_argument("--chunk_size_threshold", type=int, default=1200,
                        help="If a sequence is at least this long, enable trunk chunking "
                             "(via --chunk_size, default 64 if --chunk_size unset). Default 1200 "
                             "matches the H100 80GB OOM ceiling observed in calibration.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_paths_file = Path(args.processed_paths_file)
    processed_paths_file.touch(exist_ok=True)

    with open(args.fasta_list, "r") as f:
        paths = [line.strip() for line in f if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    hub_cache = str(Path(args.cache_dir) / "hub") if args.cache_dir else None
    offline = hub_cache is not None
    print(
        f"Loading facebook/esmfold_v1 on {device} (cache={hub_cache}, offline={offline})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir=hub_cache,
        local_files_only=offline,
    )
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir=hub_cache,
        local_files_only=offline,
        low_cpu_mem_usage=True,
        # In container mode the def file bakes only model.safetensors (not
        # pytorch_model.bin) to halve image size. Without this flag the
        # resolver looks for the .bin first, returns None, and either
        # crashes in load_state_dict or triggers a network-only
        # auto_conversion lookup that fails under offline mode.
        use_safetensors=True,
    )
    model = model.to(device)
    if device == "cuda":
        model.esm = model.esm.half()

    auto_chunk = args.chunk_size if args.chunk_size is not None else 64
    current_chunk = None  # tracks what's currently set on model.trunk

    for path_str in paths:
        path = Path(path_str)
        is_fasta = path.suffix.lower() in (".fasta", ".fa", ".a3m")
        try:
            seq, _mapping = load_seq_(path, fasta=is_fasta)
            seq_len = len(seq)
            want_chunk = auto_chunk if seq_len >= args.chunk_size_threshold else None
            if want_chunk != current_chunk:
                model.trunk.set_chunk_size(want_chunk)
                current_chunk = want_chunk
                print(f"[{path.name}] L={seq_len} chunk_size={want_chunk}",
                      flush=True)
            input_ids = tokenizer(
                [seq], return_tensors="pt", add_special_tokens=False,
            )["input_ids"].to(device)
            with torch.no_grad():
                output = model(input_ids)
            pdbs = convert_outputs_to_pdb(output)

            seq_name = path.stem
            out_subdir = output_dir / seq_name / "esmfold"
            out_subdir.mkdir(parents=True, exist_ok=True)
            (out_subdir / "structure.pdb").write_text("".join(pdbs))
            plddt = output["plddt"][0].cpu().float().numpy()
            np.save(out_subdir / "plddt.npy", plddt)

            with open(processed_paths_file, "a") as f:
                f.write(path_str + "\n")
        except (KeyError, ValueError, FileNotFoundError) as e:
            print(f"ERROR: Failed to process {path_str}: {e}", file=sys.stderr)
            print(f"Skipping and continuing with next file...", file=sys.stderr)
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing {path_str}: {e}", file=sys.stderr)
            print(f"Skipping and continuing with next file...", file=sys.stderr)
            continue
