"""ESMFold prototype: fold every FASTA in a directory with facebook/esmfold_v1.

Requirements (login env): transformers, accelerate, torch, numpy, tqdm.
Set HF_HOME to the cache populated by download_esmfold.py before running, so the
model loads offline on a compute node (combine with HF_HUB_OFFLINE=1 to be safe).

For sequences >~700 residues or on GPU OOM, pass --chunk-size (e.g. 64) to
enable folding-trunk chunking.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from transformers.models.esm.openfold_utils.protein import Protein as OFProtein
from transformers.models.esm.openfold_utils.protein import to_pdb


def parse_fasta(path: Path) -> str:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"{path} is not a valid FASTA (missing '>' header).")
    return "".join(lines[1:])


def load_fasta_dir(fasta_dir: Path) -> dict[str, str]:
    files = sorted(fasta_dir.glob("*.fasta"))
    if not files:
        raise FileNotFoundError(f"No .fasta files found in {fasta_dir}")
    return {p.stem: parse_fasta(p) for p in files}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HF cache root (same path passed to download_esmfold.py, e.g. "
             "/n/holylfs06/.../esm_models_cache). When set, loads offline.",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Folding-trunk chunk size. Enable for long sequences or on OOM.",
    )
    args = ap.parse_args()

    sequences = load_fasta_dir(args.fasta_dir)
    print(f"Found {len(sequences)} FASTA files in {args.fasta_dir}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    hub_cache = (args.cache_dir / "hub") if args.cache_dir is not None else None
    offline = hub_cache is not None
    print(
        f"Loading facebook/esmfold_v1 on {device} "
        f"(cache={hub_cache}, offline={offline})...",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir=str(hub_cache) if hub_cache else None,
        local_files_only=offline,
    )
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1",
        cache_dir=str(hub_cache) if hub_cache else None,
        local_files_only=offline,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    if device == "cuda":
        model.esm = model.esm.half()
    if args.chunk_size is not None:
        model.trunk.set_chunk_size(args.chunk_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, seq in tqdm(sequences.items(), desc="Folding"):
        try:
            input_ids = tokenizer(
                [seq], return_tensors="pt", add_special_tokens=False
            )["input_ids"].to(device)
            with torch.no_grad():
                output = model(input_ids)
            pdbs = convert_outputs_to_pdb(output)
            out_subdir = args.output_dir / name / "esmfold"
            out_subdir.mkdir(parents=True, exist_ok=True)
            (out_subdir / "structure.pdb").write_text("".join(pdbs))
            plddt = output["plddt"][0].cpu().float().numpy()
            np.save(out_subdir / "plddt.npy", plddt)
        except Exception as exc:
            print(f"[ERROR] {name}: {exc}", file=sys.stderr, flush=True)
            continue


if __name__ == "__main__":
    main()
