from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from argparse import ArgumentParser
from pathlib import Path 
import numpy as np  # local import for safe script-level use
from utils.utils import load_seq_



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fasta_list", type=str, required=True,
                        help="Path to .txt file where each line is a fasta file path")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save outputs, one subdir per fasta")
    parser.add_argument("--processed_paths_file", type=str, required=True,
                        help="Path to the processed_paths.txt file")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_paths_file = Path(args.processed_paths_file)
    processed_paths_file.touch(exist_ok=True)
    # Read fasta paths from .txt file
    with open(args.fasta_list, "r") as f:
        yaml_paths = [line.strip() for line in f if line.strip()]

    client = ESMC.from_pretrained("esmc_600m").to("cuda")

    for yaml_path_str in yaml_paths:
        yaml_path = Path(yaml_path_str)
        try:
            seq, mapping_db_seq = load_seq_(yaml_path, fasta=False)
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
            )
            # Prepare output subdirectory using fasta filename stem as unique identifier
            seq_index = yaml_path.stem
            yaml_output_dir = output_dir / seq_index
            yaml_output_dir.mkdir(parents=True, exist_ok=True)

            # Save logits and embeddings as .npy files
            logits_np = logits_output.logits.sequence.cpu().float().numpy()
            embeddings_np = logits_output.embeddings.cpu().float().numpy()


            np.save(yaml_output_dir / "logits.npy", logits_np)
            np.save(yaml_output_dir / "embeddings.npy", embeddings_np)
            # add the path to the processed_paths.txt file
            with open(processed_paths_file, "a") as f:
                f.write(yaml_path_str + "\n")
        except (KeyError, ValueError, FileNotFoundError) as e:
            print(f"ERROR: Failed to process {yaml_path_str}: {e}")
            print(f"Skipping and continuing with next file...")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing {yaml_path_str}: {e}")
            print(f"Skipping and continuing with next file...")
            continue


