import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "containers"))
from yaml_to_openfold_json import convert_input_dir


def test_convert_boltz_yaml_with_msa(tmp_path: Path):
    fixtures = Path(__file__).resolve().parents[1] / "containers" / "test" / "fixtures"
    work = tmp_path / "work"
    query_json, runner_path = convert_input_dir(fixtures, work, "/data", max_seq_count=1024)

    data = json.loads(query_json.read_text())
    assert "seq1_msa" in data["queries"]
    chain = data["queries"]["seq1_msa"]["chains"][0]
    assert chain["chain_ids"] == ["A"]
    assert chain["main_msa_file_paths"] == ["/data/msa/seq1.a3m"]
    assert (work / "msa" / "seq1.a3m").is_symlink()

    assert runner_path is not None
    runner = yaml.safe_load(runner_path.read_text())
    msa_cfg = runner["dataset_config_kwargs"]["msa"]
    assert msa_cfg["max_seq_counts"]["seq1"] == 1024
    assert msa_cfg["aln_order"] == ["seq1"]


def test_convert_msa_free_yaml(tmp_path: Path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    (input_dir / "mono.yaml").write_text(
        "version: 1\nsequences:\n  - protein:\n"
        '      id: "A"\n      sequence: ACDE\n'
    )
    work = tmp_path / "work"
    _, runner_path = convert_input_dir(input_dir, work, "/data", max_seq_count=100)
    data = json.loads((work / "query.json").read_text())
    assert data["queries"]["mono"]["chains"][0]["sequence"] == "ACDE"
    assert "main_msa_file_paths" not in data["queries"]["mono"]["chains"][0]
    assert runner_path is None
