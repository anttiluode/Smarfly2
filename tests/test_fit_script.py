import json
from pathlib import Path
import subprocess
import sys


def test_fit_script_receipt_includes_geometry_and_fixed_sequencing_parameters(tmp_path):
    out = tmp_path / "receipt.json"
    subprocess.run(
        [sys.executable, "scripts/fit_observers.py", "--synthetic", "160", "--seed", "7", "--out", str(out)],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    payload = json.loads(out.read_text())
    assert payload["world_width"] == 320
    assert payload["world_height"] == 240
    assert payload["sequencing_params"]["fast_tau"] == 0.30
    assert payload["sequencing_params"]["slow_tau"] == 3.0
    assert payload["sequencing_params"]["period_s"] == 0.8
    assert payload["sequencing_params"]["n_channels"] == 4
    assert "sequencing" in payload["metrics"]
    assert payload["invariance_checks"]["sequencing_publication_block"] is True
