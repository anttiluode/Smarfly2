from pathlib import Path
import subprocess
import sys

from smarfly2.replay import replay_frames
from smarfly2.session import Session
from smarfly2.world import SyntheticWorld


ROOT = Path(__file__).resolve().parents[1]


def legacy_session(tmp_path):
    world = SyntheticWorld(120, 80, seed=14)
    full = replay_frames([world.frame(i) for i in range(100)], seed=3, dt=1/30)
    legacy = Session(full.records, horizon=8)
    path = tmp_path / "legacy.npz"
    legacy.save(path)
    return path


def test_replay_sequencing_requires_geometry_for_legacy_session(tmp_path):
    path = legacy_session(tmp_path)
    bad = subprocess.run(
        [sys.executable, "scripts/replay_session.py", str(path), "--headless", "--inspect-sequencing"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert bad.returncode != 0
    assert "world dimensions" in bad.stderr

    good = subprocess.run(
        [sys.executable, "scripts/replay_session.py", str(path), "--headless", "--inspect-sequencing",
         "--world-width", "120", "--world-height", "80"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert good.returncode == 0, good.stderr


def test_matched_report_sequencing_requires_geometry_for_legacy_session(tmp_path):
    path = legacy_session(tmp_path)
    bad = subprocess.run(
        [sys.executable, "scripts/matched_present_report.py", str(path), "--sequencing"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert bad.returncode != 0
    assert "world dimensions" in bad.stderr

    good = subprocess.run(
        [sys.executable, "scripts/matched_present_report.py", str(path), "--sequencing",
         "--world-width", "120", "--world-height", "80"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert good.returncode == 0, good.stderr


def test_run_live_exposes_legacy_model_geometry_override_flags():
    result = subprocess.run(
        [sys.executable, "scripts/run_live.py", "--help"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--world-width" in result.stdout
    assert "--world-height" in result.stdout
