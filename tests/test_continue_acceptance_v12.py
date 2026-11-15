import csv
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "continue_acceptance_v12", ROOT / "scripts" / "continue_acceptance_v12.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_continuation_source_resolves_original_samples_and_keeps_latest_state(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(MODULE, "RUNS", runs)
    base = runs / "base_run"
    base.mkdir(parents=True)
    physical = {"length_m": 0.008, "frequency_Hz": 100_000.0}
    _write_json(base / "config.json", {
        "config_hash": "base-hash", "horizon_RF": 1000, "physical": physical,
    })
    with (base / "initial_states_r00.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "replicate", "sample_index", "x0_m", "y0_m", "vx0_m_s", "vy0_m_s", "phase_rad",
        ])
        writer.writeheader()
        writer.writerow({
            "replicate": 0, "sample_index": 0, "x0_m": 0.001, "y0_m": -0.002,
            "vx0_m_s": 2.5, "vy0_m_s": -3.5, "phase_rad": 0.73,
        })

    model_name = "A_h50_scale1"
    sample_id = "r00_000000"
    continued_state = [0.31, -0.22, 0.013, -0.017]
    n3000 = runs / "continuation_base_run_N3000_testhash"
    _write_json(n3000 / "config.json", {
        "study": "V12 sequential trajectory continuation", "config_hash": "n3000-hash",
        "source_run": "base_run", "source_config_hash": "base-hash",
        "sample_source_run": "base_run", "source_horizon_RF": 1000,
        "horizon_RF": 3000, "physical": physical,
    })
    _write_json(n3000 / "model_coefficients.json", [{"name": model_name}])
    source_row = {
        "job_id": "n3000-job", "model": model_name, "sample_id": sample_id,
        "config_hash": "n3000-hash", "status": "retained", "complete": True,
        "reached_RF": 3000, "phase_rad": 0.73, "state_dimensionless": continued_state,
    }
    (n3000 / "completed_jobs.jsonl").write_text(
        json.dumps(source_row) + "\n", encoding="utf-8"
    )

    cfg, specs, rows, states = MODULE._read_source(n3000)

    assert cfg["horizon_RF"] == 3000
    assert specs == [{"name": model_name}]
    assert rows["n3000-job"]["state_dimensionless"] == continued_state
    assert MODULE._stored_run_reference(MODULE._resolve_sample_source(n3000, cfg)) == "base_run"
    expected = [
        0.001 / 0.008,
        -0.002 / 0.008,
        2.5 / (0.008 * 2 * math.pi * 100_000.0),
        -3.5 / (0.008 * 2 * math.pi * 100_000.0),
    ]
    state, phase = states[sample_id]
    assert all(math.isclose(a, b, rel_tol=1e-14) for a, b in zip(state, expected))
    assert math.isclose(phase, 0.73)


def test_legacy_continuation_chain_follows_source_run_to_original_samples(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    monkeypatch.setattr(MODULE, "RUNS", runs)
    base = runs / "base_run"
    base.mkdir(parents=True)
    (base / "initial_states_r00.csv").write_text(
        "replicate,sample_index,x0_m,y0_m,vx0_m_s,vy0_m_s,phase_rad\n"
        "0,0,0.001,0,0,0,1.2\n",
        encoding="utf-8",
    )
    n3000 = runs / "legacy_n3000"
    _write_json(n3000 / "config.json", {
        "config_hash": "legacy-hash", "source_run": "base_run", "horizon_RF": 3000,
        "physical": {"length_m": 0.008, "frequency_Hz": 100_000.0},
    })

    resolved = MODULE._resolve_sample_source(n3000, json.loads((n3000 / "config.json").read_text()))

    assert resolved == base.resolve()
