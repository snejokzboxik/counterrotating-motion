import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_acceptance_v12", ROOT / "scripts" / "run_acceptance_v12.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _spec(model):
    return {
        "name": f"{model}_h50_scale1",
        "kind": model,
        "mesh_um": 50,
        "scale_factor": 1.0,
        "voltage_V": 250.0,
        "coupling": 0.01,
        "leading_order": 4,
        "leading_quadrature_coefficient_norm": 1.0,
        "comparison_radius_m": 0.004,
        "leading_rms_field_per_volt_V_m": 1.0,
        "total_rms_field_per_volt_V_m": 1.0,
    }


def _record(model, rep, index, status):
    return {
        "job_id": f"{model}_r{rep}_{index}",
        "model": f"{model}_h50_scale1",
        "sample_id": f"r{rep:02d}_{index:06d}",
        "status": status,
        "complete": status in ("retained", "escaped"),
        "horizon_RF": 1000,
    }


def test_numerical_error_is_unresolved_and_blocks_incomplete_acceptance(tmp_path):
    specs = [_spec("A"), _spec("B")]
    records = [
        _record("A", 0, 0, "retained"),
        _record("A", 0, 1, "escaped"),
        _record("A", 1, 0, "retained"),
        _record("A", 1, 1, "numerical_error"),
        _record("B", 0, 0, "retained"),
        _record("B", 0, 1, "escaped"),
        _record("B", 1, 0, "retained"),
        _record("B", 1, 1, "escaped"),
    ]

    summary = MODULE._summarize(tmp_path, specs, 2, 1, 0.1, 2.0, True, records=records)

    assert summary["complete_trajectory_records"] == 7
    assert summary["numerical_error_records"] == 1
    assert summary["models_without_complete_estimate"] == ["A_h50_scale1"]
    assert summary["paired_comparisons"] == 0
    assert summary["complete_estimates"] == 1
    estimate = next(__import__("csv").DictReader(
        (tmp_path / "acceptance_estimates.csv").open(encoding="utf-8")))
    assert len(__import__("json").loads(estimate["retention_fraction_student_t_95pct_ci"])) == 2


def test_paired_ratio_uses_ratio_of_mean_retention_fractions(tmp_path):
    specs = [_spec("A"), _spec("B")]
    records = [
        _record("A", 0, 0, "retained"),
        _record("A", 0, 1, "retained"),
        _record("A", 0, 2, "escaped"),
        _record("A", 0, 3, "escaped"),
        _record("A", 1, 0, "retained"),
        _record("A", 1, 1, "retained"),
        _record("A", 1, 2, "retained"),
        _record("A", 1, 3, "retained"),
        _record("B", 0, 0, "retained"),
        _record("B", 0, 1, "retained"),
        _record("B", 0, 2, "escaped"),
        _record("B", 0, 3, "escaped"),
        _record("B", 1, 0, "escaped"),
        _record("B", 1, 1, "escaped"),
        _record("B", 1, 2, "escaped"),
        _record("B", 1, 3, "retained"),
    ]

    summary = MODULE._summarize(tmp_path, specs, 2, 2, 0.1, 2.0, True, records=records)

    assert summary["complete_estimates"] == 2
    assert summary["paired_comparisons"] == 1
    assert summary["complete_trajectory_records"] == 16
    assert summary["numerical_error_records"] == 0
    row = __import__("csv").DictReader((tmp_path / "paired_acceptance_differences.csv").open(encoding="utf-8"))
    pair = next(row)
    assert float(pair["ratio_of_mean_retention_fractions"]) == 2.0
    assert len(__import__("json").loads(pair["left_minus_right_retention_student_t_95pct_ci"])) == 2


def test_matched_voltage_reports_radius_dependent_real_and_ideal_c():
    at_2mm = MODULE._voltage_table(50, "matched", 2.0)
    at_4mm = MODULE._voltage_table(50, "matched", 4.0)
    at_6mm = MODULE._voltage_table(50, "matched", 6.0)

    assert abs(at_4mm["B"] - 250.0 * 2**0.5) < 1e-10
    assert abs(at_4mm["idealB"] - at_4mm["B"]) < 1e-12
    assert at_2mm["C"] > at_4mm["C"] > at_6mm["C"]
    assert at_6mm["C"] > at_6mm["idealC"]
    assert abs(at_4mm["C"] - 755.6575062984871) < 1e-8
