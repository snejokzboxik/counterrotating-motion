"""Validate and freeze the completed V12 A/B/C baseline acceptance result.

This postprocessor never integrates trajectories or edits the source run. It
requires all 4 x 256 common Sobol starts to have complete A/B/C checkpoints,
then exports a compact trajectory table, replica-aware acceptance estimates,
paired contrasts, and publication figures to a separate results directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_acceptance_v12 as runner  # noqa: E402

DEFAULT_RUN = ROOT / (
    "outputs/octupole_comparison_home/dimensionless_acceptance_v12/runs/"
    "e6bf3a33795e4a5354a8_equal_broad_lambda_h50_4x256"
)
DEFAULT_OUT = ROOT / (
    "outputs/octupole_comparison_home/dimensionless_acceptance_v12/results/"
    "baseline_equal_250V_phase0_N1000"
)
MODELS = ("A_h50_scale1", "B_h50_scale1", "C_h50_scale1")
REPLICATES = 4
SAMPLE_POWER = 8
HORIZON_RF = 1000


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_write(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if not rows and columns is None:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = columns or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_and_collect(run_dir: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("sample_power") != SAMPLE_POWER or config.get("replicates") != REPLICATES:
        raise ValueError("Source run is not the predeclared four x 256 Sobol experiment")
    if config.get("horizon_RF") != HORIZON_RF or config.get("phase_deg") != 0.0:
        raise ValueError("Source run phase or horizon differs from the requested baseline")
    if config.get("protocol") != "equal" or config.get("meshes_um") != [50]:
        raise ValueError("Source run is not the equal-voltage h50 baseline")
    if config.get("domain", {}).get("injection_radius_mm") != 7.5 or config.get("domain", {}).get("max_energy_eV") != 5.0:
        raise ValueError("Source run injection domain differs from r<=7.5 mm, E<=5 eV")
    if config.get("physical", {}).get("mass_kg") != runner.MASS or config.get("physical", {}).get("charge_C") != runner.E_CHARGE:
        raise ValueError("Source run mass or charge differs from the V12 baseline")
    if config.get("physical", {}).get("frequency_Hz") != runner.FREQUENCY:
        raise ValueError("Source run RF frequency differs from the V12 baseline")

    all_specs = json.loads((run_dir / "model_coefficients.json").read_text(encoding="utf-8"))
    by_name = {row["name"]: row for row in all_specs}
    if not set(MODELS).issubset(by_name):
        raise ValueError("Source run lacks one or more real A/B/C h50 scale-1 models")
    specs = [by_name[name] for name in MODELS]
    for spec in specs:
        if spec["kind"] not in "ABC" or spec["mesh_um"] != 50 or spec["scale_factor"] != 1.0 or spec["voltage_V"] != 250.0:
            raise ValueError(f"Unexpected baseline model specification: {spec['name']}")

    initial_by_id: dict[str, dict] = {}
    source_hashes = {}
    n_per_rep = 2**SAMPLE_POWER
    for rep in range(REPLICATES):
        sample_path = run_dir / f"initial_states_r{rep:02d}.csv"
        source_hashes[sample_path.name] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
        rows = read_csv(sample_path)
        if len(rows) != n_per_rep:
            raise ValueError(f"Expected {n_per_rep} starts in {sample_path}, found {len(rows)}")
        for row in rows:
            sample_id = f"r{int(row['replicate']):02d}_{int(row['sample_index']):06d}"
            if sample_id in initial_by_id:
                raise ValueError(f"Duplicate Sobol sample ID {sample_id}")
            initial_by_id[sample_id] = {**row, "sample_id": sample_id}
    if len(initial_by_id) != REPLICATES * n_per_rep:
        raise ValueError("Sobol sample table does not contain the exact expected state set")

    trajectory_rows: list[dict] = []
    estimate_records: list[dict] = []
    for model_name in MODELS:
        checkpoint_dir = run_dir / "checkpoints" / model_name
        checkpoints = sorted(checkpoint_dir.glob("*.json"))
        by_id = {}
        for path in checkpoints:
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("config_hash") != config["config_hash"] or row.get("model") != model_name:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            sample_id = row.get("sample_id")
            if sample_id in by_id:
                raise ValueError(f"Duplicate checkpoint for {model_name}/{sample_id}")
            by_id[sample_id] = row
        if set(by_id) != set(initial_by_id):
            missing = sorted(set(initial_by_id) - set(by_id))[:5]
            extra = sorted(set(by_id) - set(initial_by_id))[:5]
            raise ValueError(f"Incomplete checkpoint set for {model_name}; missing={missing}, extra={extra}")
        for sample_id in sorted(initial_by_id):
            cp = by_id[sample_id]
            if not cp.get("complete") or cp.get("status") not in {"retained", "escaped"}:
                raise ValueError(f"Unresolved trajectory {model_name}/{sample_id}: {cp.get('status')}")
            if cp.get("horizon_RF") != HORIZON_RF:
                raise ValueError(f"Horizon mismatch in {model_name}/{sample_id}")
            if cp.get("status") == "retained" and cp.get("reached_RF") != HORIZON_RF:
                raise ValueError(f"Retained trajectory did not reach the full horizon: {model_name}/{sample_id}")
            if cp.get("status") == "escaped" and cp.get("escape_RF") is None:
                raise ValueError(f"Escaped trajectory has no first-exit time: {model_name}/{sample_id}")
            record = dict(cp)
            record["job_id"] = cp.get("job_id")
            estimate_records.append(record)
            initial = initial_by_id[sample_id]
            trajectory_rows.append({
                "sample_id": sample_id,
                "replicate": int(initial["replicate"]),
                "sample_index": int(initial["sample_index"]),
                "model": model_name,
                "x0_m": float(initial["x0_m"]), "y0_m": float(initial["y0_m"]),
                "vx0_m_s": float(initial["vx0_m_s"]), "vy0_m_s": float(initial["vy0_m_s"]),
                "E0_eV": float(initial["E0_eV"]), "beta_rad": float(initial["beta_rad"]),
                "phase_rad": float(initial["phase_rad"]), "voltage_V": float(cp["voltage_V"]),
                "status": cp["status"], "escape_RF": cp["escape_RF"],
                "reached_RF": cp["reached_RF"], "max_radius_sampled_rho": cp["max_radius_sampled_rho"],
                "nfev": cp["nfev"],
            })

    sample_state = next(iter(initial_by_id.values()))
    injection_radius_m = float(config["domain"]["injection_radius_m"])
    emax_ev = float(config["domain"]["max_energy_eV"])
    mass = float(config["physical"]["mass_kg"])
    charge = float(config["physical"]["charge_C"])
    omega = 2.0 * math.pi * float(config["physical"]["frequency_Hz"])
    length = float(config["physical"]["length_m"])
    rho_max = injection_radius_m / length
    velocity_max = math.sqrt(2.0 * emax_ev * runner.E_CHARGE / mass)
    u_max = velocity_max / (length * omega)
    domain_volume = math.pi**2 * rho_max**2 * u_max**2
    physical_factor = length**4 * omega**2

    area_by_id = {name: 0 for name in MODELS}
    summary = runner._summarize(
        DEFAULT_OUT, specs, REPLICATES, SAMPLE_POWER,
        domain_volume, physical_factor, True, records=estimate_records,
    )
    if summary["complete_estimates"] != len(MODELS) or summary["numerical_error_records"]:
        raise ValueError(f"Refusing to publish incomplete estimates: {summary}")
    for row in trajectory_rows:
        area_by_id[row["model"]] += row["status"] == "retained"
    csv_write(DEFAULT_OUT / "trajectory_results.csv", trajectory_rows)

    input_shas = {
        "config.json": hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest(),
        "model_coefficients.json": hashlib.sha256((run_dir / "model_coefficients.json").read_bytes()).hexdigest(),
        **source_hashes,
    }
    estimates = read_csv(DEFAULT_OUT / "acceptance_estimates.csv")
    pairs = read_csv(DEFAULT_OUT / "paired_acceptance_differences.csv")
    result = {
        "study": "V12 finite-time transverse 4D acceptance baseline",
        "status": "complete",
        "source_run": run_dir.relative_to(ROOT).as_posix(),
        "source_config_hash": config["config_hash"],
        "source_hashes": input_shas,
        "physical_parameters": config["physical"],
        "domain": config["domain"],
        "protocol": "equal 250 V electrode peak, RF phase 0 deg, real FEM N48 h50, horizon 1000 RF",
        "sampling": {
            "sampler": "paired scrambled Sobol",
            "replicates": REPLICATES, "states_per_replicate": n_per_rep,
            "common_initial_states": len(initial_by_id), "trajectories": len(trajectory_rows),
            "resolved_trajectory_outcomes": len(trajectory_rows),
        },
        "integration_domain": {
            "rho_injection_max": rho_max, "u_velocity_max": u_max,
            "a4_domain_dimensionless": domain_volume,
            "A4_domain_m4_per_s2": domain_volume * physical_factor,
            "phase_space_conversion_m4_per_s2": physical_factor,
            "measure": "uniform position disk area x uniform velocity disk area",
        },
        "uncertainty": "replicate-level standard error and Student-t 95% CI over four independent Sobol scrambles; paired model differences and bootstrap ratio intervals",
        "model_retained_counts": area_by_id,
        "estimates": estimates,
        "paired_comparisons": pairs,
        "files": {
            "acceptance_estimates": "acceptance_estimates.csv",
            "paired_differences": "paired_acceptance_differences.csv",
            "trajectory_results": "trajectory_results.csv",
            "figure": "baseline_acceptance.png/.pdf/.svg",
            "report": "BASELINE_REPORT_RU.md",
        },
    }
    (DEFAULT_OUT / "baseline_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    make_figure(estimates)
    write_report(result)
    print(json.dumps({"status": result["status"], "out": str(DEFAULT_OUT),
                      "trajectories": len(trajectory_rows), "retained": area_by_id,
                      "estimates": estimates, "paired_comparisons": pairs}, indent=2))


def make_figure(estimates: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    label = {"A_h50_scale1": "A — вращающийся октуполь",
             "B_h50_scale1": "B — стоячий октуполь",
             "C_h50_scale1": "C — противофазная схема"}
    rows = sorted(estimates, key=lambda row: row["model"])
    means = np.array([float(row["retention_fraction_mean"]) for row in rows])
    intervals = [json.loads(row["retention_fraction_student_t_95pct_ci"]) for row in rows]
    lower = np.array([means[i] - intervals[i][0] for i in range(len(rows))])
    upper = np.array([intervals[i][1] - means[i] for i in range(len(rows))])
    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    y = np.arange(len(rows))
    ax.errorbar(means, y, xerr=np.vstack((lower, upper)), fmt="o", capsize=5,
                markersize=7, color="#176B87", ecolor="#334E68", linewidth=1.6)
    ax.set_yticks(y, [label[row["model"]] for row in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Доля удержанных начальных состояний до 1000 RF-периодов")
    ax.set_title("4D аксептанс при равных 250 В, фаза 0°")
    ax.grid(axis="x", alpha=0.25)
    out = DEFAULT_OUT
    fig.savefig(out / "baseline_acceptance.pdf")
    fig.savefig(out / "baseline_acceptance.svg")
    fig.savefig(out / "baseline_acceptance.png", dpi=600)
    plt.close(fig)


def write_report(result: dict) -> None:
    text = [
        "# V12 — базовая оценка четырёхмерного аксептанса",
        "",
        "Статус: завершённая конечновременная оценка. Это доля удержанных стартов и интеграл по указанной области инжекции, не утверждение о бесконечном удержании.",
        "",
        f"- Поле и режим: {result['protocol']}",
        f"- Физические старты: {result['sampling']['common_initial_states']} общих состояний, {result['sampling']['replicates']} независимых scrambled Sobol-реплик по {result['sampling']['states_per_replicate']} точки.",
        f"- Рассчитано траекторий A/B/C: {result['sampling']['trajectories']}; все имеют разрешённый статус retained/escaped.",
        f"- Область: равномерное распределение по диску координат r₀≤{result['domain']['injection_radius_mm']} мм и диску скоростей E₀≤{result['domain']['max_energy_eV']} эВ; фаза 0°; горизонт 1000 RF.",
        f"- Четырёхмерный интеграл имеет единицы м⁴·с⁻². Объём всего домена: `{result['integration_domain']['A4_domain_m4_per_s2']:.8g}` м⁴·с⁻²; безразмерный объём домена: `{result['integration_domain']['a4_domain_dimensionless']:.8g}`.",
        "- Интервалы долей удержания и аксептанса оцениваются по разбросу четырёх независимых реплик; модельные различия парные, поскольку начальные состояния общие.",
        "",
        "## Оценки по моделям",
        "",
        "| Модель | Удержано | P_retained (95% CI) | a₄ (95% CI) | A₄, м⁴·с⁻² (95% CI) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["estimates"]:
        fraction_ci = json.loads(row["retention_fraction_student_t_95pct_ci"])
        a4_ci = json.loads(row["a4_dimensionless_student_t_95pct_ci"])
        A4_ci = json.loads(row["A4_m4_per_s2_student_t_95pct_ci"])
        retained = result["model_retained_counts"][row["model"]]
        text.append(
            f"| {row['model']} | {retained}/1024 | {float(row['retention_fraction_mean']):.4f} "
            f"[{fraction_ci[0]:.4f}, {fraction_ci[1]:.4f}] | {float(row['a4_dimensionless_mean']):.6g} "
            f"[{a4_ci[0]:.6g}, {a4_ci[1]:.6g}] | {float(row['A4_m4_per_s2_mean']):.6g} "
            f"[{A4_ci[0]:.6g}, {A4_ci[1]:.6g}] |"
        )
    text.extend(["", "## Парные сравнения", "", "| Пара | ΔP (95% CI) | A₄ отношение средних (bootstrap 95% CI) | Несогласные пары |", "|---|---:|---:|---:|"])
    for row in result["paired_comparisons"]:
        ci = json.loads(row["left_minus_right_retention_student_t_95pct_ci"])
        ratio_ci = json.loads(row["ratio_of_means_bootstrap_95pct"])
        ratio_text = "не определено" if ratio_ci is None else f"{float(row['ratio_of_mean_retention_fractions']):.4g} [{ratio_ci[0]:.4g}, {ratio_ci[1]:.4g}]"
        text.append(f"| {row['model_left']} − {row['model_right']} | {float(row['left_minus_right_retention_fraction']):.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {ratio_text} | {row['discordant_pairs']}/{row['total_pairs']} |")
    text.extend([
        "",
        "## Ограничения",
        "",
        "Это первая оценка по 1024 общим состояниям. Её статистическая неопределённость не заменяет пространственную проверку h35/h50 и контроль полного P4 FEM для чувствительных стартов. Результат относится только к указанному домену, RF-фазе и конечному горизонту.",
        "",
        "Файлы `trajectory_results.csv`, `acceptance_estimates.csv`, `paired_acceptance_differences.csv` и `baseline_summary.json` содержат исходные статусы, времена первого выхода, интервалы и provenance. Повторный экспорт выполняется `scripts/summarize_v12_baseline.py` и не интегрирует траектории.",
    ])
    (DEFAULT_OUT / "BASELINE_REPORT_RU.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"Missing source run: {run_dir}")
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    validate_and_collect(run_dir)


if __name__ == "__main__":
    main()
