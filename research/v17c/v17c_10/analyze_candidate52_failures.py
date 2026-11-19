"""Analyze strict candidate52 holdout failure modes and exploratory basin maps."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
STAGE4 = V17C9 / "optimization" / "stage4_candidate48"
RUNS = STAGE4 / "holdout_runs"
STARTS = STAGE4 / "holdout_starts"
MANIFEST = json.loads((STAGE4 / "PAIRED_HOLDOUT_MANIFEST.json").read_text(encoding="utf-8"))
GROUPS = ("rest", "weak", "wide")
RUN_TAG = "pairedholdout20261150"
SIZES = (0.0002, 0.00189)
PARAMETERS = [
    0.03956095746691348,
    0.15032976498886072,
    2.8996277751401065,
    0.015917322789905167,
    0.11307613545330587,
    9.281679945656495e-06,
    0.1868668374144136,
    0.10359181855828002,
]
QRATIO = 3.5274259874597194
STEPS = 64
ESCAPE_R = 1.0
ESCAPE_Z = 2.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def qstats(values):
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "q10": float(np.quantile(v, 0.10)),
        "q25": float(np.quantile(v, 0.25)),
        "q75": float(np.quantile(v, 0.75)),
        "q90": float(np.quantile(v, 0.90)),
    }


def read_group(group):
    tag = f"cfg52_{group}_matched_holdout_none_chi0-0_a0.0002-0.00189_{RUN_TAG}_128_15000p_64s"
    run_path = RUNS / f"{tag}.jsonl"
    meta_path = RUNS / f"{tag}.meta.json"
    start_path = STARTS / f"{group}_N128_seed{20261150 + GROUPS.index(group)}.npz"
    rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frozen = np.load(start_path)
    expected_hashes = {x["path"]: x["sha256"] for x in MANIFEST["files"]}
    paths = {
        f"optimization/stage4_candidate48/holdout_runs/{run_path.name}": run_path,
        f"optimization/stage4_candidate48/holdout_runs/{meta_path.name}": meta_path,
        f"optimization/stage4_candidate48/holdout_starts/{start_path.name}": start_path,
    }
    hashes_ok = all(sha256(path) == expected_hashes[rel] for rel, path in paths.items())
    inputs_ok = len(rows) == 128 and len({r["index"] for r in rows}) == 128
    rows_by_index = {r["index"]: r for r in rows}
    if inputs_ok:
        inputs_ok = set(rows_by_index) == set(range(128)) and all(
            np.array_equal(np.asarray(rows_by_index[i]["initial"]), frozen["initial"][i])
            and np.array_equal(np.asarray(rows_by_index[i]["phase"]), frozen["phases"][i])
            for i in range(128)
        )
    metadata_ok = (
        meta["config_id"] == 52
        and np.array_equal(np.asarray(meta["parameters"], dtype=float), np.asarray(PARAMETERS))
        and meta["qratio"] == QRATIO
        and meta["ensemble"] == group
        and meta["n"] == 128
        and meta["periods"] == 15000
        and meta["steps"] == 64
        and meta["transient"] == 200
        and meta["hydro_mode"] == "none"
        and meta["a1_over_R"] == SIZES[0]
        and meta["a2_over_R"] == SIZES[1]
    )
    return rows, rows_by_index, frozen, meta, {
        "sha256_matches_paired_manifest": hashes_ok,
        "128_unique_rows_match_frozen_starts": inputs_ok,
        "strict_holdout_metadata_matches": metadata_ok,
    }


def features(row):
    y = np.asarray(row["initial"], dtype=float)
    r1, r2 = float(np.hypot(y[0], y[1])), float(np.hypot(y[6], y[7]))
    phi1, phi2 = float(np.mod(np.arctan2(y[1], y[0]), 2*np.pi)), float(np.mod(np.arctan2(y[7], y[6]), 2*np.pi))
    phase = np.asarray(row["phase"], dtype=float)
    dphi = float(np.angle(np.exp(1j*(phase[0]-phase[1]))))
    return {
        "r1": r1,
        "r2": r2,
        "phi1": phi1,
        "phi2": phi2,
        "z1": float(y[2]),
        "z2": float(y[8]),
        "vz1": float(y[5]),
        "vz2": float(y[11]),
        "speed1": float(np.linalg.norm(y[3:6])),
        "speed2": float(np.linalg.norm(y[9:12])),
        "r12_initial": float(np.linalg.norm(y[:3]-y[6:9])),
        "phase1": float(phase[0]),
        "phase2": float(phase[1]),
        "relative_phase": dphi,
    }


def load_scan_module():
    bundled_site = HERE.parents[2] / ".venv" / "Lib" / "site-packages"
    if bundled_site.exists() and str(bundled_site) not in sys.path:
        sys.path.insert(0, str(bundled_site))
    source = V17C9 / "baseline" / "basin_scan.py"
    spec = importlib.util.spec_from_file_location("v17c10_basin_scan", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replay_first_boundary_contact(module, initial, phase, model):
    """Replay one saved failure with the source RK4 stages and locate first boundary."""
    p, family, k, coeff = model
    dt = 2.0 * math.pi / STEPS
    y = np.asarray(initial, dtype=np.float64).copy()
    y1, y2, y3, y4, tmp = (np.empty(12, dtype=np.float64) for _ in range(5))
    yn = np.empty(12, dtype=np.float64)
    a1, a2 = SIZES

    def boundary(state, step, stage, fraction):
        dx, dy, dz = state[0]-state[6], state[1]-state[7], state[2]-state[8]
        distance = math.sqrt(dx*dx+dy*dy+dz*dz)
        if distance < a1+a2:
            return {
                "cause": "particle_particle_contact", "step": step,
                "stage": stage, "fractional_period": (step+fraction)/STEPS,
                "distance": distance,
            }
        candidates = []
        for particle, (offset, radius) in enumerate(((0, a1), (6, a2)), start=1):
            radial_excess = math.hypot(state[offset], state[offset+1])+radius-ESCAPE_R
            axial_excess = abs(state[offset+2])+radius-ESCAPE_Z
            if radial_excess > 0:
                candidates.append(("radial_wall", particle, radial_excess))
            if axial_excess > 0:
                candidates.append(("axial_wall", particle, axial_excess))
        if candidates:
            return {
                "cause": "+".join(sorted({x[0] for x in candidates})),
                "particles": sorted({x[1] for x in candidates}),
                "step": step, "stage": stage,
                "fractional_period": (step+fraction)/STEPS,
                "excesses": [
                    {"surface": surface, "particle": particle, "excess": float(excess)}
                    for surface, particle, excess in candidates
                ],
            }
        return None

    max_radius = np.array([math.hypot(y[0], y[1]), math.hypot(y[6], y[7])])
    max_abs_z = np.array([abs(y[2]), abs(y[8])])
    for step in range(15000 * STEPS):
        t = step * dt
        module._rhs(t, y, p, phase, family, k, coeff, QRATIO, y1,
                    0.0, 0, a1, a2, -1.0)
        for j in range(12):
            tmp[j] = y[j] + 0.5*dt*y1[j]
        event = boundary(tmp, step, "rk4_half_1", 0.5)
        if event:
            return event

        module._rhs(t+0.5*dt, tmp, p, phase, family, k, coeff, QRATIO, y2,
                    0.0, 0, a1, a2, -1.0)
        for j in range(12):
            tmp[j] = y[j] + 0.5*dt*y2[j]
        event = boundary(tmp, step, "rk4_half_2", 0.5)
        if event:
            return event

        module._rhs(t+0.5*dt, tmp, p, phase, family, k, coeff, QRATIO, y3,
                    0.0, 0, a1, a2, -1.0)
        for j in range(12):
            tmp[j] = y[j] + dt*y3[j]
        event = boundary(tmp, step, "rk4_full_stage", 1.0)
        if event:
            return event

        module._rhs(t+dt, tmp, p, phase, family, k, coeff, QRATIO, y4,
                    0.0, 0, a1, a2, -1.0)
        for j in range(12):
            yn[j] = y[j] + dt*(y1[j]+2*y2[j]+2*y3[j]+y4[j])/6.0
        max_radius[0] = max(max_radius[0], math.hypot(yn[0], yn[1]))
        max_radius[1] = max(max_radius[1], math.hypot(yn[6], yn[7]))
        max_abs_z[0] = max(max_abs_z[0], abs(yn[2]))
        max_abs_z[1] = max(max_abs_z[1], abs(yn[8]))
        dx, dy, dz = yn[0]-yn[6], yn[1]-yn[7], yn[2]-yn[8]
        distance = math.sqrt(dx*dx+dy*dy+dz*dz)
        if distance < a1+a2:
            return {
                "cause": "particle_particle_contact", "step": step,
                "stage": "rk4_endpoint", "fractional_period": (step+1)/STEPS,
                "distance": distance,
            }
        candidates = []
        for particle, radius in enumerate((a1, a2)):
            if max_radius[particle]+radius > ESCAPE_R:
                candidates.append(("radial_wall", particle+1, max_radius[particle]+radius-ESCAPE_R))
            if max_abs_z[particle]+radius > ESCAPE_Z:
                candidates.append(("axial_wall", particle+1, max_abs_z[particle]+radius-ESCAPE_Z))
        if candidates:
            return {
                "cause": "+".join(sorted({x[0] for x in candidates})),
                "particles": sorted({x[1] for x in candidates}),
                "step": step, "stage": "rk4_endpoint_or_running_extremum",
                "fractional_period": (step+1)/STEPS,
                "excesses": [
                    {"surface": surface, "particle": particle, "excess": float(excess)}
                    for surface, particle, excess in candidates
                ],
            }
        y[:] = yn
    return {"cause": "not_reproduced_within_horizon"}


def make_probability_map(records, xname, yname, xedges, yedges):
    count = np.zeros((len(yedges)-1, len(xedges)-1), dtype=int)
    failed = np.zeros_like(count)
    for row, feat in records:
        ix = np.searchsorted(xedges, feat[xname], side="right") - 1
        iy = np.searchsorted(yedges, feat[yname], side="right") - 1
        ix = min(max(ix, 0), len(xedges)-2)
        iy = min(max(iy, 0), len(yedges)-2)
        count[iy, ix] += 1
        failed[iy, ix] += row["category"] != "CR_screen"
    rate = np.divide(failed, count, out=np.full(count.shape, np.nan), where=count > 0)
    return count, failed, rate


def write_map_csv(path, map_type, group, xedges, yedges, count, failed, rate):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["map", "ensemble", "x_bin", "x_low", "x_high", "y_bin", "y_low", "y_high", "n", "failures", "p_failure_observed"])
        for iy in range(count.shape[0]):
            for ix in range(count.shape[1]):
                w.writerow([map_type, group, ix, xedges[ix], xedges[ix+1], iy, yedges[iy], yedges[iy+1], int(count[iy, ix]), int(failed[iy, ix]), "" if np.isnan(rate[iy, ix]) else float(rate[iy, ix])])


def main():
    module = load_scan_module()
    source_model = module.load_model()
    # The V17C.5 config fixes the electrode harmonics; candidate52 supplies its
    # own optimized eight-parameter vector in the paired holdout metadata.
    model = (np.asarray(PARAMETERS, dtype=np.float64), *source_model[1:])
    all_records = []
    grouped_rows = {}
    checks = {}
    by_group = {}
    feature_names = ("r1", "r2", "z1", "z2", "speed1", "speed2", "r12_initial", "phase1", "phase2", "relative_phase")
    for group in GROUPS:
        rows, by_index, frozen, meta, group_checks = read_group(group)
        for name, passed in group_checks.items():
            checks[f"{group}_{name}"] = bool(passed)
        grouped_rows[group] = by_index
        feats = {i: features(row) for i, row in by_index.items()}
        records = [(row, feats[i]) for i, row in by_index.items()]
        all_records.extend((group, row, feats[i]) for i, row in by_index.items())
        cats = Counter(r["category"] for r in rows)
        failed_rows = [r for r in rows if r["category"] != "CR_screen"]
        wall_rows = [r for r in failed_rows if r["category"] == "finite_particle_wall_contact"]
        replayed = []
        for row in wall_rows:
            event = replay_first_boundary_contact(module, row["initial"], row["phase"], model)
            replayed.append({"index": int(row["index"]), **event})
        replay_counts = Counter(x["cause"] for x in replayed)
        event_times = np.asarray([x["fractional_period"] for x in replayed], dtype=float)
        wall_cycles = np.asarray([r["completed_periods"] for r in wall_rows], dtype=int)
        wall_events = {
            "count": len(wall_rows),
            "completed_periods": {
                "min": int(np.min(wall_cycles)) if len(wall_cycles) else None,
                "median": float(np.median(wall_cycles)) if len(wall_cycles) else None,
                "q90": float(np.quantile(wall_cycles, .90)) if len(wall_cycles) else None,
                "max": int(np.max(wall_cycles)) if len(wall_cycles) else None,
            },
            "at_or_before_period": {
                str(t): int(np.count_nonzero(wall_cycles <= t))
                for t in (0, 1, 5, 10, 25, 100, 500, 1000)
            },
            "radial_boundary_evidence_in_saved_extrema": int(sum(
                np.any(np.asarray(r["r_max"]) + np.asarray(SIZES) > 1.0)
                for r in wall_rows
            )),
            "axial_boundary_evidence_in_saved_extrema": int(sum(
                np.any(np.asarray(r["z_max"]) + np.asarray(SIZES) > 2.0)
                for r in wall_rows
            )),
        }
        per_class = {}
        for outcome, selected in (("CR_screen", [r for r in rows if r["category"] == "CR_screen"]), ("failure", failed_rows)):
            selected_features = [feats[r["index"]] for r in selected]
            per_class[outcome] = {
                name: qstats([x[name] for x in selected_features])
                for name in feature_names
            }
        radial_outer = {}
        for threshold in (0.70, 0.80, 0.85):
            selected = [
                r for r in rows
                if max(features(r)["r1"], features(r)["r2"]) >= threshold
            ]
            radial_outer[str(threshold)] = {
                "n": len(selected),
                "failures": sum(r["category"] != "CR_screen" for r in selected),
                "observed_failure_fraction": (sum(r["category"] != "CR_screen" for r in selected)/len(selected)) if selected else None,
            }
        by_group[group] = {
            "n": len(rows),
            "category_counts": dict(cats),
            "successes": int(cats.get("CR_screen", 0)),
            "failures": int(len(failed_rows)),
            "p_cr_observed": float(cats.get("CR_screen", 0)/len(rows)),
            "wall_contact_timing": wall_events,
            "replayed_first_wall_event": {
                "counts_by_cause": dict(replay_counts),
                "timing_fractional_periods": {
                    "min": float(np.min(event_times)) if len(event_times) else None,
                    "median": float(np.median(event_times)) if len(event_times) else None,
                    "q90": float(np.quantile(event_times, .90)) if len(event_times) else None,
                    "max": float(np.max(event_times)) if len(event_times) else None,
                    "within_first_period": int(np.count_nonzero(event_times < 1.0)),
                    "within_two_periods": int(np.count_nonzero(event_times < 2.0)),
                },
                "events": replayed,
                "note": "Replayed with the saved basin_scan.py RK4 stage checks, candidate52 model, qratio and finite radii.",
            },
            "feature_quantiles_by_outcome": per_class,
            "outer_radius_subsets": radial_outer,
        }

    # Maps use equal-area radial bins (four strata) and four angular sectors.
    area_edges = np.sqrt(0.03**2 + np.linspace(0, 1, 5)*(0.90**2-0.03**2))
    angle_edges = np.linspace(0, 2*np.pi, 5)
    z_edges = np.linspace(-0.8, 0.8, 5)
    maps_json = {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    map_specs = [
        ("r1_by_r2", "r1", "r2", area_edges, area_edges, "initial radius $r_1/R$", "initial radius $r_2/R$"),
        ("phase1_by_phase2", "phase1", "phase2", angle_edges, angle_edges, "RF phase 1 (rad)", "RF phase 2 (rad)"),
        ("z1_by_z2", "z1", "z2", z_edges, z_edges, "$z_1/R$", "$z_2/R$"),
    ]
    for row_index, group in enumerate(GROUPS):
        records = [(r, features(r)) for r in grouped_rows[group].values()]
        for col_index, (map_name, xname, yname, xedges, yedges, xlabel, ylabel) in enumerate(map_specs):
            count, failed, rate = make_probability_map(records, xname, yname, xedges, yedges)
            maps_json[f"{group}_{map_name}"] = {
                "x_edges": xedges.tolist(), "y_edges": yedges.tolist(),
                "counts": count.tolist(), "failures": failed.tolist(),
                "observed_failure_fraction": [[None if np.isnan(x) else float(x) for x in line] for line in rate],
            }
            write_map_csv(HERE / f"failure_map_{group}_{map_name}.csv", map_name, group, xedges, yedges, count, failed, rate)
            ax = axes[row_index, col_index]
            masked = np.ma.masked_invalid(rate)
            im = ax.imshow(masked, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
            for iy in range(count.shape[0]):
                for ix in range(count.shape[1]):
                    if count[iy, ix]:
                        ax.text((xedges[ix]+xedges[ix+1])/2, (yedges[iy]+yedges[iy+1])/2,
                                f"{failed[iy, ix]}/{count[iy, ix]}", ha="center", va="center",
                                color="black" if rate[iy, ix] > .45 else "white", fontsize=8)
            ax.set_title(f"{group}: {map_name.replace('_', ' ')}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="observed failure fraction")
    fig.suptitle("Candidate 52 strict holdout: exploratory initial-condition failure maps (N=128 per ensemble)")
    fig.savefig(HERE / "failure_maps_candidate52.png", dpi=180)
    plt.close(fig)

    all_category_counts = Counter(row["category"] for _, row, _ in all_records)
    pooled_outer_radius = {}
    for threshold in (0.70, 0.80, 0.85):
        selected = [
            row for _, row, feat in all_records
            if max(feat["r1"], feat["r2"]) >= threshold
        ]
        failures = sum(row["category"] != "CR_screen" for row in selected)
        pooled_outer_radius[str(threshold)] = {
            "n": len(selected), "failures": failures,
            "observed_failure_fraction": failures/len(selected) if selected else None,
        }
    outcome = {
        "study": "V17C.10 P1 failure analysis of candidate52 strict holdout",
        "status": "complete descriptive analysis; failure maps are exploratory at N=128 per ensemble",
        "dataset": {
            "branch_source_commit": "071c4ef85b1f822b975aacd66e4f8802fc81dde3",
            "candidate_id": 52,
            "parameters": PARAMETERS,
            "qratio": QRATIO,
            "ensemble_sizes": {g: 128 for g in GROUPS},
            "periods": 15000,
            "steps_per_period": 64,
            "transient_periods": 200,
            "finite_radii_over_R": list(SIZES),
            "hydrodynamics": "none",
            "fixed_field_config_sha256": sha256(module.SOURCE_CFG),
            "sampling": "frozen matched scrambled Sobol ensembles from the prior independent holdout",
        },
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "category_counts_all_ensembles": dict(all_category_counts),
        "pooled_outer_radius_subsets": pooled_outer_radius,
        "by_ensemble": by_group,
        "map_note": "Failure fractions are descriptive bins from a single N=128 randomized Sobol ensemble, not IID confidence estimates.",
        "maps": maps_json,
    }
    (HERE / "failure_analysis_candidate52.json").write_text(
        json.dumps(outcome, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    compact = {
        "all_checks_pass": outcome["all_checks_pass"],
        "checks": checks,
        "category_counts": dict(all_category_counts),
        "by_ensemble": {
            g: {
                "successes": by_group[g]["successes"],
                "failures": by_group[g]["failures"],
                "categories": by_group[g]["category_counts"],
                "wall_contact_timing": by_group[g]["wall_contact_timing"],
                "replayed_first_wall_event": by_group[g]["replayed_first_wall_event"]["counts_by_cause"],
                "exact_replay_timing": by_group[g]["replayed_first_wall_event"]["timing_fractional_periods"],
                "outer_radius_subsets": by_group[g]["outer_radius_subsets"],
            }
            for g in GROUPS
        },
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
    if not outcome["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
