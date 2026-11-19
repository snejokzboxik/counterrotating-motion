"""Replay config17 holdout wall failures and build matched initial-condition maps."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import analyze_candidate52_failures as shared

HERE = Path(__file__).resolve().parent
V17C9 = HERE.parent / "v17c_9"
STAGE4 = V17C9 / "optimization" / "stage4_candidate48"
STARTS = STAGE4 / "holdout_starts"
OUT = HERE / "candidate17_holdout"
GROUPS = ("rest", "weak", "wide")
N = 128
RUN_TAG = "pairedholdout20261150"
SIZES = (0.0002, 0.00189)
PARAMETER_NAMES = ("A", "B", "nu", "beta", "s_m1_over_m2", "C", "lambda1", "lambda2")


def main():
    leader = json.loads((HERE / "sobol_leader_config17.json").read_text(encoding="utf-8"))
    p_values = leader["parameters"]
    p = np.asarray([p_values[k] for k in PARAMETER_NAMES], dtype=np.float64)
    qratio = float(p_values["q2_over_q1"])
    module = shared.load_scan_module()
    source_model = module.load_model()
    model = (p, *source_model[1:])

    by_group = {}
    checks = {}
    grouped_rows = {}
    all_records = []
    for group_idx, group in enumerate(GROUPS):
        tag = f"cfg17_{group}_matched_holdout_none_chi0-0_a0.0002-0.00189_{RUN_TAG}_128_15000p_64s"
        run_path = OUT / f"{tag}.jsonl"
        meta_path = OUT / f"{tag}.meta.json"
        rows = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        keyed = {int(row["index"]): row for row in rows}
        start_path = STARTS / f"{group}_N128_seed{20261150+group_idx}.npz"
        with np.load(start_path, allow_pickle=False) as frozen:
            starts = frozen["initial"].astype(np.float64)
            phases = frozen["phases"].astype(np.float64)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checks[f"{group}_row_count_and_indices"] = len(rows) == N and set(keyed) == set(range(N))
        checks[f"{group}_metadata_parameters"] = (
            meta["config_id"] == 17
            and np.array_equal(np.asarray(meta["parameters"], dtype=float), p)
            and meta["qratio"] == qratio
            and meta["periods"] == 15000
            and meta["steps"] == 64
            and meta["transient"] == 200
            and meta["hydro_mode"] == "none"
            and meta["a1_over_R"] == SIZES[0]
            and meta["a2_over_R"] == SIZES[1]
        )
        checks[f"{group}_starts_and_phases_match_frozen"] = all(
            np.array_equal(np.asarray(keyed[i]["initial"], dtype=float), starts[i])
            and np.array_equal(np.asarray(keyed[i]["phase"], dtype=float), phases[i])
            for i in range(N)
        )

        features = {i: shared.features(row) for i, row in keyed.items()}
        records = [(row, features[i]) for i, row in keyed.items()]
        all_records.extend((group, row, features[i]) for i, row in keyed.items())
        grouped_rows[group] = keyed
        categories = Counter(row["category"] for row in rows)
        wall = [row for row in rows if row["category"] == "finite_particle_wall_contact"]
        events = [
            {"index": int(row["index"]), **shared.replay_first_boundary_contact(
                module, row["initial"], row["phase"], model, qratio=qratio
            )}
            for row in wall
        ]
        event_causes = Counter(event["cause"] for event in events)
        event_times = np.asarray([event["fractional_period"] for event in events], dtype=float)
        touches = Counter(particle for event in events for particle in event.get("particles", []))
        axis_rows = [row for row in rows if row["category"] == "axis_ambiguous"]
        axis_detail = [
            {
                "index": int(row["index"]),
                "minimum_radius_over_R": row["r_min"],
                "completed_periods": int(row["completed_periods"]),
                "turns": row["turns"],
            }
            for row in axis_rows
        ]
        outer = {}
        for threshold in (0.70, 0.80, 0.85):
            selected = [row for row in rows if max(features[row["index"]]["r1"], features[row["index"]]["r2"]) >= threshold]
            failures = sum(row["category"] != "CR_screen" for row in selected)
            outer[str(threshold)] = {
                "n": len(selected), "failures": failures,
                "observed_failure_fraction": failures/len(selected) if selected else None,
            }
        by_group[group] = {
            "n": len(rows),
            "category_counts": dict(categories),
            "successes": int(categories.get("CR_screen", 0)),
            "failures": int(N - categories.get("CR_screen", 0)),
            "replayed_wall_event_causes": dict(event_causes),
            "replayed_wall_events_by_particle": {str(k): int(v) for k, v in touches.items()},
            "replayed_wall_event_timing_fractional_periods": {
                "min": float(np.min(event_times)) if len(event_times) else None,
                "median": float(np.median(event_times)) if len(event_times) else None,
                "q90": float(np.quantile(event_times, .90)) if len(event_times) else None,
                "max": float(np.max(event_times)) if len(event_times) else None,
                "within_first_period": int(np.count_nonzero(event_times < 1.0)),
                "within_two_periods": int(np.count_nonzero(event_times < 2.0)),
            },
            "replayed_wall_events": events,
            "axis_ambiguous_details": axis_detail,
            "outer_radius_subsets": outer,
        }

    area_edges = np.sqrt(0.03**2 + np.linspace(0, 1, 5)*(0.90**2-0.03**2))
    angle_edges = np.linspace(0, 2*np.pi, 5)
    z_edges = np.linspace(-0.8, 0.8, 5)
    map_specs = [
        ("r1_by_r2", "r1", "r2", area_edges, area_edges, "initial radius $r_1/R$", "initial radius $r_2/R$"),
        ("phase1_by_phase2", "phase1", "phase2", angle_edges, angle_edges, "RF phase 1 (rad)", "RF phase 2 (rad)"),
        ("z1_by_z2", "z1", "z2", z_edges, z_edges, "$z_1/R$", "$z_2/R$"),
    ]
    maps = {}
    fig, axes = plt.subplots(3, 3, figsize=(15, 13), constrained_layout=True)
    for row_index, group in enumerate(GROUPS):
        records = [(row, shared.features(row)) for row in grouped_rows[group].values()]
        for col_index, (map_name, xname, yname, xedges, yedges, xlabel, ylabel) in enumerate(map_specs):
            count, failed, rate = shared.make_probability_map(records, xname, yname, xedges, yedges)
            maps[f"{group}_{map_name}"] = {
                "x_edges": xedges.tolist(), "y_edges": yedges.tolist(),
                "counts": count.tolist(), "failures": failed.tolist(),
                "observed_failure_fraction": [[None if np.isnan(x) else float(x) for x in line] for line in rate],
            }
            shared.write_map_csv(OUT / f"failure_map_config17_{group}_{map_name}.csv", map_name, group, xedges, yedges, count, failed, rate)
            ax = axes[row_index, col_index]
            im = ax.imshow(np.ma.masked_invalid(rate), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
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
    fig.suptitle("Config 17 strict matched holdout: exploratory initial-condition failure maps (N=128 per ensemble)")
    fig.savefig(OUT / "failure_maps_config17.png", dpi=180)
    plt.close(fig)

    pooled_outer = {}
    for threshold in (0.70, 0.80, 0.85):
        selected = [row for _, row, feat in all_records if max(feat["r1"], feat["r2"]) >= threshold]
        failures = sum(row["category"] != "CR_screen" for row in selected)
        pooled_outer[str(threshold)] = {
            "n": len(selected), "failures": failures,
            "observed_failure_fraction": failures/len(selected) if selected else None,
        }
    result = {
        "study": "V17C.10 failure analysis of strict matched config17 holdout",
        "candidate_id": 17,
        "training_status": leader["status"],
        "holdout_n_per_ensemble": N,
        "periods": 15000,
        "steps_per_period": 64,
        "candidate17_parameters": p.tolist(),
        "qratio": qratio,
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "by_ensemble": by_group,
        "pooled_outer_radius_subsets": pooled_outer,
        "map_note": "Descriptive bins from the same matched N=128 Sobol ensembles used for candidate52. The two ensembles of initial conditions differ; cells are sparse and not calibrated probability estimates.",
        "maps": maps,
    }
    (OUT / "failure_analysis_config17.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_checks_pass": result["all_checks_pass"],
        "checks": checks,
        "by_ensemble": {
            group: {
                "successes": by_group[group]["successes"],
                "failures": by_group[group]["failures"],
                "wall_causes": by_group[group]["replayed_wall_event_causes"],
                "wall_timing": by_group[group]["replayed_wall_event_timing_fractional_periods"],
                "outer_radius_subsets": by_group[group]["outer_radius_subsets"],
            }
            for group in GROUPS
        },
        "pooled_outer_radius_subsets": pooled_outer,
    }, indent=2, ensure_ascii=False))
    if not result["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
