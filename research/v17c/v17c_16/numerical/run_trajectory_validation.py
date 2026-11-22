"""Small DOP853 validation campaign for two physical trial particles.

The run set compares isolated (kappa=0) and Coulomb-coupled motion under
Stokes/Oseen drag, several deterministic starts, and both RF phases. The ideal
field is 2D-infinite-rod RF plus the Laplace DC quadrupole. This is a screening
model, not the finite 3D apparatus field.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from physical_screen import EPS0, G, LOCAL15, PAIR, build_screen
from two_particle_oseen import RFMode, local15_trial_model, rhs
from mfs_two_particle_dynamics import MFSRFField, rhs_mfs


ROOT = Path(__file__).resolve().parents[1]


def _z_gravity_equilibrium(model, i: int) -> float:
    ghat = G/(model.radial_scale_m*model.omega1_rad_s**2)
    alpha = model.charges_relative[i]/model.masses_relative[i]
    rho_p = model.densities_kg_m3[i]
    buoy = 1.0-model.air_density_kg_m3/rho_p
    return -ghat*buoy/(2.0*alpha*model.beta2)


def _make_initial(model, branch: list[dict], case: dict,
                  gravity_orientation: str = "vertical") -> np.ndarray:
    radii = case.get("radii", [branch[0]["rho"], branch[1]["rho"]])
    omegas = case.get("omegas", [branch[0]["omega"], branch[1]["omega"]])
    angles = case.get("angles", [0.0, math.pi/2])
    radial_speeds = case.get("radial_speeds", [0.0, 0.0])
    state = np.zeros((2, 6), dtype=float)
    for i in range(2):
        rho, angle = radii[i], angles[i]
        er = np.array([math.cos(angle), math.sin(angle)])
        et = np.array([-math.sin(angle), math.cos(angle)])
        state[i, :2] = rho*er
        if gravity_orientation == "vertical":
            z_default = [_z_gravity_equilibrium(model, 0),
                         _z_gravity_equilibrium(model, 1)]
        else:
            z_default = [0.0, 0.0]
        state[i, 2] = case.get("z", z_default)[i]
        state[i, 3:5] = radial_speeds[i]*er + rho*omegas[i]*et
    return state.ravel()


def _clearances(states: np.ndarray, model) -> dict:
    x = states[:, :, :3]
    D_R = 1.0/0.95
    rod_a_R = 0.05/0.95
    shield_R = 1.10/0.95
    rod_centers = D_R*np.column_stack((
        np.cos(2*np.pi*np.arange(24)/24), np.sin(2*np.pi*np.arange(24)/24)))
    particle_r_R = np.asarray(model.radii_m)/model.radial_scale_m
    rods_clearance = []
    shield_clearance = []
    for i in range(2):
        xy = x[:, i, :2]
        dist = np.linalg.norm(xy[:, None, :]-rod_centers[None, :, :], axis=2)
        rods_clearance.append(np.min(dist-rod_a_R-particle_r_R[i], axis=1))
        shield_clearance.append(shield_R-np.linalg.norm(xy, axis=1)-particle_r_R[i])
    return {"rod_clearances": np.asarray(rods_clearance),
            "shield_clearances": np.asarray(shield_clearance)}


def _run_one(payload: dict) -> dict:
    name = payload["name"]
    outdir = Path(payload["outdir"])
    model0 = local15_trial_model(coulomb_scale=payload["coulomb_scale"],
                                 gravity=True)
    gravity_orientation = payload.get("gravity_orientation", "vertical")
    ghat = G/(model0.radial_scale_m*model0.omega1_rad_s**2)
    gravity_hat = ((0.0, 0.0, -ghat) if gravity_orientation == "vertical"
                   else (-ghat, 0.0, 0.0))
    model0 = replace(model0, gravity_hat=gravity_hat)
    modes = tuple(RFMode(m.amplitude, m.frequency, m.k, phase)
                  for m, phase in zip(model0.rf_modes, payload["rf_phases"]))
    model = replace(model0, rf_modes=modes)
    eps_r = payload.get("induced_dipole_epsilon_r", 0.0)
    if eps_r > 1.0:
        polarizabilities = tuple(
            4.0*math.pi*EPS0*model.radii_m[i]**3*(eps_r-1.0)/(eps_r+2.0)
            for i in range(2))
    else:
        polarizabilities = (0.0, 0.0)
    field_kind = payload.get("field", "ideal")
    mfs_field = MFSRFField(model, payload.get("mfs_order_per_rod", 32)) \
        if field_kind == "mfs" else None
    initial = _make_initial(model, payload["branches"], payload["initial"],
                            gravity_orientation)
    nu = max(mode.frequency for mode in model.rf_modes)
    max_step = 2.0*math.pi/(payload["steps_per_fast_period"]*nu)
    sample_dt = payload["sample_dt"]
    tau_end = 2.0*math.pi*payload["primary_periods"]
    sample_t = np.arange(0.0, tau_end, sample_dt)
    sample_t = np.append(sample_t, tau_end)

    def fun(t, y):
        if mfs_field is not None:
            return rhs_mfs(t, y, model, mfs_field,
                           oseen=payload["oseen"], coulomb=True,
                           polarizabilities_SI=polarizabilities)
        return rhs(t, y, model, oseen=payload["oseen"], coulomb=True,
                   polarizabilities_SI=polarizabilities)

    def contact_event(t, y):
        z = y.reshape(2, 6)
        sep = np.linalg.norm(z[0, :3]-z[1, :3])
        rcontact = sum(model.radii_m)/model.radial_scale_m
        return sep-rcontact
    contact_event.terminal = True
    contact_event.direction = -1

    def wall_event(t, y):
        clear = _clearances(y.reshape(1, 2, 6), model)
        return float(min(np.min(clear["rod_clearances"]),
                         np.min(clear["shield_clearances"])))
    wall_event.terminal = True
    wall_event.direction = -1

    result = solve_ivp(fun, (0.0, tau_end), initial, method="DOP853",
                       t_eval=sample_t, rtol=payload["rtol"],
                       atol=payload["atol"], max_step=max_step,
                       events=(contact_event, wall_event))
    if not result.success:
        raise RuntimeError(f"{name}: {result.message}")
    times = result.t
    states = result.y.T.reshape((-1, 2, 6))
    contact_hit = bool(len(result.t_events[0]) > 0)
    wall_hit = bool(len(result.t_events[1]) > 0)
    event_index = 0 if contact_hit else 1 if wall_hit else None
    if event_index is not None:
        event_time = float(result.t_events[event_index][0])
        event_state = result.y_events[event_index][0].reshape((1, 2, 6))
        if event_time > times[-1]:
            times = np.append(times, event_time)
            states = np.concatenate([states, event_state], axis=0)
    xy_r = np.linalg.norm(states[:, :, :2], axis=2)
    angle = np.unwrap(np.arctan2(states[:, :, 1], states[:, :, 0]), axis=0)
    start = max(0, int(0.2*len(times)))
    late_t = times[start:]
    late_angle = angle[start:]
    late_r = xy_r[start:]
    omega = [float(np.polyfit(late_t, late_angle[:, i], 1)[0]) for i in range(2)]
    mean_r = np.mean(late_r, axis=0)
    sd_r = np.std(late_r, axis=0)
    distance = np.linalg.norm(states[:, 0, :3]-states[:, 1, :3], axis=1)
    clear = _clearances(states, model)
    re_coeff = [2.0*model.air_density_kg_m3*model.radii_m[i]
                * model.radial_scale_m*model.omega1_rad_s
                / model.air_dynamic_viscosity_Pa_s for i in range(2)]
    re_d = np.column_stack([re_coeff[i]*np.linalg.norm(states[:, i, 3:], axis=1)
                            for i in range(2)])
    late_re = re_d[start:]
    close = float(np.min(distance)-sum(model.radii_m)/model.radial_scale_m)
    min_rod = [float(np.min(clear["rod_clearances"][i])) for i in range(2)]
    min_shield = [float(np.min(clear["shield_clearances"][i])) for i in range(2)]
    valid_mean_window = bool(times[-1]/(2.0*math.pi) >= 50.0)
    retained = bool(not contact_hit and not wall_hit
                    and min(min_rod+min_shield) > 0.0 and close > 0.0)
    counterrotating = bool(valid_mean_window
                           and omega[0] > 0.0 and omega[1] < 0.0)
    mean_r_report = mean_r.tolist() if valid_mean_window else [None, None]
    sd_r_report = sd_r.tolist() if valid_mean_window else [None, None]
    omega_report = omega if valid_mean_window else [None, None]
    omega_dimensional_report = ((np.asarray(omega)*model.omega1_rad_s).tolist()
                                if valid_mean_window else [None, None])

    # Persist phase-resolved reduced data per case; no full solver dense output.
    path = outdir/f"{name}.npz"
    np.savez_compressed(path, tau=times, state=states)
    summary = {
        "case": name,
        "integrator": "SciPy DOP853",
        "oseen_quasisteady": payload["oseen"],
        "RF_field_model": field_kind,
        "mfs_order_per_rod": payload.get("mfs_order_per_rod") if mfs_field else None,
        "mfs_leading_transfer_check":
            mfs_field.leading_transfer_check() if mfs_field else None,
        "coulomb_scale": payload["coulomb_scale"],
        "induced_dipole_epsilon_r": eps_r,
        "polarizabilities_SI_C_m2_V": list(polarizabilities),
        "gravity_buoyancy": True,
        "gravity_orientation_relative_to_trap_axis": gravity_orientation,
        "rf_phases_rad": payload["rf_phases"],
        "initial": payload["initial"],
        "primary_RF_periods_requested": payload["primary_periods"],
        "primary_RF_periods_reached": float(times[-1]/(2.0*math.pi)),
        "termination_event": "particle_contact" if contact_hit else
            "electrode_or_shield_contact" if wall_hit else "horizon_complete",
        "last_80pct_means_valid": valid_mean_window,
        "max_step_tau": max_step,
        "steps_per_fast_RF_period_bound": payload["steps_per_fast_period"],
        "solver_nfev": int(result.nfev),
        "mean_radius_dimensionless_last_80pct": mean_r_report,
        "radius_std_dimensionless_last_80pct": sd_r_report,
        "mean_angular_velocity_dimensionless_last_80pct": omega_report,
        "mean_angular_velocity_rad_s_last_80pct": omega_dimensional_report,
        "mean_axial_position_over_R_last_80pct":
            np.mean(states[start:, :, 2], axis=0).tolist(),
        "axial_std_over_R_last_80pct":
            np.std(states[start:, :, 2], axis=0).tolist(),
        "minimum_center_distance_over_R": float(np.min(distance)),
        "minimum_surface_gap_over_R": close,
        "minimum_rod_clearance_over_R_by_particle": min_rod,
        "minimum_shield_clearance_over_R_by_particle": min_shield,
        "diameter_Re_rms_last_80pct": np.sqrt(np.mean(late_re**2, axis=0)).tolist(),
        "diameter_Re_max_sampled_last_80pct": np.max(late_re, axis=0).tolist(),
        "retained_contact_free_screen": retained,
        "counterrotating_by_mean_omega": counterrotating,
        "classification": "counterrotating-retained" if retained and counterrotating
            else "retained-not-counterrotating" if retained
            else "contact-or-wall-loss",
        "trajectory_file": path.name,
        "limitations": [
            ("RF field is the full MFS solution for a 2D infinite-rod cross-section; it is not a finite 3D electrode model."
             if field_kind == "mfs" else
             "RF field is the ideal infinite-rod leading-harmonic field, not the full 2D MFS field."),
            "Oseen drag is quasi-steady; the unsteady Stokes/Basset result is a separate linear-response control.",
            "The run horizon and fixed start set do not estimate a capture probability or prove an attractor.",
            *(["Induced-dipole sensitivity uses a constant real sphere polarizability; complex dispersion/loss, mutual dipoles, and charge-polarization coupling are omitted."] if eps_r > 1.0 else []),
        ],
    }
    (outdir/f"{name}.json").write_text(json.dumps(summary, indent=2,
                                                   ensure_ascii=False)+"\n",
                                          encoding="utf-8")
    return summary


def _branch_rows() -> list[dict]:
    data = json.loads((ROOT/"results"/"analytic_screen.json").read_text(
        encoding="utf-8"))
    return [data["particle_1"]["oseen"], data["particle_2"]["oseen"]]


def _stokes_branch_rows() -> list[dict]:
    data = json.loads((ROOT/"results"/"analytic_screen.json").read_text(
        encoding="utf-8"))
    return [data["particle_1"]["stokes"], data["particle_2"]["stokes"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=3000)
    parser.add_argument("--steps-per-fast-period", type=int, default=48)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--rtol", type=float, default=2e-9)
    parser.add_argument("--atol", type=float, default=2e-11)
    parser.add_argument("--sample-dt", type=float, default=0.25)
    parser.add_argument("--tag", type=str, default="",
                        help="optional output suffix to preserve previous checkpoints")
    parser.add_argument("--field", choices=("ideal", "mfs"), default="ideal")
    parser.add_argument("--gravity-orientation", choices=("vertical", "transverse"),
                        default="vertical")
    parser.add_argument("--induced-dipole-epsilon-r", type=float, default=0.0,
                        help="optional real dielectric epsilon_r for an induced-dipole force sensitivity")
    parser.add_argument("--mfs-order-per-rod", type=int, default=24)
    parser.add_argument("--selected-cases", nargs="*", default=None,
                        help="optional list of case names from the built-in start set")
    args = parser.parse_args()

    outdir_name = "trajectory_validation" + (f"_{args.tag}" if args.tag else "")
    outdir = ROOT/"results"/outdir_name
    outdir.mkdir(parents=True, exist_ok=True)
    oseen_branches = _branch_rows()
    stokes_branches = _stokes_branch_rows()
    base_oseen = {
        "name": "oseen_coulomb_nominal_start01",
        "coulomb_scale": 1.0, "oseen": True,
        "branches": oseen_branches,
        "initial": {"angles": [0.0, math.pi/2], "radii": None},
        "rf_phases": [0.0, 0.0],
    }
    base_oseen["initial"].pop("radii")
    specs = [
        {**base_oseen, "name": "oseen_coulomb_nominal_start01"},
        {**base_oseen, "name": "oseen_coulomb_phase_start02",
         "initial": {"angles": [math.pi/3, -math.pi/4],
                     "radii": [oseen_branches[0]["rho"]*1.01,
                               oseen_branches[1]["rho"]*0.99],
                     "radial_speeds": [0.01, -0.008]},
         "rf_phases": [math.pi/3, -math.pi/2]},
        {**base_oseen, "name": "oseen_coulomb_rest_broad_start03",
         "initial": {"angles": [-math.pi/5, math.pi/2],
                     "radii": [0.38, 0.88],
                     "omegas": [0.0, 0.0],
                     "radial_speeds": [0.0, 0.0]},
         "rf_phases": [math.pi, math.pi/4]},
        {"name": "oseen_kappa0_independent_control",
         "coulomb_scale": 0.0, "oseen": True,
         "branches": oseen_branches,
         "initial": {"angles": [0.0, math.pi/2]},
         "rf_phases": [0.0, 0.0]},
        {"name": "stokes_coulomb_nominal_control",
         "coulomb_scale": 1.0, "oseen": False,
         "branches": stokes_branches,
         "initial": {"angles": [0.0, math.pi/2]},
         "rf_phases": [0.0, 0.0]},
    ]
    if args.selected_cases:
        selected = set(args.selected_cases)
        specs = [item for item in specs if item["name"] in selected]
        missing = selected-{item["name"] for item in specs}
        if missing:
            parser.error(f"unknown case(s): {sorted(missing)}")
    for item in specs:
        item["field"] = args.field
        item["mfs_order_per_rod"] = args.mfs_order_per_rod
        item["gravity_orientation"] = args.gravity_orientation
        item["induced_dipole_epsilon_r"] = args.induced_dipole_epsilon_r
    payloads = [{**item,
                 "outdir": str(outdir),
                 "primary_periods": args.periods,
                 "steps_per_fast_period": args.steps_per_fast_period,
                 "rtol": args.rtol, "atol": args.atol,
                 "sample_dt": args.sample_dt}
                for item in specs]
    meta = {
        "campaign": "V17C.16 initial two-particle DOP853 validation; no optimization",
        "run_tag": args.tag,
        "RF_field_model": args.field,
        "gravity_orientation_relative_to_trap_axis": args.gravity_orientation,
        "induced_dipole_epsilon_r": args.induced_dipole_epsilon_r,
        "MFS_order_per_rod": args.mfs_order_per_rod if args.field == "mfs" else None,
        "primary_periods": args.periods,
        "steps_per_fast_period_bound": args.steps_per_fast_period,
        "workers": min(args.workers, len(payloads)),
        "rtol": args.rtol, "atol": args.atol, "sample_dt_tau": args.sample_dt,
        "cases": [{k: item[k] for k in ("name", "oseen", "coulomb_scale",
                                        "initial", "rf_phases")} for item in specs],
        "start_values": "fixed deterministic starts; RF phase changes included",
        "note": "Exploratory finite-horizon test. Follow-on step convergence and longer verification are required for quantitative error claims.",
    }
    (outdir/"campaign_meta.json").write_text(json.dumps(meta, indent=2,
                                                          ensure_ascii=False)+"\n",
                                               encoding="utf-8")
    results = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(payloads))) as pool:
        futures = {pool.submit(_run_one, item): item["name"] for item in payloads}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(row["case"], row["classification"],
                  row["mean_angular_velocity_dimensionless_last_80pct"],
                  row["diameter_Re_rms_last_80pct"])
    results.sort(key=lambda row: row["case"])
    summary = {"campaign_meta": meta, "results": results}
    target = outdir/"summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n",
                      encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
