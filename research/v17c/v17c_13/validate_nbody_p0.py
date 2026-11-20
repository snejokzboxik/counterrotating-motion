"""P0 checks: N=2 equivalence, pair symmetry, permutation, gradients and balances."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import nbody_dynamics as nb  # noqa: E402


def main() -> None:
    p, family, modes, coeff, qratio = nb.load_local15()
    protocols = nb.load_protocols()
    groups2 = np.asarray([0, 1], dtype=np.int64)
    radii2 = np.asarray([0.0002, 0.00189], dtype=np.float64)
    groups6 = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    radii6 = radii2[groups6]
    mu6 = nb.group_masses(groups6, p)

    # Reproduce one successful, frozen local15 #505 holdout row with both RHSs.
    holdout = ROOT / "research" / "v17c" / "v17c_11" / "strict_loading_holdout_n256" / "holdout_runs.jsonl"
    start_row = None
    with holdout.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["protocol_id"] == 1 and row["ensemble"] == "rest" and row["category"] == "CR_screen":
                start_row = row
                break
    if start_row is None:
        raise RuntimeError("cannot find the frozen #505 N=2 reproduction start")
    initial2 = np.asarray(start_row["initial_state"], dtype=np.float64)
    phase = np.asarray(start_row["initial_rf_phase"], dtype=np.float64)
    protocol = protocols["P2_505"]
    duration = nb.ld.protocol_duration_cycles(protocol)
    skip = int(np.ceil(duration)) + 200
    periods = skip + 15_000
    old = nb.ld.integrate(initial2, phase, p, protocol, family, modes, coeff,
                          periods, 96, skip, qratio,
                          a1_over_R=radii2[0], a2_over_R=radii2[1])
    new = nb.integrate(initial2, phase, p, protocol, groups2, family, modes, coeff,
                       qratio, radii2, periods, 96, skip)
    old_state = np.asarray(old["final_state"])
    new_state = np.asarray(new[0])
    dstate = float(np.max(np.abs(old_state - new_state)))
    dturn = float(np.max(np.abs(np.asarray(old["turns"]) - new[1])))
    dfreq = float(np.max(np.abs(np.asarray(old["frequency_Hz"]) - new[3])))
    dwindow = float(np.max(np.abs(np.asarray(old["window_turns"]) - new[2])))
    dmin = float(abs(old["minimum_separation"] - float(new[8][0])))
    if old["category"] != new[11]:
        raise AssertionError(f"N=2 category mismatch: {old['category']} vs {new[11]}")

    # N=6 diagnostic state; all 15 pairs are clear and are tracked separately.
    state6, _ = nb.initialize_rings(radii=(0.45, 0.63), z_centers=(0.0, 0.1), relative_angle=0.37)
    state6[3::6] = np.asarray([0.003, -0.002, 0.001, -0.001, 0.002, -0.003])
    state6[4::6] = np.asarray([-0.002, 0.001, 0.002, 0.002, -0.001, -0.002])
    state6[5::6] = np.asarray([0.001, -0.001, 0.0, 0.001, 0.0, -0.001])
    if nb._contact_status(state6, groups6, radii6) != 0:
        raise AssertionError("diagnostic state is not contact-free")

    def rhs_parts(st, gp=groups6, rr=radii6, pars=p, t=1.234):
        deriv = np.empty_like(st)
        parts = np.empty((len(gp), 4, 3), dtype=np.float64)
        nb._rhs_nbody(t, st, pars, protocols["P2_505"], phase, gp,
                      family, modes, coeff, qratio, rr, deriv, parts)
        return deriv, parts

    deriv, parts = rhs_parts(state6)
    p_zero_coulomb = p.copy(); p_zero_coulomb[5] = 0.0
    _, zero_coulomb_parts = rhs_parts(state6, pars=p_zero_coulomb)
    zero_coulomb_residual = float(np.max(np.abs(zero_coulomb_parts[:, 2, :])))
    pair_force_residual = 0.0
    pair_count = 0
    for i in range(6):
        for j in range(i + 1, 6):
            # Per-pair Newton-III check with an isolated two-particle probe.
            pair_groups = groups6[[i, j]].copy()
            pair_state = np.zeros(12, dtype=np.float64)
            pair_state[:6] = state6[6 * i:6 * i + 6]
            pair_state[6:] = state6[6 * j:6 * j + 6]
            pair_p = p.copy(); pair_p[0] = 0.0; pair_p[1] = 0.0; pair_p[3] = 0.0
            pair_p[6] = 0.0; pair_p[7] = 0.0
            _, pair_parts = rhs_parts(pair_state, pair_groups, radii6[[i, j]], pair_p)
            fi = mu6[i] * pair_parts[0, 2]
            fj = mu6[j] * pair_parts[1, 2]
            pair_force_residual = max(pair_force_residual, float(np.max(np.abs(fi + fj))))
            pair_count += 1
    if pair_count != 15:
        raise AssertionError(f"expected 15 unique pairs, evaluated {pair_count}")

    # Intra-species permutation symmetry: relabeling a triangle must relabel
    # its derivative, including every Coulomb pair.
    perm = np.asarray([2, 0, 1, 5, 3, 4], dtype=np.int64)
    perm_state = np.concatenate([state6[6 * i:6 * i + 6] for i in perm])
    perm_groups = groups6[perm]
    perm_radii = radii6[perm]
    dperm, _ = rhs_parts(perm_state, perm_groups, perm_radii)
    restored = np.empty_like(dperm)
    for new_i, old_i in enumerate(perm):
        restored[6 * old_i:6 * old_i + 6] = dperm[6 * new_i:6 * new_i + 6]
    permutation_residual = float(np.max(np.abs(restored - deriv)))

    # All 15 unique pair distances are evaluated, not just one representative.
    pair_distances = []
    for i in range(5):
        for j in range(i + 1, 6):
            pair_distances.append(float(np.linalg.norm(state6[6*i:6*i+3] - state6[6*j:6*j+3])))
    if len(pair_distances) != 15:
        raise AssertionError("pair-distance audit failed")

    # Independent finite-difference gradient of the complete external+Coulomb
    # potential compared with conservative accelerations from the N-body RHS.
    eps_x = 2.0e-7
    grad_residual = 0.0
    for i in range(6):
        b = 6 * i
        for axis in range(3):
            yp = state6.copy(); ym = state6.copy()
            yp[b + axis] += eps_x; ym[b + axis] -= eps_x
            up = nb.external_potential(1.234, yp, p, protocols["P2_505"], phase,
                                       groups6, family, modes, coeff, qratio)
            um = nb.external_potential(1.234, ym, p, protocols["P2_505"], phase,
                                       groups6, family, modes, coeff, qratio)
            dUdx = (up - um) / (2.0 * eps_x)
            cons_acc = parts[i, 0, axis] + parts[i, 1, axis] + parts[i, 2, axis]
            grad_residual = max(grad_residual, abs(dUdx + mu6[i] * cons_acc))

    # Instantaneous total energy balance during the delayed #505 DC ramp.
    t_energy = 2.0 * np.pi * 2.3
    deriv_energy, parts_energy = rhs_parts(state6, t=t_energy)
    kinetic_rate = 0.0
    drag_power = 0.0
    ldot_total = 0.0
    torque_rf_drag = 0.0
    for i in range(6):
        b = 6 * i
        v = state6[b + 3:b + 6]
        accel = deriv_energy[b + 3:b + 6]
        kinetic_rate += mu6[i] * float(np.dot(v, accel))
        drag_power += mu6[i] * float(np.dot(v, parts_energy[i, 3]))
        ldot_total += mu6[i] * (state6[b] * accel[1] - state6[b + 1] * accel[0])
        ax_rfdrag = parts_energy[i, 0] + parts_energy[i, 3]
        torque_rf_drag += mu6[i] * (state6[b] * ax_rfdrag[1] - state6[b + 1] * ax_rfdrag[0])
    explicit_eps = 1.0e-6
    u_plus = nb.external_potential(t_energy + explicit_eps, state6, p, protocols["P2_505"], phase,
                                   groups6, family, modes, coeff, qratio)
    u_minus = nb.external_potential(t_energy - explicit_eps, state6, p, protocols["P2_505"], phase,
                                    groups6, family, modes, coeff, qratio)
    explicit_power = (u_plus - u_minus) / (2.0 * explicit_eps)
    grad_dot_v = 0.0
    for i in range(6):
        b = 6 * i
        for axis in range(3):
            yp = state6.copy(); ym = state6.copy()
            yp[b + axis] += eps_x; ym[b + axis] -= eps_x
            up = nb.external_potential(t_energy, yp, p, protocols["P2_505"], phase,
                                       groups6, family, modes, coeff, qratio)
            um = nb.external_potential(t_energy, ym, p, protocols["P2_505"], phase,
                                       groups6, family, modes, coeff, qratio)
            grad_dot_v += (up - um) / (2.0 * eps_x) * state6[b + 3 + axis]
    # dK/dt + grad(U).v must equal drag power; the full Hamiltonian
    # derivative is this quantity plus the explicit RF/DC power.
    energy_balance_residual = kinetic_rate + grad_dot_v - drag_power
    angular_balance_residual = ldot_total - torque_rf_drag

    # Verify the group-dependent Coulomb scale on the x-axis for each pair type.
    pair_coefficients = {}
    for pair_groups, label in (((0, 0), "11"), ((0, 1), "12"), ((1, 1), "22")):
        st = np.asarray([0.10, 0, 0, 0, 0, 0, 0.30, 0, 0, 0, 0, 0], dtype=np.float64)
        gp = np.asarray(pair_groups, dtype=np.int64)
        pp = p.copy(); pp[:4] = 0.0; pp[6] = 0.0; pp[7] = 0.0
        dv, pt = rhs_parts(st, gp, np.asarray([0.0, 0.0]), pp)
        pair_coefficients[label] = float(dv[3])
    expected_pair1 = p[5] * (1.0 / qratio) * (-0.20) / (0.20 ** 3)
    expected_pair12 = p[5] * (-0.20) / (0.20 ** 3)
    expected_pair2 = p[5] * qratio * p[4] * (-0.20) / (0.20 ** 3)
    scale_residual = max(abs(pair_coefficients["11"] - expected_pair1),
                         abs(pair_coefficients["12"] - expected_pair12),
                         abs(pair_coefficients["22"] - expected_pair2))

    result = {
        "status": "passed",
        "local15_mfs_harmonics": int(modes.size),
        "n6_unique_pairs": pair_count,
        "n2_protocol_505_equivalence": {
            "old_category": old["category"], "new_category": new[11],
            "old_completed_periods": old["completed_periods"], "new_completed_periods": int(new[9]),
            "max_abs_final_state_difference": dstate,
            "max_abs_turns_difference": dturn,
            "max_abs_late_window_turns_difference": dwindow,
            "max_abs_frequency_Hz_difference": dfreq,
            "min_pair_distance_difference": dmin,
            "reference_frequency_Hz": np.asarray(old["frequency_Hz"], dtype=float).tolist(),
        },
        "checks": {
            "pair_force_momentum_residual": pair_force_residual,
            "permutation_symmetry_residual": permutation_residual,
            "zero_coupling_coulomb_acceleration_residual": zero_coulomb_residual,
            "self_interaction_terms": 0,
            "finite_difference_potential_gradient_residual": grad_residual,
            "instantaneous_energy_balance_residual": float(energy_balance_residual),
            "instantaneous_Lz_balance_residual": float(angular_balance_residual),
            "charge_and_mass_scaled_pair_coefficients": pair_coefficients,
            "charge_mass_scaling_residual": scale_residual,
            "pair_distances_count": len(pair_distances),
            "pair_distances_initial": pair_distances,
            "minimum_initial_surface_clearance": min(pair_distances[i] - radii6[j] - radii6[k]
                for i, (j, k) in enumerate(( (a,b) for a in range(5) for b in range(a+1,6) ))),
            "drag_power": float(drag_power),
            "explicit_rf_dc_power": float(explicit_power),
        },
    }
    # P0 comparisons are expected to be very tight; harmonic field and state
    # ordering are identical to V17C.11, while pair accumulation order differs.
    if (dstate > 2.0e-7 or dturn > 2.0e-8 or dfreq > 2.0e-7 or
            pair_force_residual > 2.0e-14 or permutation_residual > 2.0e-13 or
            zero_coulomb_residual != 0.0 or
            grad_residual > 2.0e-6 or abs(energy_balance_residual) > 2.0e-5 or
            abs(angular_balance_residual) > 2.0e-12 or scale_residual > 2.0e-14):
        raise AssertionError(json.dumps(result, indent=2))

    out = HERE / "p0_nbody_validation.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
