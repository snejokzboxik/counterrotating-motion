"""Fixed-radius 3+3 collective phase-locking continuation from V17C.13 C=0 data.

This is an analytic reduced model, not a new N-body trajectory integration.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from make_figures import P

ROOT=Path(__file__).resolve().parents[1]
C0_RUN={
    "source_branch":"research/v17c-crystals",
    "source_commit":"5438b1cdd79c47abab45c6cbaf99c6daec9fbb60",
    "source_file":"research/v17c/v17c_13/p1_coulomb_scan/coulomb_runs.jsonl",
    "task_id":"C_0",
    "group_mean_frequency_Hz":[157.29805890180182,-100.47839308168994],
    "group_mean_radius_R_each":[
        [0.5486779623572318,0.5486456177184672,0.5488988362808312],
        [0.7653965198646414,0.7655284898631348,0.7655379310651367],
    ],
    "frequency_normalization_Hz":100000.0,
    "post_load_periods":15000,
}


def bisection(fun, lo, hi, iterations=60):
    flo=fun(lo)
    for _ in range(iterations):
        mid=0.5*(lo+hi); fm=fun(mid)
        if flo*fm<=0: hi=mid
        else: lo=mid;flo=fm
    return 0.5*(lo+hi)


def main():
    rhos=[float(np.mean(q)) for q in C0_RUN["group_mean_radius_R_each"]]
    omega0=[float(f/C0_RUN["frequency_normalization_Hz"])
            for f in C0_RUN["group_mean_frequency_Hz"]]
    mu=[1.0,float(P["m2_over_m1"])]
    lam=[float(P["lambda1"]),float(P["lambda2"])]
    inertia=[3.0*mu[i]*rhos[i]**2 for i in range(2)]
    rf_torque=[lam[i]*inertia[i]*omega0[i] for i in range(2)]
    omega_lock=float(sum(rf_torque)/sum(lam[i]*inertia[i] for i in range(2)))
    required_torque=float(lam[0]*inertia[0]*(omega_lock-omega0[0]))
    c0=float(P["C0"])
    phase_period=2.0*np.pi/3.0

    def unit_pair_torque(delta):
        result=0.0
        for j in range(3):
            for ell in range(3):
                angle=delta+2*np.pi*j/3-2*np.pi*ell/3
                d2=rhos[0]**2+rhos[1]**2-2*rhos[0]*rhos[1]*np.cos(angle)
                result += rhos[0]*rhos[1]*np.sin(angle)/d2**1.5
        return float(result)

    def unit_pair_torque_derivative(delta):
        result=0.0
        for j in range(3):
            for ell in range(3):
                angle=delta+2*np.pi*j/3-2*np.pi*ell/3
                sn=np.sin(angle); cs=np.cos(angle)
                d2=rhos[0]**2+rhos[1]**2-2*rhos[0]*rhos[1]*cs
                result += rhos[0]*rhos[1]*(cs/d2**1.5-
                    3*rhos[0]*rhos[1]*sn**2/d2**2.5)
        return float(result)

    grid=np.linspace(0.0,phase_period,20001)
    torque_vals=np.array([unit_pair_torque(d) for d in grid])
    imin=int(np.argmin(torque_vals)); phase_guess=float(grid[imin])
    h=2*phase_period/(len(grid)-1)
    critical_phase=bisection(unit_pair_torque_derivative,phase_guess-h,phase_guess+h)
    minimum_torque_per_C0=unit_pair_torque(critical_phase)
    ccrit=abs(required_torque)/(c0*abs(minimum_torque_per_C0))

    scales=[0.0,2.0,3.0,float(ccrit),3.1,4.0,5.0]
    branch_rows=[]
    for scale in scales:
        def residual(d): return scale*c0*unit_pair_torque(d)-required_torque
        vals=np.array([residual(d) for d in grid])
        roots=[]
        if scale>ccrit*(1.0+1e-9):
            for idx in range(len(grid)-1):
                if vals[idx]*vals[idx+1] < 0.0:
                    delta=bisection(residual,float(grid[idx]),float(grid[idx+1]))
                    mc=scale*c0*unit_pair_torque_derivative(delta)
                    jac=np.array([[-lam[0],0.0,mc/inertia[0]],
                                  [0.0,-lam[1],-mc/inertia[1]],
                                  [1.0,-1.0,0.0]])
                    eig=np.linalg.eigvals(jac)
                    roots.append({"relative_phase_rad":delta,
                        "coulomb_torque_on_group1":scale*c0*unit_pair_torque(delta),
                        "torque_slope":mc,
                        "eigenvalues_real_imag":[[float(q.real),float(q.imag)] for q in eig],
                        "linearly_stable_reduced_phase_model":bool(np.max(eig.real)<0.0)})
        if scale < ccrit*(1.0-1e-9): status="no_phase_locked_co-rotation_equilibrium"
        elif abs(scale-ccrit)<=1e-9: status="fold_threshold_one_neutral_phase_mode"
        else: status="two_locked_equilibria_one_stable_one_unstable"
        branch_rows.append({"C_over_C0":scale,"status":status,"equilibria":roots})

    result={
        "model":"two rigid fixed-radius 3-particle rings; collective angles only; exact 9-pair Coulomb phase torque",
        "source_run":C0_RUN,
        "parameters":{"radii_R":rhos,"group_inertias":inertia,"lambda":lam,
            "C0":c0,"RF_torques_calibrated_from_C0":rf_torque,
            "C0_group_angular_speeds":omega0},
        "method":"Calibrate constant RF torque as drag torque at the C=0 measured mean speed. Solve the exact fixed-radius pair-torque balance for common group speed and relative phase. No trajectory is integrated.",
        "required_lock":{"common_omega_over_Omega1":omega_lock,
            "required_coulomb_torque_on_group1":required_torque},
        "fold_prediction":{"Ccrit_over_C0":ccrit,"relative_phase_rad":critical_phase,
            "unit_torque_per_C0_at_extremum":minimum_torque_per_C0,
            "reduced_jacobian_eigenvalues_real_imag":[[float(q.real),float(q.imag)] for q in np.linalg.eigvals(np.array([[-lam[0],0.0,ccrit*c0*unit_pair_torque_derivative(critical_phase)/inertia[0]],[0.0,-lam[1],-ccrit*c0*unit_pair_torque_derivative(critical_phase)/inertia[1]],[1.0,-1.0,0.0]]))]},
        "continuation_scales":branch_rows,
        "comparison_to_source":"The predicted reduced-model phase-lock threshold is inside the one-way V17C.13 observed counter-to-co transition bracket (2,5]. The predicted common locked frequency is close to the C/C0=5 group frequencies; this is agreement of a calibrated reduced model, not proof of the full N-body transition mechanism.",
        "trajectory_integrations":0,
        "limitations":["fixed radii and rigid ideal triangles", "constant RF torques calibrated from one finite-horizon C=0 start", "no radial response, internal modes, MFS angular modulation, or loading history", "the static phase-lock criterion does not classify basin capture or establish full-system hysteresis"]
    }
    (ROOT/"results/reduced_collective_locking.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"fold_prediction":result["fold_prediction"],"branches":branch_rows},indent=2))


if __name__=="__main__":main()
