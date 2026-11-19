"""Checkpointed P0 basin scan for the V17C.5 24-electrode dual-RF field.

This is a screening integrator, not the final candidate verifier. It keeps the
full signed MFS spectrum and ideal DC quadrupole from V17C.5. Each run records
only summary data and its end state; representative full trajectories should
be rerun with the validation scripts before claiming a confirmed attractor.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import ndtri
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SOURCE_CFG = PROJECT.parent / "v17c_5" / "input" / "config.json"
TWOPI = 2.0 * math.pi
RMIN_CLASS = 0.02
ESCAPE_R = 1.0
ESCAPE_Z = 2.0


@njit(cache=True, nogil=True)
def _rhs(t, y, p, phase, family, k, coeff, qratio, out,
         oseen_Re_per_speed=0.0, hydro_mode=0, a1_over_R=0.0, a2_over_R=0.0,
         oseen_Re2_per_speed=-1.0):
    A, B, nu, beta, s, coupling, lam1, lam2 = p
    ph1 = A * complex(math.cos(t + phase[0]), -math.sin(t + phase[0]))
    ph2 = B * complex(math.cos(nu * t + phase[1]), -math.sin(nu * t + phase[1]))
    dx, dy, dz = y[0]-y[6], y[1]-y[7], y[2]-y[8]
    d2 = dx*dx + dy*dy + dz*dz
    invd3 = 0.0
    if coupling != 0.0 and d2 > 1e-28:
        invd3 = 1.0 / (d2 * math.sqrt(d2))
    for i in range(2):
        j = 6*i
        x, yy, z, vx, vy, vz = y[j:j+6]
        u = complex(x, yy)
        ex, ey = 0.0, 0.0
        for h in range(k.size):
            mode = k[h]
            base = u if mode > 0 else u.conjugate()
            degree = abs(mode)
            carrier = ph1 if family[h] == 0 else ph2
            grad = carrier * coeff[h] * degree * base**(degree-1)
            ex -= grad.real
            ey += grad.imag if mode > 0 else -grad.imag
        invmass = 1.0 if i == 0 else s
        electric_factor = 1.0 if i == 0 else s*qratio
        lam = lam1 if i == 0 else lam2
        speed = math.sqrt(vx*vx+vy*vy+vz*vz)
        # First Oseen correction to Stokes drag, F_D/F_Stokes = 1+3 Re/16.
        # oseen_Re_per_speed maps dimensionless speed |v| to Re. It is zero
        # for the baseline model and only used after checking low-Re/low-Kn.
        chi = oseen_Re_per_speed if i == 0 or oseen_Re2_per_speed < 0.0 else oseen_Re2_per_speed
        drag_factor = 1.0 + 0.1875 * chi * speed
        hydro_x, hydro_y, hydro_z = 0.0, 0.0, 0.0
        if hydro_mode > 0 and d2 > 0.0:
            d = math.sqrt(d2)
            other_base = 6 if i == 0 else 0
            aj = a2_over_R if i == 0 else a1_over_R
            ai = a1_over_R if i == 0 else a2_over_R
            qx, qy, qz = (x-y[other_base])/d, (yy-y[other_base+1])/d, (z-y[other_base+2])/d
            ovx, ovy, ovz = y[other_base+3], y[other_base+4], y[other_base+5]
            longitudinal = qx*ovx+qy*ovy+qz*ovz
            g_iso, g_long = 1.0, 1.0
            if hydro_mode == 2:
                radii2 = ai*ai+aj*aj
                g_iso = 1.0+radii2/(3.0*d*d)
                g_long = 1.0-radii2/(d*d)
            hscale = 0.75*aj/d
            hydro_x = lam*hscale*(g_iso*ovx+g_long*qx*longitudinal)
            hydro_y = lam*hscale*(g_iso*ovy+g_long*qy*longitudinal)
            hydro_z = lam*hscale*(g_iso*ovz+g_long*qz*longitudinal)
        # C is q1*q2/(4 pi eps m1 R^3 Omega1^2); equal charges in P0.
        sign = 1.0 if i == 0 else -1.0
        cfac = sign * coupling * invd3
        out[j], out[j+1], out[j+2] = vx, vy, vz
        out[j+3] = electric_factor*(ex + beta*x) + invmass*cfac*dx - lam*drag_factor*vx + hydro_x
        out[j+4] = electric_factor*(ey + beta*yy) + invmass*cfac*dy - lam*drag_factor*vy + hydro_y
        out[j+5] = electric_factor*(-2.0*beta*z) + invmass*cfac*dz - lam*drag_factor*vz + hydro_z


@njit(cache=True, nogil=True)
def _finite_body_contact(y, a1_over_R, a2_over_R):
    dx=y[0]-y[6]; dy=y[1]-y[7]; dz=y[2]-y[8]
    if a1_over_R+a2_over_R>0.0 and dx*dx+dy*dy+dz*dz < (a1_over_R+a2_over_R)**2:
        return 1
    if ((a1_over_R>0.0 and (math.hypot(y[0],y[1])+a1_over_R>ESCAPE_R or abs(y[2])+a1_over_R>ESCAPE_Z)) or
        (a2_over_R>0.0 and (math.hypot(y[6],y[7])+a2_over_R>ESCAPE_R or abs(y[8])+a2_over_R>ESCAPE_Z))):
        return 2
    return 0


@njit(cache=True, nogil=True)
def integrate_one(initial, phase, p, family, k, coeff, periods, steps, skip, qratio=1.0,
                  oseen_Re_per_speed=0.0, hydro_mode=0, a1_over_R=0.0, a2_over_R=0.0,
                  oseen_Re2_per_speed=-1.0):
    dt = TWOPI / steps
    y = initial.copy()
    y1 = np.empty(12); y2 = np.empty(12); y3 = np.empty(12)
    y4 = np.empty(12); tmp = np.empty(12); yn = np.empty(12)
    total_turns = np.zeros(2)
    window_turns = np.zeros((3, 2))
    min_radius = np.array([math.hypot(y[0], y[1]), math.hypot(y[6], y[7])])
    max_radius = min_radius.copy()
    max_abs_dtheta = np.zeros(2)
    max_abs_z = np.array([abs(y[2]), abs(y[8])])
    min_distance = math.sqrt((y[0]-y[6])**2+(y[1]-y[7])**2+(y[2]-y[8])**2)
    if a1_over_R+a2_over_R > 0.0 and min_distance < a1_over_R+a2_over_R:
        particle_contact = True
    if ((a1_over_R > 0.0 and (max_radius[0]+a1_over_R > ESCAPE_R or max_abs_z[0]+a1_over_R > ESCAPE_Z)) or
        (a2_over_R > 0.0 and (max_radius[1]+a2_over_R > ESCAPE_R or max_abs_z[1]+a2_over_R > ESCAPE_Z))):
        electrode_contact = True
    steps_done = 0
    escaped = False
    singular = False
    particle_contact = False
    electrode_contact = False
    window_edges = np.array([skip, skip+(periods-skip)//3,
                             skip+2*(periods-skip)//3, periods])
    total_steps = periods * steps

    for step in range(total_steps):
        if particle_contact or electrode_contact:
            break
        t = step * dt
        _rhs(t, y, p, phase, family, k, coeff, qratio, y1, oseen_Re_per_speed, hydro_mode, a1_over_R, a2_over_R, oseen_Re2_per_speed)
        for j in range(12): tmp[j] = y[j] + 0.5*dt*y1[j]
        if a1_over_R+a2_over_R>0.0:
            dxs=tmp[0]-tmp[6]; dys=tmp[1]-tmp[7]; dzs=tmp[2]-tmp[8]
            min_distance=min(min_distance,math.sqrt(dxs*dxs+dys*dys+dzs*dzs))
        contact_kind=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact_kind==1:
            particle_contact=True; steps_done=step+1; break
        elif contact_kind==2:
            electrode_contact=True; steps_done=step+1; break
        _rhs(t+0.5*dt, tmp, p, phase, family, k, coeff, qratio, y2, oseen_Re_per_speed, hydro_mode, a1_over_R, a2_over_R, oseen_Re2_per_speed)
        for j in range(12): tmp[j] = y[j] + 0.5*dt*y2[j]
        if a1_over_R+a2_over_R>0.0:
            dxs=tmp[0]-tmp[6]; dys=tmp[1]-tmp[7]; dzs=tmp[2]-tmp[8]
            min_distance=min(min_distance,math.sqrt(dxs*dxs+dys*dys+dzs*dzs))
        contact_kind=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact_kind==1:
            particle_contact=True; steps_done=step+1; break
        elif contact_kind==2:
            electrode_contact=True; steps_done=step+1; break
        _rhs(t+0.5*dt, tmp, p, phase, family, k, coeff, qratio, y3, oseen_Re_per_speed, hydro_mode, a1_over_R, a2_over_R, oseen_Re2_per_speed)
        for j in range(12): tmp[j] = y[j] + dt*y3[j]
        if a1_over_R+a2_over_R>0.0:
            dxs=tmp[0]-tmp[6]; dys=tmp[1]-tmp[7]; dzs=tmp[2]-tmp[8]
            min_distance=min(min_distance,math.sqrt(dxs*dxs+dys*dys+dzs*dzs))
        contact_kind=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact_kind==1:
            particle_contact=True; steps_done=step+1; break
        elif contact_kind==2:
            electrode_contact=True; steps_done=step+1; break
        _rhs(t+dt, tmp, p, phase, family, k, coeff, qratio, y4, oseen_Re_per_speed, hydro_mode, a1_over_R, a2_over_R, oseen_Re2_per_speed)
        for j in range(12):
            yn[j] = y[j] + dt*(y1[j]+2*y2[j]+2*y3[j]+y4[j])/6.0

        r0 = math.hypot(y[0], y[1]); r1 = math.hypot(yn[0], yn[1])
        r2 = math.hypot(y[6], y[7]); r3 = math.hypot(yn[6], yn[7])
        min_radius[0] = min(min_radius[0], r0, r1)
        min_radius[1] = min(min_radius[1], r2, r3)
        max_radius[0] = max(max_radius[0], r0, r1)
        max_radius[1] = max(max_radius[1], r2, r3)
        max_abs_z[0] = max(max_abs_z[0], abs(y[2]), abs(yn[2]))
        max_abs_z[1] = max(max_abs_z[1], abs(y[8]), abs(yn[8]))
        ddx, ddy, ddz = yn[0]-yn[6], yn[1]-yn[7], yn[2]-yn[8]
        d2 = ddx*ddx+ddy*ddy+ddz*ddz
        min_distance = min(min_distance, math.sqrt(d2))
        if a1_over_R+a2_over_R > 0.0 and d2 < (a1_over_R+a2_over_R)**2:
            particle_contact = True
            steps_done = step+1
            y[:] = yn
            break
        if ((a1_over_R > 0.0 and (max_radius[0]+a1_over_R > ESCAPE_R or max_abs_z[0]+a1_over_R > ESCAPE_Z)) or
            (a2_over_R > 0.0 and (max_radius[1]+a2_over_R > ESCAPE_R or max_abs_z[1]+a2_over_R > ESCAPE_Z))):
            electrode_contact = True
            steps_done = step+1
            y[:] = yn
            break
        if d2 < 1e-10:
            singular = True
            steps_done = step+1
            break

        cycle = step // steps
        if cycle >= skip:
            win = 2
            if cycle < window_edges[1]: win = 0
            elif cycle < window_edges[2]: win = 1
            for i, (a, b) in enumerate(((0, 1), (6, 7))):
                rr = y[a]*y[a] + y[b]*y[b]
                rn = yn[a]*yn[a] + yn[b]*yn[b]
                if rr > RMIN_CLASS**2 and rn > RMIN_CLASS**2:
                    cross = y[a]*yn[b] - y[b]*yn[a]
                    dot = y[a]*yn[a] + y[b]*yn[b]
                    dtheta = math.atan2(cross, dot) / TWOPI
                    total_turns[i] += dtheta
                    window_turns[win, i] += dtheta
                    max_abs_dtheta[i] = max(max_abs_dtheta[i], abs(dtheta)*TWOPI)

        if not np.all(np.isfinite(yn)):
            singular = True
            steps_done = step+1
            break
        if max_radius[0] > ESCAPE_R or max_radius[1] > ESCAPE_R or max_abs_z[0] > ESCAPE_Z or max_abs_z[1] > ESCAPE_Z:
            escaped = True
            steps_done = step+1
            y[:] = yn
            break
        y, yn = yn, y
        steps_done = step+1

    completed = steps_done // steps
    retained = not escaped and not singular and not particle_contact and not electrode_contact and completed >= periods
    axis_ambiguous = min_radius[0] < RMIN_CLASS or min_radius[1] < RMIN_CLASS
    # Conservative finite-horizon classification; all non-CR outcomes remain
    # in the unconditional denominator. Independent refinement is required.
    stable_sign = True
    signs = np.zeros((3, 2))
    for w in range(3):
        for i in range(2):
            signs[w, i] = 0.0 if abs(window_turns[w, i]) < 0.05 else math.copysign(1.0, window_turns[w, i])
            if signs[w, i] == 0.0:
                stable_sign = False
    opposite = signs[0,0]*signs[0,1] < 0 and signs[1,0]*signs[1,1] < 0 and signs[2,0]*signs[2,1] < 0
    enough_turns = abs(total_turns[0]) >= 10.0 and abs(total_turns[1]) >= 10.0
    if singular:
        category = "point_singularity_unresolved"
    elif particle_contact:
        category = "finite_particle_contact"
    elif electrode_contact:
        category = "finite_particle_wall_contact"
    elif escaped:
        category = "lost_retention"
    elif axis_ambiguous:
        category = "axis_ambiguous"
    elif retained and enough_turns and stable_sign and opposite:
        category = "CR_screen"
    elif retained and enough_turns and stable_sign and signs[0,0]*signs[0,1] > 0:
        category = "same_direction"
    elif retained and abs(total_turns[0]) < 0.25 and abs(total_turns[1]) < 0.25:
        category = "localized_or_slow"
    elif retained:
        category = "inconclusive"
    else:
        category = "incomplete"
    freq = total_turns * 100000.0 / max(1, completed-skip)
    return (y, total_turns, window_turns, min_radius, max_radius, max_abs_z, max_abs_dtheta,
            min_distance, completed, retained, escaped, singular, category, freq)


def make_initials(name: str, count: int, seed: int, p_lo=0.03, p_hi=0.90):
    dims = 8 if name == "rest" else 14
    u = qmc.Sobol(d=dims, scramble=True, seed=seed).random_base2(int(math.log2(count)))
    out = np.zeros((count, 12), dtype=np.float64)
    rad = lambda x: np.sqrt(p_lo*p_lo + x*(p_hi*p_hi-p_lo*p_lo))
    out[:, 0] = rad(u[:, 0])*np.cos(TWOPI*u[:, 1])
    out[:, 1] = rad(u[:, 0])*np.sin(TWOPI*u[:, 1])
    out[:, 6] = rad(u[:, 2])*np.cos(TWOPI*u[:, 3])
    out[:, 7] = rad(u[:, 2])*np.sin(TWOPI*u[:, 3])
    zmax = 0.5 if name != "wide" else 0.8
    out[:, 2] = (2*u[:, 4]-1)*zmax
    out[:, 8] = (2*u[:, 5]-1)*zmax
    phases = TWOPI*u[:, 6:8]
    if name != "rest":
        sigma = 0.01 if name == "weak" else 0.05
        v = np.clip(u[:, 8:14], 1e-7, 1-1e-7)
        out[:, [3,4,5,9,10,11]] = sigma*ndtri(v)
    return out, phases


def load_model():
    cfg = json.loads(SOURCE_CFG.read_text(encoding="utf-8-sig"))
    p = np.asarray(cfg["parameters"], dtype=np.float64)
    coeff = np.asarray(cfg["coeff"], dtype=np.float64)
    coeff = coeff[:,0] + 1j*coeff[:,1]
    return p, np.asarray(cfg["family"], dtype=np.int64), np.asarray(cfg["k"], dtype=np.int64), coeff


def run(args):
    p, family, k, coeff = load_model()
    outdir = HERE / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.initials_file:
        frozen=np.load(args.initials_file)
        initial=frozen["initial"].copy(); phases=frozen["phases"].copy()
        if len(initial)!=args.count: raise ValueError("initial set size differs from --count")
    else:
        initial, phases = make_initials(args.ensemble, args.count, args.seed)
    np.savez_compressed(outdir/f"{args.ensemble}_initial_conditions_{args.steps}s.npz",
                        initial=initial, phases=phases, seed=args.seed,
                        p=p, periods=args.periods, steps=args.steps,
                        transient=args.skip, family=family, k=k, coeff=coeff)
    result_path = outdir/f"{args.ensemble}_{args.periods}p_{args.steps}s.jsonl"
    meta_path = result_path.with_suffix(".meta.json")
    meta = {"ensemble":args.ensemble,"count":args.count,"seed":args.seed,
            "periods":args.periods,"steps":args.steps,"skip":args.skip,
            "parameters":p.tolist(),"phase_sampling":"independent Sobol uniform [0,2pi)"}
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise RuntimeError(f"Checkpoint metadata mismatch: {meta_path}")
    meta_path.write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    done=set()
    if result_path.exists():
        with result_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip(): done.add(json.loads(line)["index"])
    todo=[i for i in range(args.count) if i not in done]
    if args.bench:
        todo=todo[:args.bench]
    print(f"{args.ensemble}: n={args.count}, already={len(done)}, todo={len(todo)}, "
          f"horizon={args.periods} periods, steps={args.steps}, workers={args.threads}",flush=True)

    # Compile once before threads begin, avoiding concurrent first-call compilation.
    if todo:
        integrate_one(initial[todo[0]], phases[todo[0]], p, family, k, coeff,
                      min(args.periods,2), args.steps, 0)
    t0=time.perf_counter()
    def one(i):
        r=integrate_one(initial[i], phases[i], p, family, k, coeff,
                        args.periods, args.steps, args.skip)
        (end, turns, windows, rmin, rmax, zmax, max_dtheta, dmin, completed, retained,
         escaped, singular, category, freq)=r
        return {"index":int(i),"seed":args.seed,"initial":initial[i].tolist(),
                "phase":phases[i].tolist(),"final_state":end.tolist(),
                "turns":turns.tolist(),"window_turns":windows.tolist(),
                "frequency_Hz":freq.tolist(),"r_min":rmin.tolist(),"r_max":rmax.tolist(),
                "abs_z_max":zmax.tolist(),"max_abs_inner_step_dtheta_rad":max_dtheta.tolist(),
                "distance_min":float(dmin),
                "completed_periods":int(completed),"retained":bool(retained),
                "escaped":bool(escaped),"singular_unresolved":bool(singular),
                "category":str(category)}
    with result_path.open("a",encoding="utf-8",buffering=1) as fout:
        with ThreadPoolExecutor(max_workers=args.threads) as pool:
            futures={pool.submit(one,i):i for i in todo}
            for n,future in enumerate(as_completed(futures),1):
                row=future.result()
                fout.write(json.dumps(row,separators=(",",":"))+"\n")
                if n%16==0 or n==len(todo):
                    elapsed=time.perf_counter()-t0
                    print(f"{args.ensemble}: checkpoint {n}/{len(todo)} "
                          f"({elapsed:.1f} s elapsed)",flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ensemble",choices=["rest","weak","wide"],required=True)
    ap.add_argument("--count",type=int,default=512)
    ap.add_argument("--seed",type=int,default=20261119)
    ap.add_argument("--periods",type=int,default=5000)
    ap.add_argument("--steps",type=int,default=32)
    ap.add_argument("--skip",type=int,default=200)
    ap.add_argument("--threads",type=int,default=6)
    ap.add_argument("--bench",type=int,default=0,help="run only first N pending for a throughput pilot")
    ap.add_argument("--initials-file",type=Path,default=None,help="reuse a frozen ensemble from an earlier step size")
    args=ap.parse_args()
    if args.count & (args.count-1):
        raise ValueError("Sobol ensemble sizes must be powers of two")
    run(args)


if __name__=="__main__": main()
