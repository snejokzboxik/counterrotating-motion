"""P1 early-time diagnostics on published matched local15/candidate52 starts."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V10 = HERE.parent / "v17c_10"
V9 = HERE.parent / "v17c_9"
sys.path.insert(0, str(V9 / "baseline"))
import basin_scan  # noqa: E402
import loading_dynamics as timed  # noqa: E402

GROUPS = ("rest", "weak", "wide", "edge")
LABELS = {"local15": ("local15", 15), "candidate52": ("candidate52", 36)}
SAMPLES = np.asarray([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0])
PERIODS, STEPS = 10, 128
R1, R2 = 0.0002, 0.00189
WORKERS = 4


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources(label: str, group: str):
    stem = f"{label}_{group}_matched_strict_none_a0.0002-0.00189_replicate20261410_256_15000p_64s"
    path = V10 / "focused_edge_replication" / f"{stem}.jsonl"
    meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    start_path = HERE.parents[2] / Path(meta["starts"])
    if sha256(start_path) != meta["frozen_start_sha256"]:
        raise RuntimeError(f"frozen-start SHA mismatch: {start_path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            i = int(r["index"])
            if i in rows:
                raise RuntimeError(f"duplicate row {label}/{group}/{i}")
            rows[i] = r
    if len(rows) != 256:
        raise RuntimeError(f"expected 256 rows, got {len(rows)}: {path}")
    with np.load(start_path, allow_pickle=False) as st:
        for i, r in rows.items():
            if not np.array_equal(np.asarray(r["initial"], float), st["initial"][i]):
                raise RuntimeError(f"initial differs from frozen array: {label}/{group}/{i}")
            if not np.array_equal(np.asarray(r["phase"], float), st["phases"][i]):
                raise RuntimeError(f"phase differs from frozen array: {label}/{group}/{i}")
    return rows, path, start_path


def p_for(label: str):
    if label == "local15":
        d = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
        return np.asarray(d["candidate"]["p"], float), float(d["candidate"]["q2_over_q1"])
    return np.asarray([0.03956095746691348,0.15032976498886072,2.8996277751401065,
        0.015917322789905167,0.11307613545330587,9.281679945656495e-06,
        0.1868668374144136,0.10359181855828002]), 3.5274259874597194


def force_components(tau, states, p, phase, family, modes, coeff, qratio):
    """Return acceleration vectors and mass-weighted kinetic power by source."""
    n = len(tau)
    acc = np.zeros((n, 2, 5, 3), dtype=np.float64)
    mass = (1.0, 1.0 / p[4])
    for i, base in enumerate((0, 6)):
        x, y, z = states[:, base], states[:, base+1], states[:, base+2]
        vx, vy, vz = states[:, base+3], states[:, base+4], states[:, base+5]
        u = x + 1j*y
        electric_factor = 1.0 if i == 0 else p[4]*qratio
        invmass = 1.0 if i == 0 else p[4]
        for h, mode in enumerate(modes):
            fam = int(family[h]); degree = abs(int(mode))
            amplitude = p[fam]
            omega = 1.0 if fam == 0 else p[2]
            carrier = amplitude * np.exp(-1j*(omega*tau + phase[fam]))
            base_u = u if mode > 0 else np.conjugate(u)
            grad = carrier * coeff[h] * degree * np.power(base_u, degree-1)
            acc[:, i, fam, 0] += -electric_factor * grad.real
            acc[:, i, fam, 1] += electric_factor * (grad.imag if mode > 0 else -grad.imag)
        acc[:, i, 2, 0] = electric_factor*p[3]*x
        acc[:, i, 2, 1] = electric_factor*p[3]*y
        acc[:, i, 2, 2] = -2.0*electric_factor*p[3]*z
        other = 6 if i == 0 else 0
        delta = states[:, base:base+3] - states[:, other:other+3]
        d = np.linalg.norm(delta, axis=1)
        # ``delta`` is particle i minus its partner, so the repulsive
        # acceleration has the same positive sign for both particles.
        acc[:, i, 3, :] = (invmass*p[5]/np.maximum(d,1e-14)**3)[:,None]*delta
        acc[:, i, 4, 0] = -p[6+i]*vx
        acc[:, i, 4, 1] = -p[6+i]*vy
        acc[:, i, 4, 2] = -p[6+i]*vz
    velocity = states[:, [3,4,5,9,10,11]].reshape(n,2,3)
    powers = np.stack([mass[0]*np.einsum('nfc,nc->nf',acc[:,0],velocity[:,0]),
                       mass[1]*np.einsum('nfc,nc->nf',acc[:,1],velocity[:,1])],axis=1)
    return acc, powers


def sample_indices(tau):
    return np.asarray([int(np.argmin(np.abs(tau - x*2.0*math.pi))) for x in SAMPLES],dtype=int)


def run_one(job, field):
    label, group, row = job
    p,qratio=p_for(label)
    initial=np.asarray(row["initial"],dtype=np.float64)
    phase=np.asarray(row["phase"],dtype=np.float64)
    result=timed.integrate(initial,phase,p,timed.constant_protocol(),field[0],field[1],field[2],
        PERIODS,STEPS,0,qratio,R1,R2,log_every=1)
    tau=result["log_tau"]
    state=result["log_state"]
    acc,power=force_components(tau,state,p,phase,field[0],field[1],field[2],qratio)
    if len(tau)>1:
        dt=np.diff(tau)
        work=np.zeros((len(tau),2,5),dtype=np.float64)
        work[1:]=np.cumsum(0.5*(power[:-1]+power[1:])*dt[:,None,None],axis=0)
    else:
        work=np.zeros((len(tau),2,5),dtype=np.float64)
    idx=sample_indices(tau)
    valid=(SAMPLES*2.0*math.pi <= tau[-1]+1e-14)
    sampled_state=np.full((len(SAMPLES),12),np.nan)
    sampled_acc=np.full((len(SAMPLES),2,5,3),np.nan)
    sampled_radial_acc=np.full((len(SAMPLES),2,6),np.nan)
    sampled_radial_velocity=np.full((len(SAMPLES),2),np.nan)
    sampled_wall_gap=np.full((len(SAMPLES),2),np.nan)
    sampled_power=np.full((len(SAMPLES),2,5),np.nan)
    sampled_work=np.full((len(SAMPLES),2,5),np.nan)
    actual_cycles=np.full(len(SAMPLES),np.nan)
    for j,ix in enumerate(idx):
        if valid[j]:
            sampled_state[j]=state[ix]
            sampled_acc[j]=acc[ix]
            for i,base in enumerate((0,6)):
                x,y=state[ix,base],state[ix,base+1]
                vx,vy=state[ix,base+3],state[ix,base+4]
                radius=math.hypot(x,y)
                if radius>1e-12:
                    radial=np.asarray([x,y,0.0])/radius
                    sampled_radial_acc[j,i,:5]=acc[ix,i]@radial
                    vt=(-y*vx+x*vy)/radius
                    sampled_radial_acc[j,i,5]=np.sum(sampled_radial_acc[j,i,:5])+vt*vt/radius
                    sampled_radial_velocity[j,i]=(x*vx+y*vy)/radius
                sampled_wall_gap[j,i]=1.0-radius-(R1 if i==0 else R2)
            sampled_power[j]=power[ix]
            sampled_work[j]=work[ix]
            actual_cycles[j]=tau[ix]/(2.0*math.pi)
    init_r=np.asarray([math.hypot(initial[0],initial[1]),math.hypot(initial[6],initial[7])])
    event_tau=float(tau[-1]) if row["category"]=="finite_particle_wall_contact" else None
    event_particle=None
    if event_tau is not None:
        last=state[-1]
        gaps=np.asarray([1.0-math.hypot(last[0],last[1])-R1,1.0-math.hypot(last[6],last[7])-R2])
        event_particle=int(np.argmin(gaps)+1)
    end_radius=np.asarray([math.hypot(state[-1,0],state[-1,1]),math.hypot(state[-1,6],state[-1,7])])
    return {
        "key":f"{label}:{group}:{row['index']}","candidate":label,"group":group,"index":int(row["index"]),
        "published_category":row["category"],"event_cycle":None if event_tau is None else event_tau/(2.0*math.pi),
        "event_particle_by_final_boundary_gap":event_particle,"initial_radius_R":init_r.tolist(),
        "initial_max_radius_R":float(init_r.max()),"initial_state":initial.tolist(),"rf_phase_rad":phase.tolist(),
        "early_end_cycle":float(tau[-1]/(2.0*math.pi)),"early_end_state":state[-1].tolist(),
        "early_end_radius_R":end_radius.tolist(),
        "early_end_wall_gap_R":[1.0-end_radius[0]-R1,1.0-end_radius[1]-R2],
        "integrated_work_to_early_end_by_particle_RF1_RF2_DC_Coulomb_drag":work[-1].tolist(),
        "sample_cycle_requested":SAMPLES.tolist(),"sample_cycle_actual":actual_cycles.tolist(),
        "sample_valid":valid.tolist(),"sample_state":sampled_state,"sample_acceleration":sampled_acc,
        "sample_radial_acceleration_components_and_total":sampled_radial_acc,
        "sample_radial_velocity":sampled_radial_velocity,"sample_wall_gap":sampled_wall_gap,
        "sample_power":sampled_power,"sample_work":sampled_work,
        "minimum_separation_R":float(result["minimum_separation"]),
    }


def main():
    _,family,modes,coeff=basin_scan.load_model()
    field=(family,modes,coeff)
    jobs=[]; source_info={}; grouped={}
    for group in GROUPS:
        grouped[group]={}
        for label in LABELS:
            rows,path,starts=sources(label,group)
            grouped[group][label]=rows
            source_info[f"{label}/{group}"]={"runs_sha256":sha256(path),"starts_sha256":sha256(starts)}
        for label in LABELS:
            rows=grouped[group][label]
            # All 256 published starts are replayed through the same early window.
            for i in sorted(rows):
                jobs.append((label,group,rows[i]))
    # Confirm the diagnostic force partition sums back to the exact validated RHS.
    force_errors=[]
    for label in LABELS:
        row=grouped["rest"][label][0]
        p,q=p_for(label); state=np.asarray(row["initial"],float); phase=np.asarray(row["phase"],float)
        tau=0.371
        rhs=np.empty(12,dtype=np.float64)
        timed._rhs_enveloped(tau,state,p,timed.constant_protocol(),phase,*field,q,rhs)
        accel,_=force_components(np.asarray([tau]),state[None,:],p,phase,*field,q)
        got=accel[0].sum(axis=1)
        reference=np.stack((rhs[3:6],rhs[9:12]))
        err=float(np.max(np.abs(got-reference)))
        force_errors.append({"candidate":label,"max_abs_acceleration_difference":err})
        if err>2e-13:
            raise RuntimeError(f"force diagnostic decomposition differs from RHS: {label} {err}")
    # Warm the logging specialization before worker threads start.
    warm=jobs[0]; p,q=p_for(warm[0]); basin_scan.integrate_one(
        np.asarray(warm[2]["initial"]),np.asarray(warm[2]["phase"]),p,*field,
        2,STEPS,0,qratio=q,a1_over_R=R1,a2_over_R=R2,oseen_Re2_per_speed=0.0)
    timed.integrate(np.asarray(warm[2]["initial"]),np.asarray(warm[2]["phase"]),p,
        timed.constant_protocol(),*field,2,STEPS,0,q,R1,R2,log_every=1)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows=list(pool.map(lambda j:run_one(j,field),jobs))
    elapsed=time.perf_counter()-start
    out_dir=HERE/"p1_early_diagnostics"; out_dir.mkdir(exist_ok=True)
    records=[]
    arrays={k:[] for k in ("sample_state","sample_acceleration","sample_radial_acceleration_components_and_total",
                           "sample_radial_velocity","sample_wall_gap","sample_power","sample_work")}
    for r in rows:
        records.append({k:v for k,v in r.items() if k not in arrays})
        for k in arrays: arrays[k].append(r[k])
    jsonl=out_dir/"early_cases.jsonl"
    with jsonl.open("w",encoding="utf-8",newline="\n") as f:
        for r in records: f.write(json.dumps(r,separators=(",",":"),ensure_ascii=False)+"\n")
    np.savez_compressed(out_dir/"early_samples.npz",
        keys=np.asarray([r["key"] for r in rows]),
        sample_cycle_requested=SAMPLES,
        sample_cycle_actual=np.asarray([r["sample_cycle_actual"] for r in rows]),
        sample_valid=np.asarray([r["sample_valid"] for r in rows],dtype=bool),
        sample_state=np.asarray(arrays["sample_state"]),
        sample_acceleration=np.asarray(arrays["sample_acceleration"]),
        sample_radial_acceleration_components_and_total=np.asarray(arrays["sample_radial_acceleration_components_and_total"]),
        sample_radial_velocity=np.asarray(arrays["sample_radial_velocity"]),
        sample_wall_gap=np.asarray(arrays["sample_wall_gap"]),
        sample_power=np.asarray(arrays["sample_power"]),
        sample_work=np.asarray(arrays["sample_work"]))
    # Aggregate unconditional early wall rates by terminal parameters and original group.
    counts={}
    for label in LABELS:
        counts[label]={}
        for group in GROUPS:
            rs=grouped[group][label].values()
            cc={}
            for r in rs: cc[r["category"]]=cc.get(r["category"],0)+1
            counts[label][group]={"n":256,"categories":cc}
    summary={
        "study":"V17C.11 P1 early-time radial loss diagnostics",
        "protocol":{"n_early_periods":PERIODS,"steps_per_primary_period":STEPS,
                    "sample_cycles":SAMPLES.tolist(),"workers":WORKERS,
                    "all_starts":"all 256 published starts in each of four groups for each of two candidates",
                    "same_initial_conditions":"published V17C.10 independent replication starts"},
        "source_sha256":source_info,"published_N256_categories":counts,
        "force_decomposition_RHS_checks":force_errors,
        "analyzed_trajectories":len(rows),"wall_seconds":elapsed,
        "early_trajectories_per_second":len(rows)/elapsed,
        "outputs":["p1_early_diagnostics/early_cases.jsonl","p1_early_diagnostics/early_samples.npz"]
    }
    (HERE/"p1_early_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"analyzed":len(rows),"wall_s":elapsed,
        "trajectories_per_second":len(rows)/elapsed,"output":str(HERE/"p1_early_summary.json")},indent=2))


if __name__=="__main__": main()
