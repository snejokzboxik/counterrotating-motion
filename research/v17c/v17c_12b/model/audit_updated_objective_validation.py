"""Independent integrity, labels, and dimensional reconstruction audit."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "updated_objective_validation"
EPS0 = 8.8541878128e-12
ECHARGE = 1.602176634e-19
ETA = 1.81e-5
R = 0.010
H2 = 0.870316891091623
H4 = 0.7982172624273378
GROUPS = ("rest", "weak", "wide", "edge")
N = 128


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def label_strict(raw: str, turns: np.ndarray, windows: np.ndarray, retained: bool, axis: bool) -> str:
    if raw == "finite_particle_contact": return "PARTICLE_CONTACT"
    if raw == "finite_particle_wall_contact": return "WALL_OR_AXIAL_ESCAPE"
    if raw == "lost_retention": return "ESCAPE"
    if raw == "point_singularity_unresolved": return "UNRESOLVED"
    if axis: return "AXIS_AMBIGUOUS"
    signs=np.zeros((3,2),dtype=np.int8)
    for w in range(3):
        for i in range(2):
            if abs(windows[w,i])>=0.05: signs[w,i]=1 if windows[w,i]>0 else -1
    if retained and np.all(signs!=0) and np.all(np.abs(turns)>=10.0):
        if np.all(signs[:,0]*signs[:,1]<0): return "STRICT_CR_PLUS_MINUS" if turns[0]>0 else "STRICT_CR_MINUS_PLUS"
        if np.all(signs[:,0]*signs[:,1]>0): return "STRICT_CO_PLUS_PLUS" if turns[0]>0 else "STRICT_CO_MINUS_MINUS"
    if retained and np.all(np.abs(turns)<0.25): return "NONROTATING"
    return "UNRESOLVED_LATE_WINDOW"


def main() -> None:
    meta=json.loads((OUT/"campaign_meta.json").read_text(encoding="utf-8"))
    code=json.loads((OUT/"campaign_code_sha256.json").read_text(encoding="utf-8"))
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    frozen=np.load(OUT/"frozen_starts.npz",allow_pickle=True)
    starts,phases=frozen["initial"],frozen["phases"]
    labels=frozen["ensemble"].astype(str).tolist()
    configs={c["id"]:c for c in meta["candidates"]}
    rows=[]
    with (OUT/"trajectories.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    keys=[(str(x["configuration"]),str(x["ensemble"]),int(x["start_index"])) for x in rows]
    keyset=set(keys)
    expected={(cid,g,i) for cid in configs for g in GROUPS for i in range(N)}
    starts_match=True
    horizon_ok=True
    strict_labels_ok=True
    counts=defaultdict(lambda:defaultdict(Counter))
    provisional_counts=defaultdict(lambda:defaultdict(Counter))
    all_outcomes=Counter()
    for row in rows:
        cid,g,si=row["configuration"],row["ensemble"],int(row["start_index"])
        idx=GROUPS.index(g)*N+si
        starts_match &= labels[idx]==g
        starts_match &= np.allclose(row["initial"],starts[idx],rtol=0,atol=1e-14)
        starts_match &= np.allclose(row["phase"],phases[idx],rtol=0,atol=1e-14)
        horizon_ok &= (not row["retained"]) or row["completed_periods"]==meta["periods"]
        horizon_ok &= row["retained"] or row["completed_periods"]<meta["periods"]
        axis=bool(min(row["r_min_over_R"])<0.02)
        expected_label=label_strict(row["raw_category"],np.asarray(row["turns"]),np.asarray(row["window_turns"]),row["retained"],axis)
        strict_labels_ok &= expected_label==row["strict_late_category"]
        counts[cid][g][row["strict_late_category"]]+=1
        provisional_counts[cid][g][row["provisional_category"]]+=1
        all_outcomes[row["strict_late_category"]]+=1

    dim_checks=True
    dim_reconstruction={}
    for cid,cfg in configs.items():
        ph=cfg["physical"]
        p=np.asarray(cfg["p"],dtype=float)
        p1,p2=ph["particle1"],ph["particle2"]
        m1=4*math.pi*p1["density_kg_m3"]*p1["radius_m"]**3/3
        m2=4*math.pi*p2["density_kg_m3"]*p2["radius_m"]**3/3
        q1,q2=p1["charge_C"],p2["charge_C"]
        om1,om2=ph["omega1_rad_s"],ph["omega2_rad_s"]
        vals={"mass_ratio":m1/m2,"charge_ratio":q2/q1,
              "A":q1*ph["U1_peak_electrode_V"]*H2/(m1*R**2*om1**2),
              "B":q1*ph["U2_peak_electrode_V"]*H4/(m1*R**2*om1**2),
              "nu":om2/om1,"beta":q1*ph["V2_ideal_DC_coefficient_V_not_endcap"]/(m1*R**2*om1**2),
              "C":q1*q2/(4*math.pi*EPS0*m1*R**3*om1**2),
              "lambda1":6*math.pi*ETA*p1["radius_m"]/(m1*om1),
              "lambda2":6*math.pi*ETA*p2["radius_m"]/(m2*om1)}
        expected_vals={"mass_ratio":p[4],"charge_ratio":cfg["qratio"],"A":p[0],"B":p[1],"nu":p[2],
                       "beta":p[3],"C":p[5],"lambda1":p[6],"lambda2":p[7]}
        checks={k:bool(np.isclose(vals[k],expected_vals[k],rtol=2e-10,atol=1e-13)) for k in vals}
        checks["charge1_e"]=bool(np.isclose(q1/ECHARGE,p1["charge_e"],rtol=1e-12))
        checks["charge2_e"]=bool(np.isclose(q2/ECHARGE,p2["charge_e"],rtol=1e-12))
        dim_checks &= all(checks.values())
        dim_reconstruction[cid]={"reconstructed":vals,"checks":checks}

    reconstructed_summary={}
    summary_match=True
    for cid in configs:
        per={}
        for g in GROUPS:
            c=counts[cid][g]
            provisional=provisional_counts[cid][g]
            n=sum(c.values())
            per[g]={"n":n,"provisional":dict(sorted(provisional.items())),
                "strict_late":dict(sorted(c.items())),
                "strict_CR_any_direction":c["STRICT_CR_PLUS_MINUS"]+c["STRICT_CR_MINUS_PLUS"],
                "strict_CR_plus_minus":c["STRICT_CR_PLUS_MINUS"],"strict_CR_minus_plus":c["STRICT_CR_MINUS_PLUS"],
                "co_rotation":c["STRICT_CO_PLUS_PLUS"]+c["STRICT_CO_MINUS_MINUS"],
                "escape":c["ESCAPE"]+c["WALL_OR_AXIAL_ESCAPE"]}
        cr=sum(v["strict_CR_any_direction"] for v in per.values())
        total={"rows":sum(v["n"] for v in per.values()),"strict_CR_any_direction_of_512":cr,
            "strict_CR_fraction_of_512":cr/512,"co_rotation_total":sum(v["co_rotation"] for v in per.values()),
            "escape_total":sum(v["escape"] for v in per.values()),"per_ensemble":per}
        reconstructed_summary[cid]=total
        summary_match &= total==summary["configuration_results"][cid]

    checks={
        "five_candidate_count":len(configs)==5,
        "complete_expected_rows":len(rows)==len(expected) and keyset==expected,
        "unique_configuration_ensemble_start_keys":len(keys)==len(keyset),
        "frozen_starts_and_phases_match":bool(starts_match),
        "retained_full_horizons_and_losses_early":bool(horizon_ok),
        "strict_late_labels_reproduce":bool(strict_labels_ok),
        "all_dimensional_reconstructions_match":bool(dim_checks),
        "summary_matches_raw_rows":bool(summary_match),
        "summary_row_count_matches_meta":len(rows)==meta["total_trajectories"],
        "trajectory_sha256_matches_summary":sha(OUT/"trajectories.jsonl")==summary["trajectory_sha256"],
        "runner_source_copy_matches_current":sha(ROOT/"updated_objective_validation.py")==sha(OUT/"campaign_runner_source.py"),
        "runner_hash_matches_code_metadata":sha(ROOT/"updated_objective_validation.py")==code["runner_sha256"]==code["saved_runner_sha256"],
        "integrator_hash_matches_saved_copy":sha(ROOT/"model"/"basin_scan.py")==sha(OUT/"campaign_integrator_source.py")==code["integrator_sha256"]==code["saved_integrator_sha256"],
        "field_hash_matches":sha(ROOT/"model"/"field_model.json")==code["field_model_sha256"],
        "frozen_starts_hash_matches":sha(OUT/"frozen_starts.npz")==code["frozen_starts_sha256"]==meta["frozen_starts_sha256"],
    }
    report={"audit":"updated_objective_validation","integrity_pass":all(checks.values()),"checks":checks,
        "configurations":len(configs),"rows":len(rows),"unique_keys":len(keyset),
        "strict_late_outcome_counts":dict(sorted(all_outcomes.items())),
        "configuration_results_reconstructed":reconstructed_summary,
        "dimensional_reconstruction":dim_reconstruction,
        "trajectory_sha256":sha(OUT/"trajectories.jsonl"),
        "interpretation":"Fresh-start 15,000-period validation under any-direction CR objective; strict late labels require >=10 net turns per particle and consistent signs across three late windows; not IID probabilities or proof of an attractor.",
        "audit_script_sha256":sha(Path(__file__))}
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="dimensional_reconstruction" and k!="configuration_results_reconstructed"},indent=2,ensure_ascii=True))
    if not report["integrity_pass"]: raise SystemExit(1)


if __name__=="__main__":
    main()
