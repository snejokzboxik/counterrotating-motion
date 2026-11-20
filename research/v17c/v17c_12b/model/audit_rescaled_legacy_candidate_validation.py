"""Independent integrity and scaling audit for the f1-rescaled candidates."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"rescaled_legacy_candidate_validation"
EPS0=8.8541878128e-12
ECHARGE=1.602176634e-19
ETA=1.81e-5
R=0.010
H2=0.870316891091623
H4=0.7982172624273378
GROUPS=("rest","weak","wide","edge")
N=128


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def main()->None:
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
            if line.strip():rows.append(json.loads(line))
    keys=[(r["configuration"],r["ensemble"],int(r["start_index"])) for r in rows]
    keyset=set(keys);expected={(c,g,i) for c in configs for g in GROUPS for i in range(N)}
    starts_match=True;horizon=True;outcomes=defaultdict(lambda:defaultdict(Counter));summary_match=True
    for r in rows:
        cid,g,si=r["configuration"],r["ensemble"],int(r["start_index"])
        idx=GROUPS.index(g)*N+si
        starts_match &= labels[idx]==g and np.allclose(r["initial"],starts[idx],rtol=0,atol=1e-14)
        starts_match &= np.allclose(r["phase"],phases[idx],rtol=0,atol=1e-14)
        horizon &= (not r["retained"]) or r["completed_periods"]==meta["periods"]
        horizon &= r["retained"] or r["completed_periods"]<meta["periods"]
        outcomes[cid][g][r["strict_late_category"]]+=1
    dimensional=True;recon={}
    for cid,c in configs.items():
        p=np.asarray(c["p"],float);ph=c["physical"];p1,p2=ph["particle1"],ph["particle2"]
        m1=4*math.pi*p1["density_kg_m3"]*p1["radius_m"]**3/3
        m2=4*math.pi*p2["density_kg_m3"]*p2["radius_m"]**3/3
        q1,q2=p1["charge_C"],p2["charge_C"];om1,om2=ph["omega1_rad_s"],ph["omega2_rad_s"]
        values={"mass_ratio":m1/m2,"charge_ratio":q2/q1,
            "A":q1*ph["U1_peak_electrode_V"]*H2/(m1*R**2*om1**2),
            "B":q1*ph["U2_peak_electrode_V"]*H4/(m1*R**2*om1**2),
            "nu":om2/om1,"beta":q1*ph["V2_ideal_DC_coefficient_V_not_endcap"]/(m1*R**2*om1**2),
            "C":q1*q2/(4*math.pi*EPS0*m1*R**3*om1**2),
            "lambda1":6*math.pi*ETA*p1["radius_m"]/(m1*om1),
            "lambda2":6*math.pi*ETA*p2["radius_m"]/(m2*om1)}
        expected_values={"mass_ratio":p[4],"charge_ratio":c["qratio"],"A":p[0],"B":p[1],
            "nu":p[2],"beta":p[3],"C":p[5],"lambda1":p[6],"lambda2":p[7]}
        checks={k:bool(np.isclose(values[k],expected_values[k],rtol=3e-10,atol=1e-13)) for k in values}
        checks["charge1_e"]=bool(np.isclose(q1/ECHARGE,p1["charge_e"],rtol=1e-12))
        checks["charge2_e"]=bool(np.isclose(q2/ECHARGE,p2["charge_e"],rtol=1e-12))
        dimensional &= all(checks.values());recon[cid]={"reconstructed":values,"checks":checks}
    rebuilt={}
    for cid in configs:
        per={}
        for g in GROUPS:
            c=outcomes[cid][g]
            per[g]={"n":sum(c.values()),"strict_late":dict(sorted(c.items())),
                "CR_any_direction":c["STRICT_CR_PLUS_MINUS"]+c["STRICT_CR_MINUS_PLUS"],
                "CR_plus_minus":c["STRICT_CR_PLUS_MINUS"],"CR_minus_plus":c["STRICT_CR_MINUS_PLUS"],
                "co_rotation":c["STRICT_CO_PLUS_PLUS"]+c["STRICT_CO_MINUS_MINUS"],
                "escape":c["ESCAPE"]+c["WALL_OR_AXIAL_ESCAPE"]}
        cr=sum(v["CR_any_direction"] for v in per.values())
        val={"rows":sum(v["n"] for v in per.values()),"CR_any_direction_of_512":cr,
            "CR_fraction_of_512":cr/512,"co_rotation_total":sum(v["co_rotation"] for v in per.values()),
            "escape_total":sum(v["escape"] for v in per.values()),"per_ensemble":per}
        rebuilt[cid]=val;summary_match &= val==summary["configuration_results"][cid]
    checks={
        "three_candidate_count":len(configs)==3,
        "expected_complete_rows":len(rows)==len(expected) and keyset==expected,
        "unique_keys":len(keyset)==len(keys),
        "frozen_start_phase_match":bool(starts_match),
        "retained_full_horizons_losses_early":bool(horizon),
        "all_dimensional_reconstructions":bool(dimensional),
        "summary_matches_rows":bool(summary_match),
        "row_count_matches_meta":len(rows)==meta["total_trajectories"],
        "trajectory_sha_matches_summary":sha(OUT/"trajectories.jsonl")==summary["trajectory_sha256"],
        "runner_copy_matches_source":sha(ROOT/"rescaled_legacy_candidate_validation.py")==sha(OUT/"campaign_runner_source.py")==code["runner_sha256"]==code["saved_runner_sha256"],
        "integrator_copy_and_hash_match":sha(ROOT/"model"/"basin_scan.py")==sha(OUT/"campaign_integrator_source.py")==code["integrator_sha256"]==code["saved_integrator_sha256"],
        "field_hash_matches":sha(ROOT/"model"/"field_model.json")==code["field_model_sha256"],
        "frozen_starts_hash_matches":sha(OUT/"frozen_starts.npz")==code["frozen_starts_sha256"]==meta["frozen_starts_sha256"],
    }
    report={"audit":"rescaled_legacy_candidate_validation","integrity_pass":all(checks.values()),
        "checks":checks,"configurations":len(configs),"rows":len(rows),"unique_keys":len(keyset),
        "configuration_results_reconstructed":rebuilt,"dimensional_reconstruction":recon,
        "trajectory_sha256":sha(OUT/"trajectories.jsonl"),
        "interpretation":"Independent fresh-start validation of analytically rescaled candidate physical maps; pointwise discovery frequencies, not final physical feasibility or 100-orbit proof.",
        "audit_script_sha256":sha(Path(__file__))}
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("configuration_results_reconstructed","dimensional_reconstruction")},indent=2,ensure_ascii=True))
    if not report["integrity_pass"]:raise SystemExit(1)


if __name__=="__main__":main()
