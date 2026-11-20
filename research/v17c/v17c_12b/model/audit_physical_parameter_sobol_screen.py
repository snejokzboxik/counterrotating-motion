"""Integrity and dimensional audit for the broad physical Sobol screen."""
from __future__ import annotations

import hashlib
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physical_parameter_sobol_screen"
EPS0 = 8.8541878128e-12
ECHARGE = 1.602176634e-19
ETA = 1.81e-5
G = 9.80665
R = 0.010
RHO_GAS = 1.177
MFP = 65.1e-9
H2 = 0.870316891091623
H4 = 0.7982172624273378


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def trajectory_reader():
    raw = OUT / "trajectories.jsonl"
    archived = OUT / "trajectories.jsonl.gz"
    if raw.exists():
        return raw.open(encoding="utf-8")
    return gzip.open(archived, "rt", encoding="utf-8")


def trajectory_sha256() -> str:
    raw = OUT / "trajectories.jsonl"
    if raw.exists():
        return sha(raw)
    h = hashlib.sha256()
    with gzip.open(OUT / "trajectories.jsonl.gz", "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta=json.loads((OUT/"campaign_meta.json").read_text(encoding="utf-8"))
    codes=json.loads((OUT/"campaign_code_sha256.json").read_text(encoding="utf-8"))
    summary=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    frozen=np.load(OUT/"frozen_starts.npz",allow_pickle=True)
    starts,phases=frozen["initial"],frozen["phases"]
    labels=frozen["ensemble"].astype(str).tolist()
    configs={c["id"]:c for c in meta["configurations"]}
    groups=meta["ensembles"]
    n=meta["starts_per_ensemble"]
    expected={(cid,g,i) for cid in configs for g in groups for i in range(n)}
    rows=[]
    with trajectory_reader() as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    keys=[(str(r["configuration"]),str(r["ensemble"]),int(r["start_index"])) for r in rows]
    row_set=set(keys)
    row_keys_unique=len(row_set)==len(keys)
    matches_starts=True
    full_retained=True
    short_losses=True
    outcome_by_config=defaultdict(Counter)
    retained_by_config=defaultdict(Counter)
    global_outcomes=Counter()
    losses=defaultdict(Counter)
    for row in rows:
        cid,group,si=row["configuration"],row["ensemble"],int(row["start_index"])
        idx=groups.index(group)*n+si
        matches_starts &= labels[idx]==group
        matches_starts &= np.allclose(row["initial"],starts[idx],rtol=0,atol=1e-14)
        matches_starts &= np.allclose(row["phase"],phases[idx],rtol=0,atol=1e-14)
        full_retained &= (not row["retained"]) or int(row["completed_periods"])==meta["periods"]
        short_losses &= row["retained"] or int(row["completed_periods"])<meta["periods"]
        cat=row["provisional_category"]
        outcome_by_config[cid][cat]+=1
        global_outcomes[cat]+=1
        retained_by_config[cid][group]+=int(bool(row["retained"]))
        if not row["retained"]:
            ph=configs[cid]["physical"]
            radial=max(row["r_max_over_R"][0]+ph["a1_over_R"],row["r_max_over_R"][1]+ph["a2_over_R"])>=1.0
            axial=max(row["z_max_over_R"][0]+ph["a1_over_R"],row["z_max_over_R"][1]+ph["a2_over_R"])>=2.0
            losses[cid]["radial"]+=int(radial); losses[cid]["axial"]+=int(axial); losses[cid]["both"]+=int(radial and axial)

    reconstructed={}
    dim_checks=True
    ranges=True
    for cid,cfg in configs.items():
        p=cfg["p"]; ph=cfg["physical"]; a=ph["particle1"]; b=ph["particle2"]
        om=ph["omega1_rad_s"]
        m1_calc=4*math.pi*a["density_kg_m3"]*a["radius_m"]**3/3
        m2_calc=4*math.pi*b["density_kg_m3"]*b["radius_m"]**3/3
        q1,q2=a["charge_C"],b["charge_C"]
        vals={
            "mass_ratio":a["mass_kg"]/b["mass_kg"],
            "charge_ratio":q2/q1,
            "A":q1*ph["U1_peak_electrode_V"]*H2/(a["mass_kg"]*R**2*om**2),
            "B":q1*ph["U2_peak_electrode_V"]*H4/(a["mass_kg"]*R**2*om**2),
            "nu":ph["omega2_rad_s"]/om,
            "beta":q1*ph["V2_ideal_DC_coefficient_V_not_endcap"]/(a["mass_kg"]*R**2*om**2),
            "C":q1*q2/(4*math.pi*EPS0*a["mass_kg"]*R**3*om**2),
            "lambda1":6*math.pi*ETA*a["radius_m"]/(a["mass_kg"]*om),
            "lambda2":6*math.pi*ETA*b["radius_m"]/(b["mass_kg"]*om),
            "mass1_from_sphere":m1_calc,
            "mass2_from_sphere":m2_calc,
        }
        checks={
            "mass1_sphere":np.isclose(m1_calc,a["mass_kg"],rtol=1e-12),
            "mass2_sphere":np.isclose(m2_calc,b["mass_kg"],rtol=1e-12),
            "mass_ratio":np.isclose(vals["mass_ratio"],p[4],rtol=1e-12),
            "charge_ratio":np.isclose(vals["charge_ratio"],cfg["qratio"],rtol=1e-12),
            "charge1_e":np.isclose(q1/ECHARGE,a["charge_e"],rtol=1e-12),
            "charge2_e":np.isclose(q2/ECHARGE,b["charge_e"],rtol=1e-12),
            "A":np.isclose(vals["A"],p[0],rtol=1e-12),
            "B":np.isclose(vals["B"],p[1],rtol=1e-12),
            "nu":np.isclose(vals["nu"],p[2],rtol=1e-12),
            "beta":np.isclose(vals["beta"],p[3],rtol=1e-12),
            "C":np.isclose(vals["C"],p[5],rtol=1e-12),
            "lambda1":np.isclose(vals["lambda1"],p[6],rtol=1e-12),
            "lambda2":np.isclose(vals["lambda2"],p[7],rtol=1e-12),
        }
        ranges &= (5e-6<=a["radius_m"]<=15e-6 and 15e-6<=b["radius_m"]<=50e-6 and
            100<=b["density_kg_m3"]<=140 and 1e5<=a["charge_e"]<=5e5 and 1e5<=b["charge_e"]<=1e6 and
            50<=ph["frequency_f1_Hz"]<=120 and 30<=ph["frequency_f2_Hz"]<=180 and
            ph["U1_plus_U2_peak_bound_V"]<=4400 and abs(ph["center_of_mass_equilibrium_z_over_R_analytic"])<=1.5)
        dim_checks &= all(bool(x) for x in checks.values())
        reconstructed[cid]={**vals,"checks":{k:bool(v) for k,v in checks.items()}}

    expected_summary={}
    for cid in configs:
        expected_summary[cid]={"rows":sum(k[0]==cid for k in keys),
            "outcomes":dict(sorted(outcome_by_config[cid].items())),
            "retained_by_ensemble":{g:int(retained_by_config[cid][g]) for g in groups},
            "retained_total":sum(retained_by_config[cid].values()),
            "cr_total":outcome_by_config[cid]["PROVISIONAL_CR_PLUS_MINUS"]+outcome_by_config[cid]["PROVISIONAL_CR_MINUS_PLUS"],
            "co_total":outcome_by_config[cid]["PROVISIONAL_CO_PLUS_PLUS"]+outcome_by_config[cid]["PROVISIONAL_CO_MINUS_MINUS"]}
    candidates=[cid for cid,x in expected_summary.items() if x["outcomes"].get("PROVISIONAL_CR_PLUS_MINUS",0)>0 and x["outcomes"].get("PROVISIONAL_CR_MINUS_PLUS",0)>0]
    checks={
        "configuration_count":len(configs)==1024,
        "expected_rows":len(rows)==len(expected),
        "unique_complete_keys":row_keys_unique and row_set==expected,
        "frozen_start_phases_match":bool(matches_starts),
        "full_retained_horizons_and_early_losses":bool(full_retained and short_losses),
        "all_dimensional_reconstructions":bool(dim_checks),
        "physical_filter_ranges":bool(ranges),
        "summary_config_results_match_rows":expected_summary==summary["configuration_results"],
        "summary_candidate_ids_match_rows":candidates==summary["configurations_with_both_counterrotation_labels"],
        "summary_total_outcomes_match_rows":dict(sorted(global_outcomes.items()))==summary["outcome_counts"],
        "trajectory_hash_matches_summary":trajectory_sha256()==summary["trajectory_sha256"],
        "runner_hash_matches_saved_copy":sha(ROOT/"physical_parameter_sobol_screen.py")==sha(OUT/"campaign_runner_source.py")==codes["runner_sha256"]==codes["saved_runner_sha256"],
        "integrator_hash_matches_saved_copy":sha(ROOT/"model"/"basin_scan.py")==sha(OUT/"campaign_integrator_source.py")==codes["integrator_sha256"]==codes["saved_integrator_sha256"],
        "field_hash_matches":sha(ROOT/"model"/"field_model.json")==codes["field_model_sha256"],
        "starts_hash_matches":sha(OUT/"frozen_starts.npz")==codes["frozen_starts_sha256"]==meta["frozen_starts_sha256"],
        "total_configs_trajectories_match_meta":len(rows)==meta["total_trajectories"]==len(configs)*meta["starts_per_ensemble"]*len(groups),
    }
    loss_counts={cid:{k:int(v[k]) for k in ("radial","axial","both")} for cid,v in losses.items()}
    report={"audit":"physical_parameter_sobol_screen","integrity_pass":all(checks.values()),
        "checks":checks,"configurations":len(configs),"rows":len(rows),"unique_keys":len(row_set),
        "global_outcomes":dict(sorted(global_outcomes.items())),"loss_mechanisms_by_configuration":loss_counts,
        "configurations_with_both_counterrotation_labels":candidates,
        "reconstructed_parameters":reconstructed,
        "interpretation":"Physical-parameter discovery screen with 3000-period preliminary labels; not a strict probability estimate or proof of absent attractors.",
        "trajectory_sha256":trajectory_sha256(),"audit_script_sha256":sha(Path(__file__))}
    (OUT/"independent_audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="reconstructed_parameters" and k!="loss_mechanisms_by_configuration"},indent=2,ensure_ascii=False))
    if not report["integrity_pass"]: raise SystemExit(1)

if __name__=="__main__":
    main()
