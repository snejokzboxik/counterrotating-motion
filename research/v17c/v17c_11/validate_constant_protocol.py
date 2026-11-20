"""P0: compare the constant-envelope solver with published V17C.10 rows."""
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
STEPS, PERIODS, TRANSIENT = 64, 15_000, 200
R1, R2 = 0.0002, 0.00189


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_group_rows(label: str, group: str):
    stem = f"{label}_{group}_matched_strict_none_a0.0002-0.00189_replicate20261410_256_15000p_64s"
    path = V10 / "focused_edge_replication" / f"{stem}.jsonl"
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start_path = HERE.parents[2] / Path(meta["starts"])
    if sha256(start_path) != meta["frozen_start_sha256"]:
        raise RuntimeError(f"frozen-start hash mismatch: {start_path}")
    with np.load(start_path, allow_pickle=False) as starts:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != 256 or len({int(r["index"]) for r in rows}) != 256:
            raise RuntimeError(f"incomplete or duplicate rows: {path}")
        for row in rows:
            i = int(row["index"])
            if not np.array_equal(np.asarray(row["initial"]), starts["initial"][i]):
                raise RuntimeError(f"row initial differs from frozen starts: {path}:{i}")
            if not np.array_equal(np.asarray(row["phase"]), starts["phases"][i]):
                raise RuntimeError(f"row phase differs from frozen starts: {path}:{i}")
    return {int(row["index"]): row for row in rows}, sha256(path), sha256(start_path)


def compare_one(job):
    label, group, row = job
    p = np.asarray(row["_p"], dtype=np.float64)
    phase = np.asarray(row["phase"], dtype=np.float64)
    initial = np.asarray(row["initial"], dtype=np.float64)
    fixed = basin_scan.integrate_one(initial, phase, p, FAMILY, K, COEFF,
        PERIODS, STEPS, TRANSIENT, qratio=row["_qratio"],
        oseen_Re_per_speed=0.0, hydro_mode=0, a1_over_R=R1,
        a2_over_R=R2, oseen_Re2_per_speed=0.0)
    new = timed.integrate(initial, phase, p, timed.constant_protocol(), FAMILY, K,
        COEFF, PERIODS, STEPS, TRANSIENT, row["_qratio"], R1, R2)
    names = ("final_state", "turns", "window_turns", "r_min", "r_max", "abs_z_max",
             "max_abs_inner_step_dtheta_rad", "minimum_separation", "completed_periods",
             "retained", "escaped", "singular_unresolved", "category", "frequency_Hz")
    vals = dict(zip(names, fixed))
    array_keys = ("final_state", "turns", "window_turns", "r_min", "r_max",
                  "abs_z_max", "max_abs_inner_step_dtheta_rad", "frequency_Hz")
    exact = vals["category"] == new["category"]
    max_abs = 0.0
    for key in array_keys:
        delta = np.max(np.abs(np.asarray(vals[key]) - np.asarray(new[key])))
        max_abs = max(max_abs, float(delta))
        exact &= bool(np.array_equal(np.asarray(vals[key]), np.asarray(new[key])))
    for key in ("completed_periods", "retained", "escaped", "singular_unresolved"):
        exact &= vals[key] == new[key]
    max_abs = max(max_abs, abs(float(vals["minimum_separation"])-float(new["minimum_separation"])))
    exact &= vals["minimum_separation"] == new["minimum_separation"]
    return {
        "candidate": label, "group": group, "index": int(row["index"]),
        "published_category": row["category"], "baseline_replay_category": vals["category"],
        "timed_constant_category": new["category"], "exact_full_result_match": bool(exact),
        "max_abs_numeric_difference": max_abs,
        "completed_periods": int(new["completed_periods"]),
        "turns": np.asarray(new["turns"]).tolist(),
        "r_min_R": np.asarray(new["r_min"]).tolist(),
        "wall_seconds": None,
    }


def main():
    params_doc = json.loads((HERE / "local15_parameters.json").read_text(encoding="utf-8"))
    local_p = np.asarray(params_doc["candidate"]["p"], dtype=np.float64)
    local_q = float(params_doc["candidate"]["q2_over_q1"])
    p0, family, modes, coeff = basin_scan.load_model()
    global FAMILY, K, COEFF
    FAMILY, K, COEFF = family, modes, coeff
    candidates = {
        "local15": (local_p, local_q),
        "candidate52": (np.array([0.03956095746691348,0.15032976498886072,2.8996277751401065,
            0.015917322789905167,0.11307613545330587,9.281679945656495e-06,
            0.1868668374144136,0.10359181855828002]), 3.5274259874597194),
    }
    jobs=[]; data_sources={}
    for group in GROUPS:
        local, local_json_sha, starts_sha = read_group_rows("local15", group)
        other, other_json_sha, other_starts_sha = read_group_rows("candidate52", group)
        if starts_sha != other_starts_sha:
            raise RuntimeError(f"matched starts differ between candidates in {group}")
        data_sources[group] = {"local15_rows_sha256":local_json_sha,
                               "candidate52_rows_sha256":other_json_sha,
                               "frozen_starts_sha256":starts_sha}
        picked=[]
        for category in ("CR_screen", "finite_particle_wall_contact"):
            choices=[i for i,r in local.items() if r["category"]==category]
            if not choices:
                raise RuntimeError(f"no {category} sample in local15 {group}")
            picked.append(min(choices))
        for idx in picked:
            for label, source in (("local15",local), ("candidate52",other)):
                row=dict(source[idx]); row["_p"]=candidates[label][0].tolist(); row["_qratio"]=candidates[label][1]
                jobs.append((label,group,row))
    # Compile both implementations before entering the parallel section.
    warm = jobs[0]
    candidate, group, row = warm
    p = np.asarray(row["_p"]); initial=np.asarray(row["initial"]); phase=np.asarray(row["phase"])
    basin_scan.integrate_one(initial,phase,p,FAMILY,K,COEFF,2,STEPS,0,
        qratio=row["_qratio"],a1_over_R=R1,a2_over_R=R2,oseen_Re2_per_speed=0.0)
    timed.integrate(initial,phase,p,timed.constant_protocol(),FAMILY,K,COEFF,2,STEPS,0,
        row["_qratio"],R1,R2)
    t0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(compare_one,jobs))
    elapsed=time.perf_counter()-t0
    if not all(r["exact_full_result_match"] for r in results):
        raise RuntimeError("constant-envelope equivalence failed; do not launch loading search")
    output={
        "study":"V17C.11 P0 constant-envelope equivalence",
        "source_commit":"57e197bb744a3af890d466f418a76a95cf7758b4",
        "protocol":{"n_samples":len(results),"sample_rule":"first local15 CR and first local15 finite wall-contact index in each group; same paired index on candidate52",
                    "steps_per_primary_period":STEPS,"periods":PERIODS,"transient_periods":TRANSIENT,
                    "finite_radii_over_R":[R1,R2],"workers":4,"same_rhs":"v17c_9/baseline/basin_scan.py"},
        "sources":data_sources,"all_exact_matches":True,"total_wall_seconds":elapsed,
        "trajectories_per_second":len(results)/elapsed,"results":results,
    }
    out=HERE/"p0_constant_equivalence.json"
    out.write_text(json.dumps(output,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"all_exact_matches":True,"N":len(results),"wall_s":elapsed,
                      "trajectories_per_second":len(results)/elapsed,"output":str(out)},indent=2))


if __name__=="__main__": main()
