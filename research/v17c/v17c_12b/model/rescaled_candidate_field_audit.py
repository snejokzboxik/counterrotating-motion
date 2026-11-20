"""MFS full-RF-phase-torus screening for the three rescaled physical candidates."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rescaled_legacy_candidate_validation"
MFS_PATH = ROOT / "model" / "mfs_breakdown.py"
spec = importlib.util.spec_from_file_location("v17c_mfs_breakdown", MFS_PATH)
mfs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mfs)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    meta = json.loads((OUT / "campaign_meta.json").read_text(encoding="utf-8"))
    results = {}
    for candidate in meta["candidates"]:
        cid, p = candidate["id"], candidate["physical"]
        voltage = {
            "u1_electrode_peak_V": p["U1_peak_electrode_V"],
            "u2_electrode_peak_V": p["U2_peak_electrode_V"],
            "v2_dc_coefficient_V": p["V2_ideal_DC_coefficient_V_not_endcap"],
            "D_m": p["trap_radius_m"] / 0.95,
            "R_m": p["trap_radius_m"],
        }
        orders = (16, 24, 32) if "refine12" in cid else (24,)
        results[cid] = {
            "voltage": voltage,
            "mfs_convergence": [mfs.sample_fields(order, 512, voltage) for order in orders],
        }
    output = {
        "study": "V17C.12B rescaled candidates; MFS surface field over full two-phase RF torus",
        "geometry": mfs.GEOMETRY,
        "source_meta_sha256": sha(OUT / "campaign_meta.json"),
        "mfs_source_sha256": sha(MFS_PATH),
        "results": results,
        "limitations": [
            "Infinite 2D rods and grounded shield; no rod ends, feedthroughs, or physical DC segmentation.",
            "RF basis fields are superposed over the independent two-phase torus.",
            "The ideal analytic DC quadrupole is added as a near-center estimate, not solved as a conductor-boundary problem.",
            "No independent FEM, corona, or gas-discharge model is included; this is not a breakdown clearance.",
        ],
    }
    path = OUT / "mfs_field_audit.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    compact = {cid: [{k: v for k, v in row.items() if k in ("p_collocation_per_rod", "rf_only_max_surface_field_V_m_all", "max_surface_field_ideal_DC_analytically_added_V_m_not_a_conductor_BVP", "support_vs_direct_torus_max_rel_diff")} for row in r["mfs_convergence"]] for cid, r in results.items()}
    print(json.dumps({"saved": str(path), "result": compact}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
