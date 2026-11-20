"""Surface-angle refinement at converged MFS order for RF-only Emax."""
from __future__ import annotations

import json
from pathlib import Path

from mfs_breakdown import sample_fields

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"


def main() -> None:
    seed = json.loads((RESULTS/"physical_seed.json").read_text(encoding="utf-8"))
    tr, drive = seed["trap"], seed["electrode_drive"]
    voltage = {
        "u1_electrode_peak_V": drive["U1_peak_V_electrode_not_RMS_or_peak_to_peak"],
        "u2_electrode_peak_V": drive["U2_peak_V_electrode_not_RMS_or_peak_to_peak"],
        "v2_dc_coefficient_V": drive["ideal_DC_quadrupole_coefficient_V_not_an_endcap_voltage"],
        "D_m": tr["electrode_center_radius_D_m"], "R_m": tr["free_aperture_radius_R_m"],
    }
    rows = [sample_fields(40, n, voltage) for n in (128, 256, 512, 1024)]
    base = rows[-1]["rf_only_max_surface_field_V_m_all"]
    for row in rows:
        row["relative_to_n1024"] = row["rf_only_max_surface_field_V_m_all"]/base-1.0
    out = {"study": "MFS surface sampling refinement", "mfs_order": 40,
        "phase_torus_direction_samples": 16384, "seed": seed["label"],
        "rows": rows,
        "note": "Only RF conductor-surface maxima are used for the MFS/FEM comparison; ideal DC quadrupole values at electrode surfaces are not a physical boundary solution."}
    (RESULTS/"mfs_surface_refinement.json").write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
