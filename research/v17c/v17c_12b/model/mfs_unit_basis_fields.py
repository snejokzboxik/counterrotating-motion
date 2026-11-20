"""Unit-electrode-amplitude MFS surface-field bounds for the V17C geometry."""
from __future__ import annotations

import json
from pathlib import Path

from mfs_breakdown import sample_fields

HERE = Path(__file__).resolve().parent
OUT = HERE.parent/"results"/"mfs_unit_basis_fields.json"


def main() -> None:
    cases = {
        "RF1_k_plus2_1V": {"u1_electrode_peak_V": 1.0, "u2_electrode_peak_V": 0.0},
        "RF2_k_minus4_1V": {"u1_electrode_peak_V": 0.0, "u2_electrode_peak_V": 1.0},
    }
    rows = {}
    for name, voltage in cases.items():
        values = {**voltage, "v2_dc_coefficient_V": 0.0, "D_m": 1.0, "R_m": 0.95}
        result = sample_fields(40, 128, values)
        rows[name] = {"rod_max_V_m_per_electrode_V": result["rf_only_rod_max_surface_field_V_m"],
            "shield_max_V_m_per_electrode_V": result["rf_only_shield_max_surface_field_V_m"],
            "boundary_residual": result["max_boundary_potential_residual_V"],
            "surface_samples_per_rod": 128}
    out = {"study": "MFS single-basis RF surface field per electrode volt at D=1 m",
        "geometry": "24 cylindrical rods, a/D=0.05, center radius D, grounded shield 1.10D",
        "basis_solutions": rows,
        "conservative_combination_bound": "Emax <= Emax_RF1_unit*U1 + Emax_RF2_unit*U2; bound is for RF-only fields and can be loose.",
        "limitation": "Surface-angle sampling is 128 points per rod; use the high-resolution phase/angle MFS and FEM checks for finalists."}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
