"""Run a two-phase-torus MFS field audit for the frozen Local15 voltages."""
from __future__ import annotations

import json
from pathlib import Path

from physical_screen import MFS, PAIR
from mfs_breakdown import sample_fields


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    screen = json.loads((root/"results"/"dimensional_screen.json").read_text(
        encoding="utf-8"))
    fields = screen["field_and_electrodes"]
    voltage = {
        "u1_electrode_peak_V": fields["U1_electrode_peak_V"],
        "u2_electrode_peak_V": fields["U2_electrode_peak_V"],
        "v2_dc_coefficient_V": fields[
            "V2_DC_ideal_quadrupole_coefficient_V_not_electrode_voltage"],
        "D_m": fields["D_m"],
        "R_m": PAIR["radial_scale_R_m"],
    }
    convergence = [sample_fields(p, 256, voltage) for p in (16, 24, 32)]
    result = {
        "study": "24-rod 2D infinite-cylinder MFS surface field; full independent RF phase torus",
        "geometry": {
            "electrodes": MFS["electrodes"],
            "R_over_D": MFS["R_over_D"],
            "rod_radius_over_D": MFS["rod_radius_over_D"],
            "shield_radius_over_D": MFS["shield_radius_over_D"],
        },
        "input_peak_voltages_V": {
            "RF1": voltage["u1_electrode_peak_V"],
            "RF2": voltage["u2_electrode_peak_V"],
            "DC_quadrupole_coefficient_not_electrode_voltage":
                voltage["v2_dc_coefficient_V"],
        },
        "MFS_convergence": convergence,
        "limitations": [
            "This is the infinite-rod 2D cross-section model; end effects, endcaps, feedthroughs, and real DC segmentation are absent.",
            "The RF conductor boundary problem is solved by MFS, but no independent FEM solve is included in V17C.16 yet.",
            "The reported full-phase maxima are exact within phase support-function quadrature and the sampled surface grid; surface mesh convergence is assessed across MFS order and 256 points per rod.",
            "The analytically added ideal DC field is not a conductor-boundary solution and is reported separately from the RF-only boundary field.",
            "No threshold for corona or complete breakdown is inferred from the field maximum alone.",
        ],
    }
    out = root/"results"/"mfs_surface_audit.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(out)
    print(json.dumps({"max_surface_fields_V_m": [
        {"order": x["p_collocation_per_rod"],
         "RF_rod": x["rf_only_rod_max_surface_field_V_m"],
         "RF_shield": x["rf_only_shield_max_surface_field_V_m"],
         "RF_with_ideal_DC_rod": x["rod_max_surface_field_ideal_DC_added_V_m"],
         "RF_with_ideal_DC_shield": x["shield_max_surface_field_ideal_DC_added_V_m"],
         "boundary_residual": x["max_boundary_potential_residual_V"],
         "direct_vs_support_rel": x["support_vs_direct_torus_max_rel_diff"]}
        for x in convergence]}, indent=2))


if __name__ == "__main__":
    main()
