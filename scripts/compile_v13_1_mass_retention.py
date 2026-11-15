"""Combine ideal and full-FEM V13.1 retention summaries with source labels."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
items = []

for path, field, run in [
    (BASE / "ideal_pilot/retention_summary.csv", "ideal leading harmonics", "ideal pilot"),
    (BASE / "fem_h50_p4_mu100_105/retention_summary.csv", "full h50 P4 FEM", "FEM mu 1.00, 1.05"),
    (BASE / "fem_h50_p4_mu110/retention_summary.csv", "full h50 P4 FEM", "FEM mu 1.10"),
]:
    for r in csv.DictReader(path.open(encoding="utf-8", newline="")):
        items.append({
            "field": field, "run": run,
            "mu": r["mu"], "phase_deg": r["phase_deg"], "horizon_rf": r["horizon_rf"],
            "model": r["model"], "eta": r["eta"],
            "n_total": r.get("n_total", r.get("n_resolved", "")),
            "n_resolved": r.get("n_resolved", ""), "n_unresolved": r.get("n_unresolved", ""),
            "n_step_sensitive": r.get("n_step_sensitive", "0"),
            "n_retained": r["n_retained"], "retention": r["retention"],
            "retention_basis": r.get("retention_basis", "resolved cases only"),
            "retention_bound_low_all_n": r.get("retention_bound_low_all_n", r.get("retention", "")),
            "retention_bound_high_all_n": r.get("retention_bound_high_all_n", r.get("retention", "")),
            "ci95_low": r.get("wilson95_low", r.get("retention_wilson95_low", "")),
            "ci95_high": r.get("wilson95_high", r.get("retention_wilson95_high", "")),
        })

out = BASE / "mass_retention_table.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(items[0]))
    writer.writeheader(); writer.writerows(items)
print(f"wrote {len(items)} rows to {out}")
