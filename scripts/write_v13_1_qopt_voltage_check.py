"""Write per-electrode peak voltages for the Qopt coarse Floquet control."""
from pathlib import Path
import csv
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/octupole_comparison_home/mass_filter_v13/v13_1"
U2, VDC = 190.0, 30.0
rows = []
for electrode in range(12):
    theta = 2 * math.pi * electrode / 12
    c4, s4, q2 = math.cos(4 * theta), math.sin(4 * theta), math.cos(2 * theta)
    rf_cos, rf_sin, dc = U2 * q2, 0.0, VDC * q2
    values = (rf_cos + dc, -rf_cos + dc)
    rows.append({"scheme": "Qopt", "eta": 0.0, "U4_V": 0.0, "electrode": electrode,
                 "theta_deg": electrode * 30, "C4": c4, "S4": s4, "Q2": q2,
                 "rf_cos_V": rf_cos, "rf_sin_V": rf_sin, "dc_V": dc,
                 "peak_abs_V": max(abs(v) for v in values), "minimum_V": min(values),
                 "within_250V": max(abs(v) for v in values) <= 250.0})

with (OUT / "qopt_electrode_voltage_peaks.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
peak = max(r["peak_abs_V"] for r in rows)
summary_path = OUT / "electrode_voltage_summary.csv"
existing = list(csv.DictReader(summary_path.open(encoding="utf-8", newline="")))
existing = [r for r in existing if r["scheme"] != "Qopt"]
existing.append({"scheme": "Qopt", "eta": 0.0, "U4_V": 0.0, "max_abs_electrode_V": peak,
                 "max_electrode": next(r["electrode"] for r in rows if r["peak_abs_V"] == peak),
                 "within_250V": peak <= 250.0})
with summary_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(existing[0])); w.writeheader(); w.writerows(existing)
print(f"Qopt: max |V|={peak:.4f} V; all 12 electrodes checked")
