"""Independent audit and horizon-convergence summary for the Local15 exact-RF run."""
import csv, hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PANEL=ROOT/"results/frozen_local15_single_particle_panel.csv"
RUN=ROOT/"results/local15_exact_rf/rk4_96_15000.json"
OUT=ROOT/"results/local15_exact_rf/independent_audit.json"
d=json.loads(RUN.read_text(encoding="utf-8"))
panel=list(csv.DictReader(PANEL.open(newline="",encoding="utf-8")))
panel_hash=hashlib.sha256(PANEL.read_bytes()).hexdigest()
leading={
 "species1":{"rho":0.5503691530067869,"omega":0.001440054163088106},
 "species2":{"rho":0.7663818561742767,"omega":-0.0010579232360122394},
}
summary={"status":"completed exact ideal two-mode RF audit; leading averaging comparison",
 "source_parameters":d["parameters"],"panel_sha256":panel_hash,
 "panel_size_per_species":len(panel),"panel_seed":15152026,
 "panel_is_iid_probability_sample":False,
 "protocol":d["protocol"],"horizons":{},
 "leading_roots":leading,"species":{},"paired_start_outcomes":{},
 "trajectory_accounting":{
   "unique_frozen_initial_states":len(panel),
   "species_runs_full_horizon":len(panel)*len(d["results"]),
   "full_horizon_primary_periods_per_run":15000,
   "full_horizon_integrations_primary_periods":len(panel)*len(d["results"])*15000,
   "separate_benchmark_runs":48,"benchmark_primary_periods_per_run":100,
   "benchmark_not_pooled_with_final_ensemble":True},
 "limitations":[
  "The panel is a fixed coverage panel, not an iid draw; no capture probability or binomial CI is claimed.",
  "Escape is defined by first crossing rho=1 in the ideal normalized aperture; it is a model classification.",
  "The measured orbit radius and angular frequency are sampled at the primary RF phase once per primary period; high-frequency micromotion is audited separately.",
  "No Coulomb interaction, MFS higher harmonics, field boundary solution, or physical Cunningham drag is included."
 ]}
by_id={}
for r in d["results"]:
 name=r["species"]
 final=[x for x in r["windows"] if x["horizon_periods"]==15000]
 esc=[x for x in final if x["escape_period"] is not None and x["escape_period"]<=15000]
 kept=[x for x in final if x["escape_period"] is None or x["escape_period"]>15000]
 root=leading[name]
 rm=float(np.mean([x["rho_mean"] for x in kept]))
 om=float(np.mean([x["omega_mean"] for x in kept]))
 stats={
  "runs":len(final),"retained":len(kept),"escaped":len(esc),
  "escape_ids":[x["start_id"] for x in esc],
  "rotation_signs":{"positive":sum(x["omega_mean"]>0 for x in kept),
                    "negative":sum(x["omega_mean"]<0 for x in kept)},
  "mean_late_rho":rm,"sd_across_retained_start_rho":float(np.std([x["rho_mean"] for x in kept],ddof=1)),
  "mean_late_omega":om,"sd_across_retained_start_omega":float(np.std([x["omega_mean"] for x in kept],ddof=1)),
  "mean_sampled_rho_std":float(np.mean([x["rho_std"] for x in kept])),
  "max_late_rho_std":float(max(x["rho_std"] for x in kept)),
  "leading_rho":root["rho"],"leading_omega":root["omega"],
  "relative_rho_error_percent":100*(rm-root["rho"])/root["rho"],
  "relative_abs_omega_error_percent":100*(abs(om)-abs(root["omega"]))/abs(root["omega"]),
  "sign_agreement":all(np.sign(x["omega_mean"])==np.sign(root["omega"]) for x in kept)}
 summary["species"][name]=stats
 by_id[name]={x["start_id"]:x for x in final}
for h in (5000,10000,15000):
 summary["horizons"][str(h)]={}
 for r in d["results"]:
  name=r["species"]
  rows=[x for x in r["windows"] if x["horizon_periods"]==h]
  kept=[x for x in rows if x["escape_period"] is None or x["escape_period"]>h]
  root=leading[name]
  summary["horizons"][str(h)][name]={
    "retained":len(kept),"escaped_by_horizon":len(rows)-len(kept),
    "mean_late_rho":float(np.mean([x["rho_mean"] for x in kept])),
    "mean_late_omega":float(np.mean([x["omega_mean"] for x in kept])),
    "relative_rho_error_percent":float(100*(np.mean([x["rho_mean"] for x in kept])-root["rho"])/root["rho"]),
    "relative_abs_omega_error_percent":float(100*(np.mean([abs(x["omega_mean"]) for x in kept])-abs(root["omega"]))/abs(root["omega"]))}
 ids=set(by_id["species1"])|set(by_id["species2"])
 paired={"both_retained":0,"species1_only_retained":0,"species2_only_retained":0,
         "both_escaped":0,"counterrotating_retained_pairs":0,"same_sign_retained_pairs":0}
 for sid in sorted(ids):
  a=by_id["species1"][sid]; b=by_id["species2"][sid]
  ka=a["escape_period"] is None or a["escape_period"]>15000
  kb=b["escape_period"] is None or b["escape_period"]>15000
  if ka and kb:
   paired["both_retained"]+=1
   if a["omega_mean"]*b["omega_mean"]<0: paired["counterrotating_retained_pairs"]+=1
   else: paired["same_sign_retained_pairs"]+=1
  elif not ka and not kb: paired["both_escaped"]+=1
  elif ka: paired["species1_only_retained"]+=1
  else: paired["species2_only_retained"]+=1
summary["paired_start_outcomes"]=paired
OUT.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
