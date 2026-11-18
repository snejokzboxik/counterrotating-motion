"""Post-process saved V17C.5 checkpoints without integrating any trajectory."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RK = ROOT/"rk"/"results"
DOP = ROOT/"dop"
with np.load(RK/"nom64.npz") as d64, np.load(RK/"nom128.npz") as d128:
    states64=d64["state"].reshape(-1,2,6)
    states128=d128["state"].reshape(-1,2,6)
with np.load(DOP/"continuous.npz") as d:
    states_dop=d["state"][:,:12].reshape(-1,2,6)
transient=2000
comparisons={}
for name, left, right in [
    ("RK4_64_vs_RK4_128",states64,states128),
    ("DOP853_vs_RK4_128",states_dop,states128),
]:
    diff=np.abs(left-right)
    comparisons[name]={
        "max_cartesian_state_abs_all_periods":float(diff.max()),
        "max_cartesian_state_abs_after_transient":float(diff[transient:].max()),
        "final_cartesian_state_abs_max":float(diff[-1].max()),
        "mean_radius_stroboscopic_left":np.linalg.norm(left[transient:,:,:2],axis=2).mean(axis=0).tolist(),
        "mean_radius_stroboscopic_right":np.linalg.norm(right[transient:,:,:2],axis=2).mean(axis=0).tolist()
    }
dop=json.loads((DOP/"continuous.json").read_text())
rk=json.loads((RK/"nom128.json").read_text())
out={
    "basis":"all three integrations sample at integer primary RF periods",
    "transient_periods_discarded":transient,
    "comparisons":comparisons,
    "nominal_turns_DOP853":dop["turns_after_transient"],
    "nominal_turns_RK4_128":rk["turns"],
    "nominal_frequency_Hz_DOP853":dop["frequency_Hz"],
    "nominal_frequency_Hz_RK4_128":rk["hz"]
}
(ROOT/"independent_comparison.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,indent=2))
