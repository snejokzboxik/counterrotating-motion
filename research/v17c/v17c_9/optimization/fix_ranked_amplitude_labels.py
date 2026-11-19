"""Re-label the frozen first-screen ratios using the archived MFS voltage factors."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
ratio=np.load(HERE/"stage1"/"configurations.npz")
p=HERE/"stage1"/"ranked.json"
rows=json.loads(p.read_text(encoding="utf-8"))
for r in rows:
    i=r["config_id"]; params=r["params"]
    params.pop("U2_U1",None)
    params["harmonic_potential_ratio_B_over_A"]=float(ratio["harmonic_potential_ratio_B_over_A"][i])
    params["drive_U2_U1_physical"]=float(ratio["drive_U2_U1_physical"][i])
p.write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
