"""Single longer full-RF check at the closest coarse CR grid point A/Anom=0.95."""
import json
from pathlib import Path
import full_rf_validation as full
import attractor_search as src

def main():
    payload=json.loads((full.BASE/"results"/"continuation_A.json").read_text(encoding="utf-8"))
    row=next(r for seq in payload["directions"].values() for r in seq if abs(r["factor_from_nominal"]-.95)<1e-12)
    spec={"parameter":"A","factor":.95,"row":row}
    case=src.load_candidates()["scaled_refine12_sobol17_f50hz"]
    start=src.choose_refine_long_start()
    field=src.load_field()
    row=full._run_case("M3_CR_A095_long",spec,case,start,field,26000,64,1000)
    summary_path=full.BASE/"results"/"full_rf_validation.json"
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    summary["rows"].append(row)
    summary["status"]="COMPLETE"
    summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")

if __name__=="__main__":main()
