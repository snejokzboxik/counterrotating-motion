"""Create a SHA-256 inventory for the V17C.21.1 deliverable."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE.parents[2]

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):
   h.update(block)
 return h.hexdigest()

def git(*args):
 return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()

def main():
 files=[]
 for p in sorted(BASE.rglob("*")):
  if not p.is_file() or p.name=="MANIFEST.json" or "__pycache__" in p.parts:
   continue
  files.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)})
 external_sources={
  "v21_analytical_model":"research/v17c/v17c_21/code/analytical_model.py",
  "v21_m2_stability_record":"research/v17c/v17c_21/results/sensitivity_and_stability.json",
  "v20b_reference_trajectory":"research/v17c/v17c_20b/results/refine12_240k_corrected_drag_trajectory.npz",
  "v20b_reference_summary":"research/v17c/v17c_20b/results/refine12_240k_corrected_drag_summary.json",
 }
 inputs={}
 for label,rel in external_sources.items():
  p=ROOT/rel
  if p.exists(): inputs[label]={"path":rel,"bytes":p.stat().st_size,"sha256":sha(p)}
 out={
  "study":"V17C.21.1 M3 convergence and stability",
  "branch":git("branch","--show-current"),
  "source_commit":"05a8d8ce811c9858ab1e74b67b15139c557018b1",
  "source_commit_verified":git("rev-parse","HEAD"),
  "generated_utc":subprocess.check_output(["powershell","-NoProfile","-Command","(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')"],cwd=ROOT,text=True).strip(),
  "manifest_excludes_itself":True,
  "file_count":len(files),
  "files":files,
  "external_input_files":inputs,
  "notes":[
   "Historical V20A/V20B/V21 files are inputs and remain unmodified.",
   "Generated PNG figures are saved at 600 dpi; vector counterparts are PDF and SVG.",
   "MANIFEST.json does not hash itself to avoid recursive content."
  ]
 }
 (BASE/"MANIFEST.json").write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
 print(json.dumps({"file_count":len(files),"external_input_files":inputs},indent=2,ensure_ascii=False))

if __name__=="__main__":main()
