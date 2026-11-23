from pathlib import Path
import json,hashlib,datetime,platform
import numpy,scipy,numba,matplotlib

ROOT=Path(__file__).resolve().parents[1]
MAN=ROOT/'MANIFEST.json'
def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==MAN:continue
        if '__pycache__' in p.parts or p.suffix.lower() in ('.pyc','.nbc','.nbi','.tmp'):continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':h})
    doc={'study':'V17C.20B','source_branch':'research/v17c18b19-axial-stability-diagrams','source_commit':'45866d0559db1615a5e2aebc521ddae90f3a0101',
      'result_branch':'research/v17c20b-particle-attractor-search','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'runtime':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'numba':numba.__version__,'matplotlib':matplotlib.__version__},
      'files_count':len(files),'total_bytes':sum(x['bytes'] for x in files),'files':files,
      'interpretation':'Checksums cover code, reports, plots, configurations, trajectories and checkpoint files. Historical source data are referenced by hashes in candidate_parameters.json and audit_checks.json.'}
    MAN.write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f"manifest: {len(files)} files, {sum(x['bytes'] for x in files)} bytes")
if __name__=='__main__':main()
