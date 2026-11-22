from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent; RES=HERE/'results'; MANIFEST=HERE/'MANIFEST.json'

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def read_json(path): return json.loads(path.read_text(encoding='utf-8'))
def read_jsonl(path): return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def main():
    cfg=read_json(RES/'input_configuration.json')
    scan=read_jsonl(RES/'coulomb_scan_corrected_drag.jsonl')
    keys=[(r['configuration'],float(r['coulomb_scale'])) for r in scan]
    assert len(scan)==20 and len(set(keys))==20, f'bad scan rows/duplicate IDs: {len(scan)}/{len(set(keys))}'
    assert all(r['periods_requested']==r['periods_completed']==15000 and r['retained'] and r['counterrotating_screen'] for r in scan)
    assert all(r['status']=='COMPLETE' for r in scan)
    stokes=read_jsonl(RES/'stokes_only_preliminary'/'coulomb_scan.jsonl')
    assert len(stokes)==15 and len(set((r['configuration'],float(r['coulomb_scale'])) for r in stokes))==15
    robust=read_json(RES/'finite_perturbation_robustness.json')['rows']
    assert len(robust)==26 and len(set((r['configuration'],r['perturbation']) for r in robust))==26
    assert all(sum(r['configuration']==c for r in robust)==13 for c in ('phys0570','scaled_refine12_sobol17_f50hz'))
    long=read_json(RES/'long_horizon_50000p.json')['rows']; assert len(long)==2 and all(r['periods_completed']==50000 and r['counterrotating_screen'] for r in long)
    balance=read_json(RES/'refined_energy_balance.json')['rows']; assert len(balance)==4
    validation=read_json(RES/'numerical_validation.json')['validation']; assert len(validation)==6 and all(r.get('success',True) for r in validation)
    minima=list(csv.DictReader((RES/'stroboscopic_minima.csv').open(encoding='utf-8'))); assert len(minima)==314
    checks={'scan_trajectory_npz':0,'stroboscopic_csv_rows':0,'finite_npz':0,'long_npz':0,'dop853_npz':0}
    for case in cfg['cases']:
        for r in scan:
            if r['configuration']!=case['id']: continue
            stem=f"{case['id']}_C{r['coulomb_scale']:g}_rk464_15000p.npz"
            p=RES/'trajectories'/stem;assert p.exists(),stem
            with np.load(p) as z:
                assert z['states'].shape==(15001,12), (stem,z['states'].shape)
                assert z['work_history'].shape==(15001,21)
                assert np.isfinite(z['states']).all() and np.isfinite(z['work_history']).all()
            checks['scan_trajectory_npz']+=1
        for scale in (0.0,1.0):
            p=RES/f'timeseries_{case["id"]}_C{scale:g}.csv'
            rows=list(csv.DictReader(p.open(encoding='utf-8'))); assert len(rows)==15001
            checks['stroboscopic_csv_rows']+=len(rows)
    for p in (RES/'independent_dop853_scaled_refine12_sobol17_f50hz_C1_3000p.npz',RES/'independent_dop853_phys0570_C1_3000p.npz'):
        assert p.exists()
        with np.load(p) as z: assert z['y'].shape==(12,48001) and np.isfinite(z['y']).all()
        checks['dop853_npz']+=1
    for p in (RES/'long_horizon_50000p.json',): assert p.exists()
    checks['long_npz']=len(list((RES/'trajectories').glob('*rk464_50000p_long.npz')))
    assert checks['long_npz']==2
    checks['finite_npz']=len(list((RES/'trajectories').glob('*_rk464_15000p.npz')))-20
    files=[]
    for p in sorted(HERE.rglob('*')):
        if not p.is_file() or p==MANIFEST: continue
        if '__pycache__' in p.parts or p.suffix in ('.pyc','.nbc','.nbi'): continue
        files.append({'path':p.relative_to(HERE).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    audit=read_json(RES/'model_and_history_audit.json')
    summary={
      'scan_rows':len(scan),'scan_unique_ids':len(set(keys)),'scan_cr_and_retained':sum(r['counterrotating_screen'] and r['retained'] for r in scan),
      'scan_status_counts':{name:sum(r['status']==name for r in scan) for name in sorted({r['status'] for r in scan})},
      'robustness_rows':len(robust),'robustness_per_configuration':{c:sum(r['configuration']==c and r['counterrotating_screen'] for r in robust) for c in ('phys0570','scaled_refine12_sobol17_f50hz')},
      'long_horizon_rows':len(long),'balance_rows':len(balance),'validation_rows':len(validation),'passage_windows':len(minima),
      'archive_replay_max_state_error':max(x['end_state_max_abs_diff'] for x in audit['archived_exact_replays'].values()),
      'MFS_gradient_fd_max_error':audit['force_field_checks']['finite_difference_MFS_potential_gradient_max_abs_error'],
      'files_checked':len(files),'file_groups':checks
    }
    manifest={'study':'V17C.17','branch':'research/v17c-two-particle-coulomb-dynamics','base_branch':'research/v17c-theory-validation','base_sha':'9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b','scope':'Only research/v17c/v17c_17 artifacts. Historical branches/results are unchanged.','audit_summary':summary,'files':files}
    MANIFEST.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
