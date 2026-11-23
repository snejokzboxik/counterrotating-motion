from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import attractor_search as a

OUT=a.OUT;ROOT=OUT.parent
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def check_unique(path,key):
 r=lines(path);ids=[x[key] for x in r];return {'rows':len(r),'unique':len(set(ids)),'duplicates':len(ids)-len(set(ids))}

def main():
 checks={}
 for fn,key,expect in [('parameter_grid_checkpoint.jsonl','run_id',504),('coulomb_sensitivity_checkpoint.jsonl','run_id',30),
   ('univariate_scan_checkpoint.jsonl','run_id',72),('multivariate_sobol_checkpoint.jsonl','run_id',96),('basin_ensemble_checkpoint.jsonl','run_id',96)]:
  q=check_unique(OUT/fn,key);checks[fn]=q;assert q['rows']==expect and q['unique']==expect and q['duplicates']==0,q
 base=a.read_json(OUT/'baseline_reproduction.json')['rows'];assert len(base)==3
 for r in base: assert r['status']==0 and r['ten_turn_CR'] and r['pair_min_R']>0
 checks['baseline']={'rows':len(base),'all_retained':all(r['retained'] for r in base),'all_ten_turn_CR':all(r['ten_turn_CR'] for r in base)}
 for name in ('candidate52_50000_common_corrected','local15_50000_common_corrected'):
  s=a.read_json(OUT/f'{name}_summary.json');assert s['status']==0 and s['completed_periods']==50000 and s['all_late_chunks_counterrotating']
  z=np.load(OUT/f'{name}_trajectory.npz');assert z['states'].shape[0]==501 and np.isfinite(z['states']).all()
 s=a.read_json(OUT/'refine12_240k_corrected_drag_summary.json');assert s['status']==0 and s['completed_periods']==240000 and s['all_late_chunks_counterrotating']
 en=s['energy_angular_momentum'];assert en['relative_energy_residual']<2e-4 and abs(en['internal_Coulomb_torque_sum'])<1e-14
 z=np.load(OUT/'refine12_240k_corrected_drag_trajectory.npz');assert z['states'].shape[0]==2401 and np.isfinite(z['states']).all()
 checks['long_runs']={'candidate52_periods':50000,'Local15_periods':50000,'refine12_periods':240000,'all_retained':True}
 dop=a.read_json(OUT/'independent_dop853_validation.json')['rows'];ref=a.read_json(OUT/'step_refinement_3000p.json')['rows']
 assert len(dop)==3 and len(ref)==3 and all(x['success'] for x in dop) and all(x['status']==0 for x in ref)
 checks['independent_integrators']={'DOP853_runs':len(dop),'RK4_128_runs':len(ref),'all_opposite_signs':all(x['frequency_Hz'][0]>0 and x['frequency_Hz'][1]<0 for x in dop+ref)}
 va=a.read_json(OUT/'variational_audit.json');assert len(va['rows'])==3 and all(x['max_relative_jacobian_error']<1e-7 for x in va['rows'])
 checks['variational_jacobian']={'candidates':len(va['rows']),'max_error':max(x['max_relative_jacobian_error'] for x in va['rows'])}
 basin=a.read_json(OUT/'basin_ensemble_summary.json');assert len(basin['candidate_ensemble_results'])==12 and all(x['n']==8 for x in basin['candidate_ensemble_results'])
 checks['basin_panel']={'groups':len(basin['candidate_ensemble_results']),'total':sum(x['n'] for x in basin['candidate_ensemble_results'])}
 # All checkpoint chunks must form a complete, non-overlapping integer range.
 chunk={}
 for name,expect in [('candidate52_50000_common_corrected',10),('local15_50000_common_corrected',10),('refine12_240k_corrected_drag',48)]:
  files=sorted((OUT/'checkpoints'/name).glob('chunk_*.npz'));assert len(files)==expect,(name,len(files))
  ids=[int(f.stem.split('_')[-1]) for f in files];assert ids==list(range(expect)),(name,ids)
  for i,f in enumerate(files):
   z=np.load(f);assert np.isfinite(z['states']).all() and np.isfinite(z['work_history']).all()
   assert int(z['chunk_start_period'])==i*5000 and int(z['chunk_periods'])==5000
  chunk[name]=len(files)
 checks['checkpoint_chunks']=chunk
 checks['source_integrity']={'frozen_starts_sha256':hashlib.sha256((a.V12/'results'/'rescaled_legacy_candidate_validation'/'frozen_starts.npz').read_bytes()).hexdigest(),
   'field_sha256':a.sha256(a.FIELD_PATH),'integrator_sha256':a.sha256(a.V12/'results'/'rescaled_legacy_candidate_validation'/'campaign_integrator_source.py')}
 a.write_json(ROOT/'results'/'audit_checks.json',checks)
 print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
