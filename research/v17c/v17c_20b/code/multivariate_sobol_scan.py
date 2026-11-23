from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time
import numpy as np
from scipy.stats import qmc
import attractor_search as a

OUT=a.OUT
PARAMS=['alpha1','alpha2','lambda1','lambda2','mu','C12']
SEED=20261123
N=32
def main():
 cfg=a.read_json(OUT/'candidate_parameters.json');field=a.load_field();start=cfg['matched_common_start'];path=OUT/'multivariate_sobol_checkpoint.jsonl';done=set()
 if path.exists():done={json.loads(x)['run_id'] for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}
 tasks=[]
 for ci,c in enumerate(cfg['candidates']):
  u=qmc.Sobol(d=len(PARAMS),scramble=True,seed=SEED+ci).random_base2(m=5)
  for i,row in enumerate(u):
   mult=.6+.8*row
   vals=dict(zip(PARAMS,mult.tolist()));ov={}
   for key in PARAMS:
    v=vals[key]
    if key=='lambda1':ov['lam1']=c['lambda_corrected_divide_Cc'][0]*v
    elif key=='lambda2':ov['lam2']=c['lambda_corrected_divide_Cc'][1]*v
    elif key=='C12':ov['C']=c['p_corrected'][5]*v
    else:ov[key]=({ 'alpha1':c['alpha1'],'alpha2':c['alpha2'],'mu':c['mu']}[key])*v
   tasks.append((c,i,vals,ov))
 todo=[x for x in tasks if f"{x[0]['id']}|sobol|{x[1]}" not in done];t0=time.perf_counter()
 def one(t):
  c,i,vals,ov=t;r=a.run_summary(c,field,start,3000,32,600,**ov)
  r.update({'run_id':f"{c['id']}|sobol|{i}",'candidate':c['id'],'design':'scrambled Sobol','seed':SEED+cfg['candidates'].index(c),
    'index':i,'nominal_multipliers':vals,'derived_Q':r['parameters']['Q_derived'],'mathematical_scan':True})
  return r
 with path.open('a',encoding='utf-8',buffering=1) as f:
  with ThreadPoolExecutor(max_workers=4) as pool:
   futs=[pool.submit(one,x) for x in todo]
   for k,fut in enumerate(as_completed(futs),1):
    r=fut.result();f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');f.flush()
    if k%16==0 or k==len(todo):print(f'Sobol {k}/{len(todo)} elapsed {time.perf_counter()-t0:.1f}s',flush=True)
if __name__=='__main__':main()
