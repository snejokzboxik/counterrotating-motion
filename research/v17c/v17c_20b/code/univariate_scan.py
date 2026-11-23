from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time
import attractor_search as a

OUT=a.OUT
def tasks(cfg):
    scales=[.6,.8,1.2,1.4]
    params=['alpha1','alpha2','lambda1','lambda2','mu','C12']
    for c in cfg['candidates']:
      for par in params:
        for mult in scales: yield c,par,mult
def overrides(c,key,m):
    base={'alpha1':c['alpha1'],'alpha2':c['alpha2'],'lambda1':c['lambda_corrected_divide_Cc'][0],
      'lambda2':c['lambda_corrected_divide_Cc'][1],'mu':c['mu'],'C12':c['p_corrected'][5]}
    v=base[key]*m
    return {'lam1':v} if key=='lambda1' else {'lam2':v} if key=='lambda2' else {'C':v} if key=='C12' else {key:v}
def main():
    cfg=a.read_json(OUT/'candidate_parameters.json');field=a.load_field();start=cfg['matched_common_start'];path=OUT/'univariate_scan_checkpoint.jsonl';done=set()
    if path.exists():done={json.loads(x)['run_id'] for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}
    alltasks=list(tasks(cfg));todo=[t for t in alltasks if f"{t[0]['id']}|{t[1]}|{t[2]}" not in done]
    t0=time.perf_counter()
    def one(t):
      c,par,m=t;r=a.run_summary(c,field,start,3000,32,600,**overrides(c,par,m));r.update({'run_id':f'{c["id"]}|{par}|{m}','candidate':c['id'],'parameter':par,'nominal_multiplier':m})
      return r
    with path.open('a',encoding='utf-8',buffering=1) as f:
      with ThreadPoolExecutor(max_workers=4) as pool:
        futs=[pool.submit(one,x) for x in todo]
        for i,fut in enumerate(as_completed(futs),1):
          r=fut.result();f.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');f.flush()
          if i%12==0 or i==len(todo):print(f'univariate {i}/{len(todo)}; elapsed {time.perf_counter()-t0:.1f}s',flush=True)
if __name__=='__main__':main()
