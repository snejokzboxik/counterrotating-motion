from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,time,hashlib
import numpy as np
import attractor_search as a

OUT=a.OUT

def main():
    cfg=a.read_json(OUT/'candidate_parameters.json');cases=cfg['candidates'];field=a.load_field()
    rows=[json.loads(x) for x in a.ROWS.read_text(encoding='utf-8').splitlines() if x.strip()]
    source_sha=a.sha256(a.V12/'results'/'rescaled_legacy_candidate_validation'/'frozen_starts.npz')
    # Eight evenly spaced frozen starts per published ensemble; no conditioning on the new outcomes.
    indices=list(range(0,128,16));ensembles=['rest','weak','wide','edge'];jobs=[]
    for c in cases:
      for e in ensembles:
        for i in indices:
          r=next(x for x in rows if x['configuration']==c['id'] and x['ensemble']==e and int(x['start_index'])==i)
          jobs.append((c,r))
    path=OUT/'basin_ensemble_checkpoint.jsonl';done=set()
    if path.exists():
      done={json.loads(x)['run_id'] for x in path.read_text(encoding='utf-8').splitlines() if x.strip()}
    todo=[x for x in jobs if f"{x[0]['id']}|{x[1]['ensemble']}|{x[1]['start_index']}" not in done]
    # Warm the compiled path before worker fanout.
    c,r=todo[0] if todo else jobs[0]
    a.run_summary(c,field,{'initial':r['initial'],'phase':r['phase']},2,32,0)
    t0=time.perf_counter();n=0
    def one(item):
      c,r=item;start={'initial':r['initial'],'phase':r['phase']}
      out=a.run_summary(c,field,start,15000,64,1000)
      out.update({'run_id':f"{c['id']}|{r['ensemble']}|{r['start_index']}",'candidate':c['id'],
       'ensemble':r['ensemble'],'start_index':int(r['start_index']),'source_seed':int(r['seed']),
       'initial':r['initial'],'phase':r['phase'],'source_v12b_category':r['strict_late_category'],
       'source_frozen_starts_sha256':source_sha,'sample_design':'deterministic systematic subsample of 8/128 indices: 0,16,...,112; descriptive, not IID'})
      return out
    with path.open('a',encoding='utf-8',buffering=1) as f:
      with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,x) for x in todo]
        for future in as_completed(futures):
          out=future.result();f.write(json.dumps(out,separators=(',',':'),allow_nan=False)+'\n');f.flush();n+=1
          if n%8==0 or n==len(todo):print(f'basin panel {n}/{len(todo)} new; elapsed {time.perf_counter()-t0:.1f}s',flush=True)
    allrows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    report={'source_frozen_starts_sha256':source_sha,'indices_per_ensemble':indices,'n_per_ensemble':len(indices),
      'periods':15000,'steps_per_rf':64,'transient_skip':1000,'denominator':'all scheduled starts; contacts, escapes and unresolved cases remain failures',
      'candidate_ensemble_results':[]}
    for c in cases:
      for e in ensembles:
        group=[x for x in allrows if x['candidate']==c['id'] and x['ensemble']==e]
        counts={k:sum(1 for x in group if (x['screen_class']==k if k not in ('STRICT_CR','NOT_STRICT_CR') else x['ten_turn_requirement_met']==(k=='STRICT_CR'))) for k in ['SCREEN_CR','SCREEN_CO','PARTICLE_CONTACT','WALL_CONTACT','ESCAPE','UNRESOLVED','STRICT_CR','NOT_STRICT_CR']}
        report['candidate_ensemble_results'].append({'candidate':c['id'],'ensemble':e,'n':len(group),'counts':counts,
          'strict_cr_fraction':counts['STRICT_CR']/len(group) if group else None,
          'retained_fraction':sum(x['retained'] for x in group)/len(group) if group else None})
    a.write_json(OUT/'basin_ensemble_summary.json',report)

if __name__=='__main__':main()
