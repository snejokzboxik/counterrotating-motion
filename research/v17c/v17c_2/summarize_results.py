"""Export compact CSV views of the larger exploratory scans."""
import csv,json
from pathlib import Path
from collections import Counter
from single_particle import RESULTS

def write_csv(name,rows):
    if not rows:return
    fields=list(rows[0])
    with (RESULTS/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:
            w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})

def main():
    src=json.loads((RESULTS/'negative_damping_scan.json').read_text());write_csv('negative_damping_scan.csv',src)
    outer=json.loads((RESULTS/'outer_phase_search.json').read_text());write_csv('outer_phase_search.csv',outer)
    sync=json.loads((RESULTS/'synchronous_resonance_scan.json').read_text());write_csv('synchronous_resonance_scan.csv',sync)
    geom=json.loads((RESULTS/'averaged_orbit_equilibria.json').read_text())
    compact=[]
    for name in sorted(set(x['name'] for x in geom)):
      for lam in sorted(set(x['lambda_'] for x in geom if x['name']==name)):
       for sign in (-1,1):
        q=[x for x in geom if x['name']==name and x['lambda_']==lam and x['stable'] and x['torque_sign']==sign]
        if q:compact.append(dict(name=name,lambda_=lam,torque_sign=sign,count=len(q),r_min=min(x['r'] for x in q),
           r_max=max(x['r'] for x in q),alpha_min=min(x['amplitude'] for x in q),alpha_max=max(x['amplitude'] for x in q),
           max_eigenvalue_real=max(max(complex(*v).real for v in x['eigenvalues']) for x in q)))
    (RESULTS/'averaged_orbit_equilibria_summary.json').write_text(json.dumps(compact,indent=2),encoding='utf-8')
    summary=dict(
      negative_damping_runs=len(src),negative_damping_states=dict(Counter(x['state'] for x in src)),
      negative_damping_by_geometry={g:dict(Counter(x['state'] for x in src if x['name']==g)) for g in sorted(set(x['name'] for x in src))},
      outer_phase_runs=len(outer),outer_phase_negative_counts={g:sum(x['name']==g and x['state']=='negative' for x in outer) for g in sorted(set(x['name'] for x in outer))},
      synchronous_runs=len(sync),synchronous_state_counts=dict(Counter(f"{x['sign']}:{x['state']}" for x in sync)),
      averaged_fixedpoint_rows=len(geom),stable_fixedpoint_summary_rows=len(compact),
      note='Long-horizon and balance checks are listed in separate JSON summaries; short scan states are not confirmations.')
    (RESULTS/'search_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
