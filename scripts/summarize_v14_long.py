"""Four-replica paired h50 survival/exit summaries at 100, 300, 1000 RF."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.stats import t as tdist
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_fem_so import OUT,OLD
from analyze_v14_ideal import write_csv

def main():
    base={}
    with OLD.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                base[(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))]=r
    for line in (OUT/'fem_so_checkpoint.jsonl').read_text().splitlines():
        if line.strip():
            r=json.loads(line);base[(r['model'],r['replica'],r['phase_index'],r['state_i'])]=r
    long={}
    for line in (OUT/'fem_long_checkpoint.jsonl').read_text().splitlines():
        if line.strip():
            r=json.loads(line);key=(r['horizon'],r['model'],r['replica'],r['phase_index'],r['state_i'])
            if key in long:raise RuntimeError('duplicate long key')
            long[key]=r
    assert len(base)==1024 and len(long)==2048
    summary=[]
    for horizon in (100,300,1000):
      def row(model,rep,p,i):
          return base[(model,rep,p,i)] if horizon==100 else long[(horizon,model,rep,p,i)]
      for model in ('Q','RO','SO_equal_harmonic','SO_equal_rms'):
        means=[sum(row(model,rep,p,i)['status']=='retained' for p in range(4) for i in range(16))/64 for rep in range(4)]
        m=float(np.mean(means));half=float(tdist.ppf(.975,3)*np.std(means,ddof=1)/2)
        exits=[float(row(model,rep,p,i)['exit_periods']) for rep in range(4) for p in range(4) for i in range(16)
               if row(model,rep,p,i)['status']=='escaped' and row(model,rep,p,i)['exit_periods'] is not None and str(row(model,rep,p,i)['exit_periods'])!='']
        summary.append(dict(horizon=horizon,model=model,P_ret=m,ci95_low=m-half,ci95_high=m+half,
                            retained=int(round(m*256)),n=256,median_exit_rf=float(np.median(exits)),
                            replicate_means=json.dumps(means)))
      for control in ('Q','SO_equal_harmonic','SO_equal_rms'):
        diff=[np.mean([int(row('RO',rep,p,i)['status']=='retained')-int(row(control,rep,p,i)['status']=='retained')
                       for p in range(4) for i in range(16)]) for rep in range(4)]
        m=float(np.mean(diff));half=float(tdist.ppf(.975,3)*np.std(diff,ddof=1)/2)
        summary.append(dict(horizon=horizon,model='RO-minus-'+control,P_ret=m,ci95_low=m-half,
                            ci95_high=m+half,retained='',n=256,median_exit_rf='',
                            replicate_means=json.dumps([float(v) for v in diff])))
    write_csv(OUT/'fem_long_summary.csv',summary)
    for r in summary:
        if r['horizon']==1000:print(r['model'],r['P_ret'],r['ci95_low'],r['ci95_high'])

if __name__=='__main__':main()
