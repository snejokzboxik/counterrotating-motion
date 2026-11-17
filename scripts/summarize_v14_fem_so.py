"""Paired four-replica h50 FEM Q/SO/RO comparison, using saved V13.3 Q/RO."""
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
    qro={}
    with OLD.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                key=(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))
                if key in qro:raise RuntimeError('duplicate QRO source')
                qro[key]=r
    so={}
    for line in (OUT/'fem_so_checkpoint.jsonl').read_text().splitlines():
        if line.strip():
            r=json.loads(line);key=(r['model'],r['replica'],r['phase_index'],r['state_i'])
            if key in so:raise RuntimeError('duplicate SO result')
            so[key]=r
    assert len(qro)==512 and len(so)==512
    allrows={**qro,**so}
    summary=[];rep_rows=[]
    for phase in (0,1,2,3,4):
      phases=(phase,) if phase<4 else (0,1,2,3)
      for model in ('Q','RO','SO_equal_harmonic','SO_equal_rms'):
        means=[]
        for rep in range(4):
            sample=[allrows[(model,rep,p,i)] for p in phases for i in range(16)]
            if any(r['status']=='unresolved' for r in sample):raise RuntimeError('unresolved FEM status')
            mean=sum(r['status']=='retained' for r in sample)/len(sample)
            means.append(mean)
            reprows=dict(phase_deg=90*phase if phase<4 else 'average',model=model,
                         replica=rep,P_ret=mean,n_retained=int(mean*len(sample)),n_total=len(sample))
            reprows.update(seed=(513217,883391,20261114,941029)[rep]);rep_rows.append(reprows)
        mean=float(np.mean(means));half=float(tdist.ppf(.975,3)*np.std(means,ddof=1)/2)
        summary.append(dict(phase_deg=90*phase if phase<4 else 'average',model=model,
                            P_ret=mean,ci95_low=mean-half,ci95_high=mean+half,
                            n_retained=int(round(mean*len(phases)*64)),n_total=len(phases)*64,
                            replicate_means=json.dumps(means)))
      for control in ('Q','SO_equal_harmonic','SO_equal_rms'):
        diff=[]
        for rep in range(4):
            d=np.mean([int(allrows[('RO',rep,p,i)]['status']=='retained')-
                       int(allrows[(control,rep,p,i)]['status']=='retained')
                       for p in phases for i in range(16)])
            diff.append(float(d))
        m=float(np.mean(diff));half=float(tdist.ppf(.975,3)*np.std(diff,ddof=1)/2)
        summary.append(dict(phase_deg=90*phase if phase<4 else 'average',model='RO-minus-'+control,
                            P_ret=m,ci95_low=m-half,ci95_high=m+half,
                            n_retained='',n_total=len(phases)*64,replicate_means=json.dumps(diff)))
    write_csv(OUT/'fem_so_paired_summary.csv',summary)
    write_csv(OUT/'fem_so_replicates.csv',rep_rows)
    for r in summary:
        if r['phase_deg']=='average':print(r['model'],r['P_ret'],r['ci95_low'],r['ci95_high'])

if __name__=='__main__':main()
