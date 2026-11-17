"""Summarize independent A9 confirmation and continuous 1000-RF extension."""
from __future__ import annotations
import csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15/confirmation_a9_uniform4d"

def ci(x):
    a=np.asarray(x,float); m=float(np.mean(a))
    h=float(student_t.ppf(.975,len(a)-1)*np.std(a,ddof=1)/math.sqrt(len(a))) if len(a)>1 else math.nan
    return m,m-h,m+h
def write(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    base=[json.loads(x) for x in (OUT/'checkpoint.jsonl').read_text(encoding='utf8').splitlines() if x]
    long=[json.loads(x) for x in (OUT/'checkpoint_1000_extension.jsonl').read_text(encoding='utf8').splitlines() if x]
    rows=base+long
    if len({r['key'] for r in rows})!=len(rows): raise RuntimeError('duplicate individual key across horizons')
    grouped=defaultdict(list)
    for r in rows: grouped[(r['delta_rad'],r['tau0_rad'],r['config'],r['horizon_rf'],r['replica'])].append(r)
    repmetrics={}; summaries=[]
    for key,rr in grouped.items():
        delta,tau,c,h,rep=key
        plus=[r for r in rr if r['Lz_sign']>0];minus=[r for r in rr if r['Lz_sign']<0]
        pp=float(np.mean([r['status']=='retained' for r in plus]));pm=float(np.mean([r['status']=='retained' for r in minus]))
        allret=float(np.mean([r['status']=='retained' for r in rr]))
        vals={'P_plus':pp,'P_minus':pm,'P_total':allret,'Delta_chi':pp-pm,
              'Delta_Lz_all':float(np.mean([r['Delta_Lz'] for r in rr])),
              'Delta_Lz_retained':float(np.mean([r['Delta_Lz'] for r in rr if r['status']=='retained'])) if any(r['status']=='retained' for r in rr) else math.nan,
              'Delta_Lz_escaped':float(np.mean([r['Delta_Lz'] for r in rr if r['status']=='escaped'])) if any(r['status']=='escaped' for r in rr) else math.nan,
              'median_exit_RF':float(np.median([r['exit_rf'] for r in rr if r['status']=='escaped'])) if any(r['status']=='escaped' for r in rr) else math.nan,
              'max_abs_torque_residual':max(abs(r['torque_residual']) for r in rr),
              'n_unresolved':sum(r['status']=='unresolved' for r in rr)}
        repmetrics[key]=vals
    bygroup=defaultdict(lambda:defaultdict(list))
    for (d,tau,c,h,rep),vals in repmetrics.items():
        for m,v in vals.items():
            if m=='max_abs_torque_residual': continue
            bygroup[(d,tau,c,h)][m].append(v)
        bygroup[(d,tau,c,h)]['max_abs_torque_residual'].append(vals['max_abs_torque_residual'])
    for (d,tau,c,h),metrics in sorted(bygroup.items()):
        for m,v in metrics.items():
            if m=='max_abs_torque_residual': mean,lo,hi=max(v),None,None
            elif np.isfinite(v).any(): mean,lo,hi=ci(np.asarray(v)[np.isfinite(v)])
            else: mean,lo,hi=None,None,None
            summaries.append({'delta_rad':d,'tau0_rad':tau,'config':c,'horizon_rf':h,'metric':m,'mean':mean,'ci95_low':lo,'ci95_high':hi,
                              'replicate_values':json.dumps(v,separators=(',',':')),'n_scrambles':len(v)})
    write(OUT/'summary_by_phase_model_horizon.csv',summaries)
    pairwise=[]
    for (d,tau,c,h),metrics in bygroup.items():
        if c not in ('RO_PLUS','RO_MINUS'): continue
        for control in ('Q','SO_equal_harmonic','SO_equal_rms'):
            diffs=[]
            for rep in range(8):
                a=repmetrics[(d,tau,c,h,rep)]['P_total'];b=repmetrics[(d,tau,control,h,rep)]['P_total']
                diffs.append(a-b)
            m,lo,hi=ci(diffs)
            pairwise.append({'delta_rad':d,'tau0_rad':tau,'horizon_rf':h,'contrast':f'{c}-{control}',
                'mean_difference_pp':100*m,'ci95_low_pp':100*lo,'ci95_high_pp':100*hi,
                'replicate_differences_pp':json.dumps([100*x for x in diffs],separators=(',',':')),
                'n_scrambles':len(diffs)})
    write(OUT/'paired_retention_differences.csv',pairwise)
    # Pairwise exact symmetries and data integrity.
    lookup={(r['delta_rad'],r['tau0_rad'],r['horizon_rf'],r['replica'],r['pair_index'],r['config'],r['Lz_sign']):r for r in rows}
    so_mismatch=0; rev_mismatch=0
    for (d,tau,h,rep,pair,c,sign),r in lookup.items():
        if c.startswith('SO_'):
            other=lookup.get((d,tau,h,rep,pair,c,-sign))
            if other and r['status']!=other['status']: so_mismatch+=1
        if c=='RO_PLUS':
            other=lookup.get((d,tau,h,rep,pair,'RO_MINUS',-sign))
            if other and r['status']!=other['status']: rev_mismatch+=1
    audit={'rows':len(rows),'unique_keys':len(lookup),'horizons':sorted({r['horizon_rf'] for r in rows}),
           'unresolved':sum(r['status']=='unresolved' for r in rows),'over_250V':sum(r['peak_electrode_V']>250 for r in rows),
           'max_abs_torque_identity_residual':max(abs(r['torque_residual']) for r in rows),
           'SO_mirror_status_mismatches':so_mismatch,'ROplus_to_ROminus_mirror_mismatches':rev_mismatch,
           'n_scrambled_sobol_replicates':8,'pairs_per_replicate':128,'total_pairs_per_condition':1024,
           'protocol_hash':json.loads((OUT/'manifest.json').read_text(encoding='utf8'))['protocol_hash']}
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    # Plots at delta=tau0=0: actual sampled endpoints only, no interpolated line.
    fdir=OUT/'figures';fdir.mkdir(exist_ok=True)
    configs=('Q','SO_equal_harmonic','SO_equal_rms','RO_PLUS','RO_MINUS')
    colors={'Q':'#333333','SO_equal_harmonic':'#3c8d5a','SO_equal_rms':'#7eaa78','RO_PLUS':'#a51c30','RO_MINUS':'#2455a4'}
    fig,ax=plt.subplots(figsize=(9,5.4))
    for c in configs:
        means=[];lo=[];hi=[]
        for h in (100,300,1000):
            v=[repmetrics[(0.,0.,c,h,rep)]['P_total'] for rep in range(8)]
            m,l,u=ci(v);means.append(100*m);lo.append(100*(m-l));hi.append(100*(u-m))
        ax.errorbar([100,300,1000],means,yerr=[lo,hi],marker='o',capsize=4,color=colors[c],label=c)
    ax.set_xscale('log');ax.set_xticks([100,300,1000],labels=['100','300','1000'])
    ax.set_xlabel('Горизонт интегрирования, RF-периоды');ax.set_ylabel('Доля удержанных, %')
    ax.set_title('A9, μ=1, δ=τ₀=0°: Q, SO и RO на общем ансамбле')
    ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
    fig.savefig(fdir/'v15_a9_retention_horizon_uniform4d.png',dpi=220);fig.savefig(fdir/'v15_a9_retention_horizon_uniform4d.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.4))
    for c in ('RO_PLUS','RO_MINUS','SO_equal_harmonic'):
        means=[];lo=[];hi=[]
        for h in (100,300,1000):
            v=[repmetrics[(0.,0.,c,h,rep)]['Delta_chi'] for rep in range(8)]
            m,l,u=ci(v);means.append(100*m);lo.append(100*(m-l));hi.append(100*(u-m))
        ax.errorbar([100,300,1000],means,yerr=[lo,hi],marker='o',capsize=4,color=colors[c],label=c)
    ax.axhline(0,color='black',lw=.7);ax.set_xscale('log');ax.set_xticks([100,300,1000],labels=['100','300','1000'])
    ax.set_xlabel('Горизонт, RF-периоды');ax.set_ylabel('Δχ = P₊ − P₋, п.п.')
    ax.set_title('A9, μ=1, δ=τ₀=0°: направленный контраст удержания')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(fdir/'v15_a9_chirality_horizon_uniform4d.png',dpi=220);fig.savefig(fdir/'v15_a9_chirality_horizon_uniform4d.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.4))
    for c in configs:
        means=[];lo=[];hi=[]
        for h in (100,300,1000):
            v=[repmetrics[(0.,0.,c,h,rep)]['Delta_Lz_all'] for rep in range(8)]
            m,l,u=ci(v);means.append(m);lo.append(m-l);hi.append(u-m)
        ax.errorbar([100,300,1000],means,yerr=[lo,hi],marker='o',capsize=4,color=colors[c],label=c)
    ax.axhline(0,color='black',lw=.7);ax.set_xscale('log');ax.set_xticks([100,300,1000],labels=['100','300','1000'])
    ax.set_xlabel('Горизонт, RF-периоды');ax.set_ylabel('⟨ΔLz⟩/(MΩR²), безразмерно')
    ax.set_title('A9, μ=1, δ=τ₀=0°: накопленный угловой момент')
    ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8);fig.tight_layout()
    fig.savefig(fdir/'v15_a9_angular_momentum_horizon_uniform4d.png',dpi=220);fig.savefig(fdir/'v15_a9_angular_momentum_horizon_uniform4d.svg');plt.close(fig)
    print(json.dumps(audit,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
