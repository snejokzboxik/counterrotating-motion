"""Broader Sobol and directed initial-condition search; resumable checkpoints."""
import json
import time
import numpy as np
from scipy.stats import qmc
from scipy.optimize import differential_evolution
from two_particle_search import OUT, coefficients, initial, integrate, diagnostics


def evaluate(vec, cf, periods=600, steps=48):
    r1,r2,angle,dz,s1,s2,vr1,vr2,vz1,vz2,phase,C,w,g1,g2,m2,q2=vec
    y=initial(r1,r2,angle,dz,s1,s2,vr1,vr2,vz1,vz2)
    masses=np.array([1.,m2]);charges=np.array([1.,q2]);gamma=np.array([g1,g2])
    t,tr,sep,maxr,escaped=integrate(y,cf,C,w,gamma,masses,charges,phase,periods,steps)
    diag=diagnostics(t,tr,periods,escaped)
    wins=np.asarray(diag['windows_turns'])
    positive=min(wins[:,0].min(),-wins[:,1].max())
    negative=min(wins[:,1].min(),-wins[:,0].max())
    score=max(positive,negative)
    if escaped: score-=3
    if sep<.03:score-=1
    if diag['min_radius']<.015:score-=1
    return score,dict(parameters=np.asarray(vec).tolist(),score=float(score),minsep=float(sep),
                      maxr=float(maxr),**diag)


def main():
    OUT.mkdir(exist_ok=True)
    cf=coefficients();bounds=np.array([[.12,.65],[.12,.65],[.2,3.1],[0,1.0],
         [-.15,.15],[-.15,.15],[-.025,.025],[-.025,.025],[-.025,.025],[-.025,.025],
         [0,2*np.pi],[1e-6,5e-4],[.025,.12],[.0005,.008],[.0005,.008],[.5,3],[.5,2]])
    sampler=qmc.Sobol(d=len(bounds),scramble=True,seed=1717)
    pts=qmc.scale(sampler.random_base2(10),bounds[:,0],bounds[:,1]);pts[:,11]=10**np.interp(pts[:,11],[1e-6,5e-4],[-6,-3.3])
    checkpoint=OUT/'adaptive_checkpoint.json'
    rows=json.loads(checkpoint.read_text(encoding='utf-8')) if checkpoint.exists() else []
    start=time.perf_counter()
    for i,vec in enumerate(pts):
        if i<len(rows): continue
        _,row=evaluate(vec,cf);rows.append(row)
        if i%32==31:
            (OUT/'adaptive_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print('Sobol',i+1,'best',max(r['score'] for r in rows),flush=True)
    # Continue around the strongest surviving Sobol state with a global optimizer.
    center=np.asarray(max(rows,key=lambda r:r['score'])['parameters'])
    local=[tuple(x) for x in bounds[:6]]
    for side in (0,1):
        def fun(v):
            trial=center.copy();trial[:6]=v
            trial[4+side]=abs(trial[4+side])*(1 if side==0 else -1)
            score,_=evaluate(trial,cf,periods=600)
            return -score
        opt=differential_evolution(fun,local,maxiter=8,popsize=5,seed=177+side,
                                   polish=False,workers=1,updating='immediate')
        trial=center.copy();trial[:6]=opt.x
        score,row=evaluate(trial,cf,periods=600)
        row['optimization_side']=side;rows.append(row)
        print('DE',side,score,flush=True)
    winners=sorted(rows,key=lambda r:r['score'],reverse=True)[:6]
    extended=[]
    for j,row in enumerate(winners):
        vec=np.asarray(row['parameters']);score,long=evaluate(vec,cf,periods=2400)
        long['source_score']=row['score'];extended.append(long)
        print('extended',j,score,long['windows_turns'],flush=True)
    (OUT/'adaptive_summary.json').write_text(json.dumps(dict(screened=len(rows),elapsed_s=time.perf_counter()-start,
        best_score=max(r['score'] for r in rows),winners=winners,extended=extended),indent=2),encoding='utf-8')


if __name__=='__main__': main()
