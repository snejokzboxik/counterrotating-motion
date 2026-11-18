"""Coarse one-factor maps for the G2 thin six-electrode geometry."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from single_particle import load_geometry,integrate,classify,RESULTS
from guided_search import omega_circular

def main():
    RESULTS.mkdir(exist_ok=True); name='6_ABC_thin'; d,modes,coeff=load_geometry(name)
    lead=abs(coeff[np.where(modes==d['m'])[0][0]]); base=.1044/lead; root=.56741
    radii=[.18,.28,.36,.45,.54,.61,.70,.80]; scales=[.02,.05,.1,.2,.4,.7]
    cases=[]
    # q/m is the effective RF coefficient with U0 fixed; circles initialized
    # from the pseudopotential estimate and torque-predicted direction.
    for r in radii:
      sign=1 if r<root else -1
      for s in scales:
        a=base*s;w=omega_circular(r,modes,coeff,a)*sign
        for label,lam,om,aa in [('q_over_m',.0005,w,a),('rf_amplitude',.0005,w,a)]:
          tr,mx,ex=integrate(r,.371,0.,om,0.,modes,coeff,aa,lam,2000,64,True)
          cases.append(dict(map=label,r0=r,parameter=s,lambda_=lam,omega0=om,
             max_radius=float(mx),**classify(tr,2000,ex)))
    # r-by-damping map at common RF strength; radius-adapted initial spin.
    for r in radii:
      sign=1 if r<root else -1;om=sign*omega_circular(r,modes,coeff,base*.1)
      for lam in [.0001,.0005,.001,.002,.005,.01]:
        tr,mx,ex=integrate(r,.371,0.,om,0.,modes,coeff,base*.1,lam,2000,64,True)
        cases.append(dict(map='friction',r0=r,parameter=lam,lambda_=lam,omega0=om,
          max_radius=float(mx),**classify(tr,2000,ex)))
    # r-by-initial angular velocity map, holding drive and drag fixed.
    for r in radii:
      wc=omega_circular(r,modes,coeff,base*.1)
      for f in [-1.5,-1.,-.5,0.,.5,1.,1.5]:
        tr,mx,ex=integrate(r,.371,0.,f*wc,0.,modes,coeff,base*.1,.0005,2000,64,True)
        cases.append(dict(map='initial_omega',r0=r,parameter=f,lambda_=.0005,omega0=f*wc,
          max_radius=float(mx),**classify(tr,2000,ex)))
    with (RESULTS/'single_particle_maps.csv').open('w',newline='',encoding='utf-8') as f:
      w=csv.DictWriter(f,fieldnames=cases[0].keys());w.writeheader();w.writerows(cases)
    (RESULTS/'single_particle_maps.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    labels=['q_over_m','rf_amplitude','friction','initial_omega']
    fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors={'escape':0,'uncertain':1,'stationary':2,'positive':3,'negative':4,'transient':5}
    for ax,label in zip(axs.flat,labels):
      data=[x for x in cases if x['map']==label]
      xs=sorted(set(x['r0'] for x in data));ys=sorted(set(x['parameter'] for x in data))
      grid=np.full((len(ys),len(xs)),np.nan)
      for x in data:grid[ys.index(x['parameter']),xs.index(x['r0'])]=colors[x['state']]
      im=ax.imshow(grid,origin='lower',aspect='auto',extent=[min(xs),max(xs),min(ys),max(ys)],vmin=-.5,vmax=5.5,cmap='tab10')
      ax.axvline(root,color='white',ls='--',lw=1);ax.set_title(label);ax.set_xlabel('initial r/R');ax.set_ylabel('parameter')
    fig.savefig(RESULTS/'single_particle_maps.png',dpi=160)
    print('cases',len(cases),'states', {s:sum(x['state']==s for x in cases) for s in colors})

if __name__=='__main__':main()
