"""Directed single-ion orbit search around torque-reversal radii."""
import json,time
from pathlib import Path
import numpy as np
from single_particle import load_geometry,integrate,classify,RESULTS

def pseudo_potential(r,modes,coeff,amp):
    vals=[]
    for th in np.arange(256)*2*np.pi/256:
        z=r*np.exp(1j*th);ex=0j;ey=0j
        for k,c in zip(modes,coeff):
            n=abs(int(k));w=amp*n*c*(z**(n-1) if k>0 else z.conjugate()**(n-1))
            ex-=w;ey+=(-1j*w if k>0 else 1j*w)
        vals.append((abs(ex)**2+abs(ey)**2)/4)
    return float(np.mean(vals))

def omega_circular(r,modes,coeff,amp):
    h=2e-4;du=(pseudo_potential(r+h,modes,coeff,amp)-pseudo_potential(r-h,modes,coeff,amp))/(2*h)
    return float(np.sqrt(max(du/r,0.)))

def main():
    RESULTS.mkdir(exist_ok=True); checkpoint=RESULTS/'guided_checkpoint.json'
    rows=[]; trials=[]
    # Search both sides of the leading torque reversal for the prioritized ABC geometries.
    configs=[('6_ABC_moderate',[.40,.78],.6277),('6_ABC_thin',[.34,.76],.5674),
             ('9_ABC_moderate',[.58,.88],.7915),('12_ABC_moderate',[.70,.93],.8715)]
    for name,rs,root in configs:
        d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==d['m'])[0][0]])
        amp0=.1044/lead
        for r in rs:
            for af in (.35,.7,1.0):
                amp=amp0*af; w0=omega_circular(r,modes,coeff,amp)
                predicted_sign=1 if r<root else -1
                for lf in (.0005,.00477464829):
                    for wf in (.8,1.,1.2):
                        # Include both predicted sign and opposite initial sense as a basin control.
                        for sgn in (predicted_sign,-predicted_sign):
                            omega=sgn*w0*wf
                            tr,rmax,escaped=integrate(r,.371,0.,omega,0.,modes,coeff,amp,lf,2400,64,True)
                            diag=classify(tr,2400,escaped)
                            row=dict(name=name,r0=r,amp_factor=af,lambda_=lf,omega0=omega,
                                     omega_circular=w0*wf,expected_sign=predicted_sign,
                                     max_radius=float(rmax),**diag)
                            rows.append(row)
                            if not escaped and abs(diag['turns_total'])>=2 and diag['min_radius']>.015:
                                trials.append((row,tr.copy()))
                        if len(rows)%16==0:
                            checkpoint.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(name,r,'runs complete',len(rows),flush=True)
    (RESULTS/'guided_search.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    # Extend the strongest orbiting states, both signs, on 8000 RF to test long-term sign.
    selected=sorted(trials,key=lambda x:abs(x[0]['turns_total']),reverse=True)[:12]
    long=[]
    for j,(row,_) in enumerate(selected):
        d,modes,coeff=load_geometry(row['name']);lead=abs(coeff[np.where(modes==d['m'])[0][0]])
        amp=.1044/lead*row['amp_factor']
        tr,rmax,escaped=integrate(row['r0'],.371,0.,row['omega0'],0.,modes,coeff,amp,row['lambda_'],8000,96,True)
        diag=classify(tr,8000,escaped); rr=dict(source=row,max_radius=float(rmax),**diag)
        long.append(rr)
        if not escaped and abs(diag['turns_total'])>=10 and diag['min_radius']>.015:
            np.savez_compressed(RESULTS/f'long_orbit_{j}.npz',tr=tr,turns=diag['turns_total'],source=json.dumps(row))
        print('long',j,row['name'],row['r0'],diag['state'],diag['turns_total'],diag['turns_windows'],flush=True)
    (RESULTS/'guided_long.json').write_text(json.dumps(long,indent=2),encoding='utf-8')

if __name__=='__main__':main()
