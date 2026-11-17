"""Basic frequency diagnostics of 300-period V14 four-dimensional strobe records."""
from __future__ import annotations
import csv,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_ideal_pilot import OUT
from analyze_v14_ideal import write_csv

def main():
    groups={}
    with (OUT/'stroboscopic_states.csv').open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):groups.setdefault((r['field'],r['model'],r['start_id']),[]).append(r)
    rows=[]
    for key,group in groups.items():
        group.sort(key=lambda r:int(r['n_rf']))
        x=np.array([float(r['x']) for r in group]);y=np.array([float(r['y']) for r in group])
        vx=np.array([float(r['vx']) for r in group]);vy=np.array([float(r['vy']) for r in group])
        out=dict(field=key[0],model=key[1],start_id=key[2],n_strobes=len(group),
                 y_mean=float(np.mean(y)),y_min=float(np.min(y)),y_max=float(np.max(y)),
                 max_stroboscopic_rho=float(np.max(np.hypot(x,y))))
        if len(group)>=256:
            for coord,values in (('x',x),('y',y),('vx',vx),('vy',vy)):
                v=values-np.mean(values);power=abs(np.fft.rfft(v*np.hanning(len(v))))**2
                frequencies=np.fft.rfftfreq(len(v))
                if len(power)>1:
                    index=1+np.argmax(power[1:]);out[f'{coord}_dominant_cycles_per_rf']=float(frequencies[index])
                    out[f'{coord}_peak_power']=float(power[index])
        rows.append(out)
    write_csv(OUT/'stroboscopic_spectra.csv',rows)
    for r in rows:print(r['field'],r['model'],r['start_id'],r.get('y_dominant_cycles_per_rf'))

if __name__=='__main__':main()
