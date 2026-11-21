"""Freeze the preregistered 24-start Local15 single-particle panel."""
import csv
import hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/frozen_local15_single_particle_panel.csv"
SEED=15152026

def build():
    rng=np.random.default_rng(SEED)
    radii=np.tile(np.array([0.35,0.45,0.55,0.65,0.75,0.85,0.92,0.96]),3)
    theta=rng.uniform(-np.pi,np.pi,24)
    phases1=rng.uniform(-np.pi,np.pi,24)
    phases2=rng.uniform(-np.pi,np.pi,24)
    kinds=["rest"]*8+["radial_velocity"]*8+["tangential_velocity"]*8
    speed=np.tile(np.linspace(0.005,0.04,8),3)
    rows=[]
    for i in range(24):
        vx=vy=0.0
        if i<8:
            pass
        elif i<16:
            sign=1.0 if i%2==0 else -1.0
            vx=sign*speed[i]*np.cos(theta[i])
            vy=sign*speed[i]*np.sin(theta[i])
        else:
            sign=1.0 if i%2==0 else -1.0
            vx=-sign*speed[i]*np.sin(theta[i])
            vy= sign*speed[i]*np.cos(theta[i])
        rows.append({
            "start_id":f"S{i+1:02d}","family":kinds[i],"rho0":float(radii[i]),
            "theta0":float(theta[i]),"x0":float(radii[i]*np.cos(theta[i])),
            "y0":float(radii[i]*np.sin(theta[i])),"vx0":float(vx),"vy0":float(vy),
            "phase1":float(phases1[i]),"phase2":float(phases2[i]),"seed":SEED
        })
    return rows

def main():
    rows=build()
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    digest=hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"frozen starts={len(rows)} seed={SEED} sha256={digest} file={OUT}")
if __name__=="__main__": main()
