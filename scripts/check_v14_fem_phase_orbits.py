"""Local FEM phase continuation of the RO 1-RF off-axis orbit."""
from __future__ import annotations
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import root
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import OUT,make_rhs,flow,jacobian,J
from run_v14_ideal_pilot import KAPPA
from analyze_v14_ideal import minima

def main():
    source=json.loads((OUT/'phase_ideal_orbits.json').read_text())
    base=json.loads((OUT/'fem_periodic_orbits.json').read_text())
    out=[]
    for mesh in (50,35):
      fem.initialize(mesh)
      previous=np.array(next(r['state'] for r in base if r['mesh_um']==mesh and r['model']=='RO'))
      if previous[1]<0:previous=-previous
      for deg in (45,90,135,180):
        angle=math.radians(deg);fun=make_rhs('RO',KAPPA,angle)
        seeds=[previous]
        seeds.extend(np.array(r['state']) for r in source if r['delta_deg']==deg and r['state'][1]>0)
        seeds.extend(np.r_[m,[0.,0.]] for m in minima('RO',KAPPA,angle) if m[1]>0)
        found=[]
        for ix,seed in enumerate(seeds):
            def residual(z):
                try:f,_=flow(z,fun)
                except Exception:return np.full(4,10.)+z-seed
                return f-z
            try:
                sol=root(residual,seed,method='hybr',options={'xtol':1e-10,'maxfev':100})
                final,rmax=flow(sol.x,fun,step=128,rtol=1e-11,atol=1e-13)
                resid=float(np.linalg.norm(final-sol.x,np.inf))
                if resid>1e-7 or np.linalg.norm(sol.x[:2])<.03 or rmax>=1:continue
                if any(np.linalg.norm(sol.x-np.array(r['state']))<1e-4 for r in found):continue
                M=jacobian(sol.x,fun,step=128);eig=np.linalg.eigvals(M)
                found.append(dict(mesh_um=mesh,delta_deg=deg,seed_index=ix,state=sol.x.tolist(),
                                  closure_residual=resid,max_radius=rmax,spectral_radius=float(max(abs(eig))),
                                  symplectic_defect=float(np.linalg.norm(M.T@J@M-J,np.inf)),
                                  multipliers=[[float(v.real),float(v.imag)] for v in eig]))
            except Exception:pass
        if found:previous=np.array(found[0]['state'])
        out.extend(found)
        print('phase FEM orbit',mesh,deg,'searched',len(seeds),'found',len(found),flush=True)
        (OUT/'fem_phase_orbit_search.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
