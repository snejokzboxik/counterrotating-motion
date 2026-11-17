"""Check whether the saved quadrupole FEM basis itself has off-axis RF orbits."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.optimize import root
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import OUT,make_rhs,flow,jacobian,J

def main():
    seeds=[]
    with (OUT/'stroboscopic_states.csv').open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['field']=='fem_h50' and r['model']=='Q' and r['start_id']=='near_RO_ideal' and int(r['n_rf'])%40==0:
                seeds.append(np.array([float(r[k]) for k in ('x','y','vx','vy')]))
    out=[]
    for mesh in (50,35):
      fem.initialize(mesh);fun=make_rhs('Q',0.,0.)
      found=[]
      for ix,seed in enumerate(seeds):
        def residual(z):
            try:final,_=flow(z,fun)
            except Exception:return np.full(4,10.)+z-seed
            return final-z
        try:
            sol=root(residual,seed,method='hybr',options={'xtol':1e-10,'maxfev':80})
            final,rmax=flow(sol.x,fun,step=128,rtol=1e-11,atol=1e-13)
            resid=float(np.linalg.norm(final-sol.x,np.inf))
            if resid>1e-7 or np.linalg.norm(sol.x[:2])<.03 or rmax>=1:continue
            if any(np.linalg.norm(sol.x-np.array(r['state']))<1e-4 for r in found):continue
            M=jacobian(sol.x,fun,step=128)
            eig=np.linalg.eigvals(M)
            r=dict(mesh_um=mesh,seed_i=ix,state=sol.x.tolist(),closure_residual=resid,
                   radius_max=rmax,spectral_radius=float(max(abs(eig))),
                   symplectic_defect=float(np.linalg.norm(M.T@J@M-J,np.inf)),
                   multipliers=[[float(v.real),float(v.imag)] for v in eig])
            found.append(r);print('Q FEM orbit',mesh,r['state'],r['spectral_radius'],flush=True)
        except Exception:pass
      out.extend(found)
      print('Q FEM searched',mesh,len(seeds),'found',len(found),flush=True)
      (OUT/'fem_q_orbit_search.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
