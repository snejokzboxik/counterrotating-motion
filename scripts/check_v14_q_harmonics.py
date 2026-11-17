"""Test whether measured m=10 leakage in the FEM m=2 basis explains Q off-axis orbit."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_ideal_pilot import OUT,ALPHA,BETA
from run_v14_fem_orbits import J

HARM=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/quadrupole_h50_P4_harmonics.csv'

def main():
    with HARM.open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    c2=float(next(r['normalized_cos_coefficient'] for r in rows if r['radius_mm']=='6' and r['order_m']=='2'))
    c10=float(next(r['normalized_cos_coefficient'] for r in rows if r['radius_mm']=='6' and r['order_m']=='10'))
    gamma=c10/c2
    femq=json.loads((OUT/'fem_q_orbit_search.json').read_text())
    z0=np.array(next(r['state'] for r in femq if r['mesh_um']==50))
    out=[]
    for g in (0.,gamma):
      def fun(t,state):
        z=complex(state[0],state[1]);q=ALPHA+BETA*math.cos(t)
        w=q*(2*z+10*g*z**9)
        return np.array([state[2],state[3],-w.real,w.imag])
      def flow(z,step=128):
        sol=solve_ivp(fun,(0,2*math.pi),z,method='DOP853',rtol=1e-11,atol=1e-13,max_step=2*math.pi/step)
        if not sol.success:raise RuntimeError(sol.message)
        return sol.y[:,-1],float(np.max(np.hypot(sol.y[0],sol.y[1])))
      def residual(z):return flow(z)[0]-z
      sol=root(residual,z0,method='hybr',options={'xtol':1e-10,'maxfev':100})
      final,rmax=flow(sol.x)
      M=np.stack([(flow(sol.x+np.eye(4)[j]*1e-6)[0]-flow(sol.x-np.eye(4)[j]*1e-6)[0])/(2e-6) for j in range(4)],axis=1)
      eig=np.linalg.eigvals(M)
      out.append(dict(gamma=g,c2=c2,c10=c10,root_success=bool(sol.success),state=sol.x.tolist(),
                      closure_residual=float(np.linalg.norm(final-sol.x,np.inf)),max_radius=rmax,
                      spectral_radius=float(max(abs(eig))),symplectic_defect=float(np.linalg.norm(M.T@J@M-J,np.inf)),
                      multipliers=[[float(v.real),float(v.imag)] for v in eig]))
      print('gamma',g,'state',sol.x,'resid',out[-1]['closure_residual'],'rho',out[-1]['spectral_radius'])
    (OUT/'q_m10_harmonic_control.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
