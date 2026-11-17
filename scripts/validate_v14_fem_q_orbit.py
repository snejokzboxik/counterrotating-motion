"""Finite-difference and mesh convergence of full-FEM Q off-axis orbit monodromy."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_v13_3_fem_controls as fem
from run_v14_fem_orbits import OUT,make_rhs,flow,jacobian,J
from analyze_v14_ideal import write_csv

def main():
    orbit=json.loads((OUT/'fem_q_orbit_search.json').read_text())
    rows=[]
    for mesh in (50,35):
        fem.initialize(mesh);f=make_rhs('Q',0.,0.)
        z=np.array(next(r['state'] for r in orbit if r['mesh_um']==mesh))
        matrices={}
        for eps in (1e-5,2e-6,1e-6):
            M=jacobian(z,f,eps,step=128);matrices[eps]=M
            eig=np.linalg.eigvals(M)
            rows.append(dict(mesh_um=mesh,finite_difference_eps=eps,
                             symplectic_defect_inf=float(np.linalg.norm(M.T@J@M-J,np.inf)),
                             spectral_radius=float(max(abs(eig))),
                             eigenvalue_moduli=json.dumps([float(abs(v)) for v in eig]),
                             deviation_from_eps_1e5_inf=float(np.linalg.norm(M-matrices[1e-5],np.inf))))
            print(mesh,eps,rows[-1]['symplectic_defect_inf'],rows[-1]['spectral_radius'],flush=True)
    write_csv(OUT/'fem_q_orbit_monodromy_refinement.csv',rows)

if __name__=='__main__':main()
