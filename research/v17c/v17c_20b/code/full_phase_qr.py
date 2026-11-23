from pathlib import Path
import json,math
import numpy as np
import attractor_search as a
from variational_qr import full_phase_lyapunov_qr

OUT=a.OUT
def main():
    cfg=a.read_json(OUT/'candidate_parameters.json');field=a.load_field();rows=[]
    for case in cfg['candidates']:
        tag=('refine12_240k_corrected_drag' if case['legacy_candidate']=='refine12_sobol17' else f"{case['legacy_candidate']}_50000_common_corrected")
        z=np.load(OUT/f'{tag}_trajectory.npz');state=z['states'][-1];n=int(z['periods']);p,q=a.arrays(case,field)
        phase=(z['phase']+np.array([1.0,float(p[2])])*2*math.pi*n)%(2*math.pi)
        result=full_phase_lyapunov_qr(case,state,phase,periods=500,steps=32,qr_periods=10)
        rows.append(result);print(case['legacy_candidate'],result['rows'][-1]['exponents_per_tau'][-2:],flush=True)
    a.write_json(OUT/'full_phase_lyapunov_500p.json',{'rows':rows,'purpose':'Extended autonomous phase-space tangent system; interpreted separately from conditional mechanical spectra.'})
if __name__=='__main__':main()
