import json
from pathlib import Path
import numpy as np
from collective_rings import (intraring_energy_direct,intraring_energy_formula,
    radial_coulomb_force_direct,radial_coulomb_force_formula,
    vertical_laplacian,vertical_laplacian_spectrum_formula)

out=[]
for n in (2,3,6,8,12,16,24,32):
    eig_direct=np.linalg.eigvalsh(vertical_laplacian(n,1.0))
    eig_formula=np.sort(vertical_laplacian_spectrum_formula(n,1.0))
    out.append({
      "N":n,"R_over_Rref":1.0,"Q":1.0,"mu":1.0,"kappa":1.0,
      "intraring_energy_direct":intraring_energy_direct(n,1.0,1.0),
      "intraring_energy_formula":intraring_energy_formula(n,1.0,1.0),
      "radial_force_per_particle_direct":radial_coulomb_force_direct(n,1.0,1.0),
      "radial_force_per_particle_formula":radial_coulomb_force_formula(n,1.0,1.0),
      "vertical_laplacian_lambda_max":float(eig_direct[-1]),
      "max_eigenvalue_fourier_error":float(np.max(np.abs(eig_direct-eig_formula))),
      "beta2_critical_for_quadrupole_Kz_2beta":float(eig_formula[-1]/2)
    })
p=Path(__file__).resolve().parents[1]/"results/collective_ring_checks.json"
p.write_text(json.dumps({"status":"analytic/direct finite-sum checks; fixed R=Q=mu=kappa=1, not an optimized trap",
                         "rows":out},indent=2)+"\n",encoding="utf-8")
print(p)
print(json.dumps(out,indent=2))
