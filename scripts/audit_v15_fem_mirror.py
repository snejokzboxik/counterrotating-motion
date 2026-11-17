"""Check saved h50/h35 P4 m=2/m=4 field symmetries under y reflection."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"scripts"),str(ROOT/"src")]
import run_v13_3_fem_controls as fem
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v15"

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    rows=[]; hashes={}
    rng=np.random.default_rng(150015)
    pts=[]
    while len(pts)<300:
      xy=rng.uniform(-.0075,.0075,2)
      if np.linalg.norm(xy)<.0075 and np.linalg.norm(xy)>.0003: pts.append(xy)
    pts=np.asarray(pts)
    for mesh in (50,35):
      fem.initialize(mesh)
      if mesh==50:
        p4=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
        p2=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz"
        msh=ROOT/"outputs/meshes/gmsh_50um/trap.msh"
      else:
        p4=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz"
        p2=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/fem/quadrupole_h35_P4_sector_solution.npz"
        msh=ROOT/"outputs/meshes/gmsh_35um/trap.msh"
      hashes.update({str(q.relative_to(ROOT)):sha(q) for q in (p4,p2,msh)})
      errors={"G2_x_even":[],"G2_y_odd":[],"G4c_x_even":[],"G4c_y_odd":[],"G4s_x_odd":[],"G4s_y_even":[]}
      for x,y in pts:
        qx,qy=fem._grad2(x,y); qxm,qym=fem._grad2(x,-y)
        cpx,cpy,spx,spy=fem._grad4(x,y); cmx,cmy,smx,smy=fem._grad4(x,-y)
        for key,diff,scale in (
          ("G2_x_even",qxm-qx,max(abs(qxm),abs(qx),1e-20)),("G2_y_odd",qym+qy,max(abs(qym),abs(qy),1e-20)),
          ("G4c_x_even",cmx-cpx,max(abs(cmx),abs(cpx),1e-20)),("G4c_y_odd",cmy+cpy,max(abs(cmy),abs(cpy),1e-20)),
          ("G4s_x_odd",smx+spx,max(abs(smx),abs(spx),1e-20)),("G4s_y_even",smy-spy,max(abs(smy),abs(spy),1e-20))):
            errors[key].append(abs(float(diff))/scale)
      rows.append({"mesh_um":mesh,"n_points":len(pts),"max_relative_symmetry_error":max(max(v) for v in errors.values()),
                   "rms_relative_symmetry_error":math.sqrt(np.mean([e*e for vals in errors.values() for e in vals])),
                   "component_max_relative_errors":{k:max(v) for k,v in errors.items()},
                   "reflection":"(x,y)->(x,-y); G2/G4c even; G4s odd in potential; derivative parity checked"})
    result={"geometry":"saved 12 electrodes at theta_j=2pi*j/12","fields":"saved P4 unit m=2 and m=4 bases, no new FEM solve",
      "sample_seed":150015,"radius_m":.0075,"n_points":len(pts),"meshes":rows,"field_hashes":hashes,
      "interpretation":"potential/gradient mirror symmetry supports an exact reflected-trajectory identity; this audit does not test trajectory convergence or electrode circuit impedance"}
    out=OUT/"fem_mirror_symmetry.json";out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
