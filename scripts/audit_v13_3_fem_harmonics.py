"""Project saved h35/h50 P4 unit fields onto radial Fourier harmonics."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas"
R=.008;RADII=(.20,.30,.40,.60,.80,.90);NTHETA=1440

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,rows):
    if not rows:return
    names=[]
    for row in rows:names.extend(k for k in row if k not in names)
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--meshes",nargs="+",type=int,choices=(35,50),default=(35,50));args=parser.parse_args()
    sys.path.insert(0,str(ROOT/"src"))
    from three_phase_trap.direct_fem import SectorSolution,DirectFEMRF
    from three_phase_trap.gmsh_mesh import load_gmsh_mesh
    out=[];hashes={}
    for h in args.meshes:
        mesh_path=ROOT/f"outputs/meshes/gmsh_{h}um/trap.msh"
        if h==50:
            f4_path=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
            f2_path=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz"
        else:
            f4_path=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz"
            f2_path=BASE/"fem/quadrupole_h35_P4_sector_solution.npz"
        mesh=load_gmsh_mesh(mesh_path);s4=SectorSolution.load(f4_path);s2=SectorSolution.load(f2_path)
        if not (s4.points.shape==s2.points.shape and s4.triangles.shape==s2.triangles.shape and
                np.array_equal(s4.points,s2.points) and np.array_equal(s4.triangles,s2.triangles)):
            raise RuntimeError(f"h{h} m2/m4 P4 sector mesh arrays differ")
        f4=DirectFEMRF(s4,mesh,amplitude=1.,rotation_factor=complex(-.5,-math.sqrt(3)/2))
        f2=DirectFEMRF(s2,mesh,amplitude=1.,rotation_factor=complex(.5,math.sqrt(3)/2))
        hashes.update({str(p.relative_to(ROOT)):sha(p) for p in (mesh_path,f4_path,f2_path)})
        th=np.linspace(0.,2*math.pi,NTHETA,endpoint=False)
        for rho in RADII:
            xy=np.column_stack((R*rho*np.cos(th),R*rho*np.sin(th)))
            phi4=f4.sample(xy)[0];phi2=f2.sample(xy)[0]
            for field,values,leading in (("m4",phi4,4),("m2",phi2,2)):
                # FEM convention: G4c=Re(H4), G4s=-Im(H4); G2=Re(H2).
                channels={"cos":values.real}
                if field=="m4":channels["sin"]=-values.imag
                coeffs={}
                for order in range(1,13):
                    scale=rho**order
                    ac=2*float(np.mean(values.real*np.cos(order*th)))/scale
                    bs=2*float(np.mean(values.real*np.sin(order*th)))/scale
                    if field=="m4":
                        acs=2*float(np.mean((-values.imag)*np.cos(order*th)))/scale
                        bss=2*float(np.mean((-values.imag)*np.sin(order*th)))/scale
                    else:acs=bss=math.nan
                    amp=math.hypot(ac,bs)
                    coeffs[order]=amp
                    out.append({"mesh_um":h,"field":field,"rho":rho,"fourier_order":order,
                                "cos_coefficient_normalized":ac,"sin_coefficient_normalized":bs,
                                "octupole_sine_cos_coefficient_normalized":acs,
                                "octupole_sine_sin_coefficient_normalized":bss,
                                "harmonic_amplitude_normalized":amp,
                                "relative_potential_to_leading":math.hypot(ac,bs)*rho**order/
                                    max(1e-300,math.hypot(*[2*float(np.mean(channels["cos"]*np.cos(leading*th)))/rho**leading,
                                                               2*float(np.mean(channels["cos"]*np.sin(leading*th)))/rho**leading])*rho**leading),
                                "phase_rad":math.atan2(bs,ac),"normalization":"2< G cos(m theta)>/rho^m; matching sine projection"})
    out_path=BASE/"fem_harmonic_projection.csv";write(out_path,out)
    summary=[]
    for h in args.meshes:
        for field,n in (("m2",2),("m4",4)):
            rows=[r for r in out if r["mesh_um"]==h and r["field"]==field and r["fourier_order"]==n]
            for r in rows:
                summary.append({"mesh_um":h,"field":field,"rho":r["rho"],
                                "cos_coefficient":r["cos_coefficient_normalized"],
                                "cosine_magnitude":r["harmonic_amplitude_normalized"],
                                "phase_rad":r["phase_rad"],
                                "octupole_sine_leading_sine_coefficient":r["octupole_sine_sin_coefficient_normalized"],
                                "phase_convention":"m4 rotating sine basis is -Im(H4); m2 DC/RF basis is Re(H2)"})
    write(BASE/"fem_harmonic_coefficients_by_radius.csv",summary)
    metadata={"R_m":R,"theta_samples":NTHETA,"radii_rho":RADII,"sector_mesh_arrays_identical_m2_m4":True,
              "field_convention":{"m4_cos":"Re(H4)","m4_sin":"-Im(H4)","m2_cos":"Re(H2)"},
              "projection":"c_m(r)=1/(pi rho^m) integral_0^{2pi} G_m cos(m theta) dtheta; sine projection tested independently",
              "source_hashes":hashes,"outputs":[str(out_path.relative_to(ROOT)),str((BASE/"fem_harmonic_coefficients_by_radius.csv").relative_to(ROOT))],
              "note":"c2 and d2 use the same quasi-static unit spatial basis; equality c2=d2 follows from using that one basis for RF and DC, not from separate impedance models."}
    (BASE/"fem_harmonic_projection.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"summary":summary,"metadata":metadata},indent=2))
if __name__=="__main__":main()
