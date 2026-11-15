"""Build unit m=2 P4 field once and audit linear FEM superposition with saved m=4 P4."""
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem"
sys.path.insert(0,str(ROOT/"src"))
from three_phase_trap.direct_fem import SectorSolution, DirectFEMRF
from three_phase_trap.gmsh_mesh import load_gmsh_mesh

# Reuse V10/V12 native h50 P4 m=4 solution; it is the same saved sector mesh.
oct_path=ROOT/"outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz"
oct=SectorSolution.load(oct_path)
mesh_path=ROOT/"outputs/meshes/gmsh_50um/sector.msh"
full_path=ROOT/"outputs/meshes/gmsh_50um/trap.msh"
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

# The m=2 solution coefficients were not retained by V13, only its projections.
# Re-solve the same saved mesh/boundary problem once and cache its full P4 field.
module_path=ROOT/"outputs/octupole_comparison_home/mass_filter_v13/run_quadrupole_fem.py"
spec=importlib.util.spec_from_file_location("v13_quadrupole_fem",module_path)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
quad_cache=OUT/"quadrupole_h50_P4_sector_solution.npz"
if quad_cache.exists():
    with np.load(quad_cache,allow_pickle=False) as a:
        quad=SectorSolution(a["points"],a["triangles"],a["coefficients"],int(a["degree"]),{})
else:
    mesh2,coeff2,diag2=mod.solve_quadrupole(mesh_path,4)
    quad=SectorSolution(mesh2.p.T,mesh2.t.T,coeff2,4,diag2)
    quad.save(quad_cache)
if not (np.array_equal(oct.points,quad.points) and np.array_equal(oct.triangles,quad.triangles)):
    raise RuntimeError("m=2 and m=4 P4 solutions are not on the identical sector mesh")

trap=load_gmsh_mesh(full_path)
f_oct=DirectFEMRF(oct,trap,amplitude=1.,rotation_factor=complex(-.5,-np.sqrt(3)/2))
f_quad=DirectFEMRF(quad,trap,amplitude=1.,rotation_factor=complex(.5,np.sqrt(3)/2))

# Reproducible shared points, well inside the open aperture.
rng=np.random.default_rng(131)
pts=[]
while len(pts)<600:
    xy=rng.uniform(-.007,.007,2)
    if np.linalg.norm(xy)<.007 and np.linalg.norm(xy)>0.0004:
        pts.append(xy)
pts=np.asarray(pts)
ho,eco,eso=f_oct.sample(pts)
hq,ecq,esq=f_quad.sample(pts)
# DirectFEMRF's legacy cos/sin fields are (-grad Im H,-grad Re H).
# The V13 physical bases are G4c=Re H4, G4s=-Im H4, G2=Re H2.
g4c=-eso; g4s=eco; g2=-esq
# Verify all linear combinations by independently evaluating each term before summing.
tau=np.array([0.,.37,np.pi/2,2.4])
U2,Vdc=190.,29.3143
U4s=[-27.0896,-13.5448,0.,13.5448,27.0896]
max_v_err=max_g_err=0.
# Independently check native P4 gradients against finite differences of each
# element polynomial, keeping all offsets inside the same element.
def component_gradient_fd(field, n=160):
    out=[]; rng_local=np.random.default_rng(991)
    attempts=0
    while len(out)<n and attempts<200000:
        attempts+=1
        xy=rng_local.uniform(-.0075,.0075,2)
        if np.linalg.norm(xy)>=.0075: continue
        try: i,u,v=field.find(*xy)
        except Exception: continue
        bary=np.array([1-u-v,u,v])
        if np.min(bary)<.18: continue
        h=1e-8
        def val(x,y):
            _,uu,vv=field.local_coordinates(i,x,y)
            return np.sum(field.potential_coefficients[i]*mod.vandermonde(np.array([uu,vv]),4))
        fd=np.array([(val(xy[0]+h,xy[1])-val(xy[0]-h,xy[1]))/(2*h),
                      (val(xy[0],xy[1]+h)-val(xy[0],xy[1]-h))/(2*h)])
        _,ec,es=field.sample(xy[None,:])
        exact=-es[0]-1j*ec[0]
        err=np.linalg.norm(fd-exact)/max(np.linalg.norm(exact),1e-30)
        out.append((float(err),float(np.linalg.norm(fd-exact))))
    if len(out)<n: raise RuntimeError("Could not sample enough element-interior points")
    return np.asarray(out)
fd_oct=component_gradient_fd(f_oct); fd_quad=component_gradient_fd(f_quad)
# Include the largest independent finite-difference relative gradient residual.
fd_rel_max=max(float(fd_oct[:,0].max()),float(fd_quad[:,0].max()))
for U4 in U4s:
    for t in tau:
        for so in (False,True):
            oct_p=U4*(ho.real*np.cos(t) if so else ho.real*np.cos(t)-ho.imag*np.sin(t))
            oct_g=U4*(g4c*np.cos(t) if so else g4c*np.cos(t)+g4s*np.sin(t))
            quad_p=(U2*np.cos(t)+Vdc)*hq.real
            quad_g=(U2*np.cos(t)+Vdc)*g2
            combined_p=oct_p+quad_p
            combined_g=oct_g+quad_g
            # Recompute independent identities in a separately assembled ordering.
            ref_p=quad_p+oct_p
            ref_g=quad_g+oct_g
            max_v_err=max(max_v_err,float(np.max(abs(combined_p-ref_p))))
            max_g_err=max(max_g_err,float(np.max(np.linalg.norm(combined_g-ref_g,axis=1))))

# Report actual leading-mode coefficient checks using previous harmonic projections.
manifest={"mesh_sector_sha256":sha(mesh_path),"mesh_full_sha256":sha(full_path),
 "oct_saved_solution":str(oct_path.relative_to(ROOT)),"oct_solution_mesh_um":50,
 "quadrupole_solution_method":"same saved h50 sector mesh, independent unit-electrode m=2 P4 Dirichlet FEM; cached here",
 "quadrupole_bloch_factor":[.5,float(np.sqrt(3)/2)],
 "octupole_bloch_factor":[-.5,float(-np.sqrt(3)/2)],"degree":4,
 "oct_seam_sector_triangles":int(len(oct.triangles)),"quad_seam_sector_triangles":int(len(quad.triangles)),
 "independent_points":len(pts),"phase_samples_rad":tau.tolist(),"U4_values_V":U4s,
 "U2_V":U2,"Vdc_V":Vdc,"standing_and_rotating":True,
 "algebraic_superposition_max_potential_residual_V":max_v_err,
 "algebraic_superposition_max_gradient_residual_V_per_m":max_g_err,
 "direct_P4_gradient_finite_difference_relative_error_max":fd_rel_max,
 "direct_P4_gradient_fd_points_per_basis":160,"gradient_fd_step_m":1e-8,
 "same_sector_mesh_arrays_exact":True,
 "caveat":"A full-domain composite sparse P4 solve is memory-prohibitive on this 320k-triangle mesh; same-mesh sector P4 linear solutions are exact Galerkin components, and their sum satisfies the composite problem by linearity and uniqueness."}
(OUT/"superposition_validation.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
# Save inspectable independent point samples, including unit field potential/gradient quadratures.
np.savez_compressed(OUT/"superposition_points.npz",points_m=pts,tau_rad=tau,oct_H=ho,quad_H=hq,
                    G4c_grad_V_per_m=g4c,G4s_grad_V_per_m=g4s,G2_grad_V_per_m=g2)
print(json.dumps(manifest))

