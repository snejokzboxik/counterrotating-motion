"""Finite-length 3-D tetrahedral FEM for the V17C.5 24-electrode cross-section.

The solver uses first-order tetrahedra for Laplace's equation. RF fields are
solved as two independent complex Dirichlet bases. The DC basis is a realizable
endcap configuration: both grounded side screen and RF rods are at 0 V DC;
the two end plates are at +1 V. This is not the ideal quadrupole boundary.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import gmsh
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
from scipy.spatial import cKDTree
import pyamg

HERE=Path(__file__).resolve().parent
CFG_PATH=HERE.parent.parent/"v17c_5"/"input"/"config.json"

def assemble(points,tets):
    xyz=points[tets]
    edge=np.transpose(xyz[:,1:,:]-xyz[:,:1,:],(0,2,1))
    inv=np.linalg.inv(edge)
    vol=np.abs(np.linalg.det(edge))/6
    grad=np.empty((len(tets),4,3),float)
    grad[:,1:,:]=inv
    grad[:,0,:]=-inv.sum(axis=1)
    ke=vol[:,None,None]*np.einsum('tki,tji->tkj',grad,grad)
    ii=np.repeat(tets,4,axis=1).reshape(-1)
    jj=np.tile(tets,(1,4)).reshape(-1)
    kk=ke.reshape(-1)
    K=coo_matrix((kk,(ii,jj)),shape=(len(points),len(points))).tocsc()
    K.sum_duplicates()
    return K,grad,vol,inv

def build_mesh(hmax,hmin,rod_length_R=8.,screen_length_R=10.):
    D=1.; a=.05; B=1.10; R=D-a
    rodL=rod_length_R*R; screenL=screen_length_R*R
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",1)
    gmsh.option.setNumber("Mesh.MshFileVersion",4.1)
    gmsh.option.setNumber("Mesh.MeshSizeMax",hmax)
    gmsh.option.setNumber("Mesh.MeshSizeMin",hmin)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",16)
    gmsh.option.setNumber("Mesh.Algorithm3D",1)
    gmsh.option.setNumber("Mesh.Optimize",1)
    gmsh.model.add("V17C9_finite_24_rod")
    occ=gmsh.model.occ
    screen=occ.addCylinder(0,0,-screenL/2,0,0,screenL,B)
    rods=[]; theta=[]
    for j in range(24):
        th=2*math.pi*j/24
        theta.append(th)
        x,y=D*math.cos(th),D*math.sin(th)
        rods.append(occ.addCylinder(x,y,-rodL/2,0,0,rodL,a))
    vac,mapout=occ.cut([(3,screen)],[(3,t) for t in rods],removeObject=True,removeTool=False)
    occ.synchronize()
    volumes=gmsh.model.getEntities(3)
    vac_tags=[tag for dim,tag in vac if dim==3]
    if len(vac_tags)!=1:
        gmsh.finalize(); raise RuntimeError(f"Expected one vacuum region; got {vac_tags}")
    vacuum=vac_tags[0]
    # Tool volume tags remain valid after a cut with removeTool=False.
    rod_entities=[]
    for j,tag in enumerate(rods):
        if not any(dim==3 and t==tag for dim,t in volumes):
            # OCC may renumber retained tool volumes; select by center/radius bbox.
            x,y=D*math.cos(theta[j]),D*math.sin(theta[j])
            matches=[]
            for dim,t in volumes:
                if dim!=3 or t==vacuum: continue
                bb=gmsh.model.getBoundingBox(3,t)
                if abs((bb[0]+bb[3])/2-x)<1e-5 and abs((bb[1]+bb[4])/2-y)<1e-5:
                    matches.append(t)
            if len(matches)!=1:
                gmsh.finalize(); raise RuntimeError(f"Could not identify electrode solid {j}: {matches}")
            rod_entities.append(matches[0])
        else: rod_entities.append(tag)
    rod_surfaces=[]
    for tag in rod_entities:
        bnd=gmsh.model.getBoundary([(3,tag)],combined=False,oriented=False,recursive=False)
        rod_surfaces.append(sorted(t for dim,t in bnd if dim==2))
    # Remove conductor volumes from the meshed domain while preserving their
    # shared CAD boundary surfaces; only vacuum tetrahedra then enter K.
    occ.remove([(3,t) for t in rod_entities],recursive=False)
    occ.synchronize()
    vacuum_surfaces=sorted(t for dim,t in gmsh.model.getBoundary([(3,vacuum)],combined=False,oriented=False) if dim==2)
    # Surfaces on each conductor are also surfaces of the vacuum region.
    rod_face_set=set(t for group in rod_surfaces for t in group)
    if not rod_face_set.issubset(set(vacuum_surfaces)):
        gmsh.finalize(); raise RuntimeError("Electrode/vacuum interface surfaces are not conformal")
    endcaps=[]
    for s in vacuum_surfaces:
        if s in rod_face_set: continue
        bb=gmsh.model.getBoundingBox(2,s)
        if abs(bb[5]-bb[2])<1e-5 and (abs(bb[2]-screenL/2)<1e-5 or abs(bb[2]+screenL/2)<1e-5):
            endcaps.append(s)
    if len(endcaps)!=2:
        detail={int(s):[float(x) for x in gmsh.model.getBoundingBox(2,s)] for s in vacuum_surfaces if s not in rod_face_set}
        gmsh.finalize(); raise RuntimeError(f"Expected two endcap surfaces, got {endcaps}; remaining={detail}")
    # Refine near electrode metal/vacuum interfaces; retain a coarser central bulk.
    field=gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(field,"FacesList",list(rod_face_set))
    gmsh.model.mesh.field.setNumber(field,"Sampling",1000)
    threshold=gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold,"InField",field)
    gmsh.model.mesh.field.setNumber(threshold,"SizeMin",hmin)
    gmsh.model.mesh.field.setNumber(threshold,"SizeMax",hmax)
    gmsh.model.mesh.field.setNumber(threshold,"DistMin",.07)
    gmsh.model.mesh.field.setNumber(threshold,"DistMax",.30)
    gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    try:
        gmsh.model.mesh.generate(3)
    except Exception:
        error=gmsh.logger.getLastError()
        gmsh.finalize()
        raise RuntimeError(f"Gmsh 3D mesh generation failed: {error}")
    tags,coords,_=gmsh.model.mesh.getNodes()
    points=coords.reshape(-1,3)
    map_tags={int(t):i for i,t in enumerate(tags)}
    etags,conn=gmsh.model.mesh.getElementsByType(4,vacuum)
    if not len(etags):
        gmsh.finalize(); raise RuntimeError("No tetrahedra generated for vacuum region")
    tets=np.fromiter((map_tags[int(t)] for t in conn),dtype=np.int64,count=len(conn)).reshape(-1,4)
    def nodes_of(surflist):
        merged=set()
        for s in surflist:
            nd,_,_=gmsh.model.mesh.getNodes(2,s,includeBoundary=True)
            merged.update(map_tags[int(t)] for t in nd)
        return np.asarray(sorted(merged),dtype=np.int64)
    electrode_nodes=[nodes_of(g) for g in rod_surfaces]
    cap_nodes=nodes_of(endcaps)
    all_bnd=nodes_of(vacuum_surfaces)
    # Boundary values are assigned by a single-valued Dirichlet map.
    bmask=np.zeros(len(points),bool); bmask[all_bnd]=True
    # Check no electrode nodes overlap with endcap metal nodes.
    if set(map(int,cap_nodes)) & set(map(int,np.concatenate(electrode_nodes))):
        gmsh.finalize(); raise RuntimeError("Electrodes touch grounded/endcap boundary nodes")
    gmsh.finalize()
    return points,tets,theta,electrode_nodes,cap_nodes,bmask,{
        "a_over_D":a/D,"B_over_D":B/D,"D":D,"R":R,"rod_length_over_R":rod_length_R,
        "screen_length_over_R":screen_length_R,"hmax":hmax,"hmin":hmin,
        "n_nodes":len(points),"n_tetrahedra":len(tets),"n_electrodes":24}

def boundary_vector(points,electrode_nodes,cap_nodes,bmask,theta,k,dc=False):
    u=np.zeros(len(points),np.complex128)
    if dc:
        u[cap_nodes]=1.0
    else:
        for j,nodes in enumerate(electrode_nodes):
            u[nodes]=np.exp(1j*k*theta[j])
    return u

def solve_bases(K,bmask,boundary_vectors):
    bc=np.flatnonzero(bmask); interior=np.flatnonzero(~bmask)
    Kii=K[interior][:,interior].tocsc()
    Kib=K[interior][:,bc]
    # Algebraic multigrid preconditioning keeps peak memory below sparse-LU
    # fill-in for the finite-electrode mesh.
    ml=pyamg.smoothed_aggregation_solver(Kii,max_coarse=24)
    M=ml.aspreconditioner(cycle="V")
    print(f"AMG hierarchy ready: {Kii.shape[0]} unknowns, levels={len(ml.levels)}",flush=True)
    ub=np.stack(boundary_vectors,axis=1)
    rhs=-(Kib@ub[bc])
    ui=np.zeros_like(rhs,dtype=np.complex128); diagnostics=[]
    for col in range(rhs.shape[1]):
        for label,b in (("real",rhs[:,col].real),("imag",rhs[:,col].imag)):
            count=[0]
            def cb(_): count[0]+=1
            x,info=cg(Kii,b,rtol=2e-9,atol=1e-12,maxiter=3000,M=M,callback=cb)
            rel=float(np.linalg.norm(Kii@x-b)/max(np.linalg.norm(b),1e-300))
            diagnostics.append({"basis":col,"component":label,"iterations":count[0],"info":int(info),"relative_residual":rel})
            if info!=0 or rel>1e-7: raise RuntimeError(f"AMG-CG not converged: {diagnostics[-1]}")
            if label=="real": ui[:,col]=x
            else: ui[:,col]+=1j*x
        print(f"basis {col+1}/{rhs.shape[1]} solve complete",flush=True)
    u=np.zeros((len(bmask),len(boundary_vectors)),np.complex128)
    u[bc,:]=ub[bc,:];u[interior,:]=ui
    return u,diagnostics

def interp(points,tets,inv,values,query):
    cent=points[tets].mean(axis=1); tree=cKDTree(cent)
    _,candidates=tree.query(query,k=min(24,len(tets)))
    out=np.empty(len(query),complex); xyz=points[tets]
    for i,p in enumerate(query):
        found=False
        for ti in np.atleast_1d(candidates[i]):
            xi=inv[ti]@(p-xyz[ti,0]); bary=np.r_[1.-xi.sum(),xi]
            if bary.min()>=-2e-8 and bary.max()<=1.+2e-8:
                out[i]=bary@values[tets[ti]];found=True;break
        if not found: out[i]=np.nan+1j*np.nan
    return out

def coeff_spectra(points,tets,inv,u1,u2,R):
    cfg=json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))
    k=np.asarray(cfg["k"],int); fam=np.asarray(cfg["family"],int); c=np.asarray(cfg["coeff"],float); c=c[:,0]+1j*c[:,1]
    theta=np.arange(256)*2*np.pi/256
    rows=[]; samples=[]
    for rfrac in (.15,.25,.35,.45,.55,.65,.75,.82):
        q=np.column_stack((rfrac*R*np.cos(theta),rfrac*R*np.sin(theta),np.zeros(len(theta))))
        v1=interp(points,tets,inv,u1,q);v2=interp(points,tets,inv,u2,q)
        samples.append((rfrac,v1,v2))
        for family,u in ((0,v1),(1,v2)):
            for mode in k[fam==family]:
                fh=np.mean(u*np.exp(-1j*mode*theta))
                lead=2 if family==0 else -4
                expected=c[np.where((fam==family)&(k==mode))[0][0]]/c[np.where((fam==family)&(k==lead))[0][0]]
                leadhat=np.mean(u*np.exp(-1j*lead*theta))
                ratio=(fh/leadhat)*(rfrac*R)**(abs(lead)-abs(mode))
                rows.append({"r_over_R":rfrac,"family":family,"mode":int(mode),
                    "complex_coefficient_ratio_recovered": [float(ratio.real),float(ratio.imag)],
                    "magnitude_ratio_recovered":float(abs(ratio)),"phase_deg_recovered":float(np.degrees(np.angle(ratio))),
                    "MFS_coefficient_ratio": [float(expected.real),float(expected.imag)],
                    "magnitude_ratio_MFS":float(abs(expected)),
                    "relative_magnitude_error":float(abs(abs(ratio)-abs(expected))/max(abs(expected),1e-30))})
    return rows,samples

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--hmax",type=float,default=.18);ap.add_argument("--hmin",type=float,default=.03);ap.add_argument("--tag",default="coarse");a=ap.parse_args()
    tic=time.perf_counter()
    pts,tets,theta,elec,cap,bmask,meshmeta=build_mesh(a.hmax,a.hmin)
    print(f"mesh ready: {meshmeta['n_nodes']} nodes, {meshmeta['n_tetrahedra']} tetrahedra",flush=True)
    K,grad,vol,inv=assemble(pts,tets)
    print(f"stiffness assembled: {K.shape}, nnz={K.nnz}",flush=True)
    bases,solver_diagnostics=solve_bases(K,bmask,[boundary_vector(pts,elec,cap,bmask,theta,+2),
                                boundary_vector(pts,elec,cap,bmask,theta,-4),
                                boundary_vector(pts,elec,cap,bmask,theta,0,dc=True)])
    u1,u2,udc=bases[:,0],bases[:,1],bases[:,2]
    gc1=np.einsum("ti,tij->tj",u1[tets],grad);gc2=np.einsum("ti,tij->tj",u2[tets],grad)
    e1=np.linalg.norm(gc1,axis=1);e2=np.linalg.norm(gc2,axis=1)
    spectra,samples=coeff_spectra(pts,tets,inv,u1,u2,.95)
    # DC-axis/near-axis finite differences give a locally axisymmetric quadrupole coefficient.
    zvals=np.array([.05,.08,.11,.14,.17,.20,.24,.28])* .95
    zq=np.column_stack((np.zeros(len(zvals)),np.zeros(len(zvals)),zvals))
    dcaxis=interp(pts,tets,inv,udc,zq)
    zcoef=np.polyfit(zvals**2,dcaxis.real-dcaxis.real[0],1)[0]
    # Axisymmetric harmonicity predicts radial curvature = -1/2 axial curvature.
    xvals=np.array([.05,.08,.11,.14,.17,.20,.24,.28])* .95
    xq=np.column_stack((xvals,np.zeros(len(xvals)),np.zeros(len(xvals))))
    dcx=interp(pts,tets,inv,udc,xq)
    rcoef=np.polyfit(xvals**2,dcx.real-dcx.real[0],1)[0]
    out={"model":"3D Laplace FEM; first-order tetrahedra; finite rods and grounded cylindrical shield",
      "geometry":meshmeta,"field_boundary_convention":"V_j phasor exp(i*k*theta_j), time factor exp(-i*omega*t); k=+2 and -4",
      "MFS_input_file":str(CFG_PATH),"MFS_k":json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))["k"],
      "MFS_coefficients":json.loads(CFG_PATH.read_text(encoding="utf-8-sig"))["coeff"],
      "RF_complex_spectra_by_radius":spectra,
      "solver":"SciPy conjugate gradient, pyamg smoothed-aggregation V-cycle preconditioner; separate real/imaginary RHS",
      "solver_residuals":solver_diagnostics,
      "field_magnitude_max_in_vacuum_per_unit_electrode_amplitude":{"RF_k_plus2":float(e1.max()),"RF_k_minus4":float(e2.max())},
      "DC_boundary":"both finite grounded-shield end plates +1 V; side shield and all 24 RF rods 0 V DC",
      "DC_local_quadratic_fit":{"coefficient_phi_z2_per_V":float(zcoef),"coefficient_phi_x2_per_V":float(rcoef),
          "laplace_ratio_radial_over_axial":float(rcoef/zcoef) if zcoef else None,
          "ideal_axisymmetric_quadrupole_ratio":-0.5,"fit_rms_z":float(np.sqrt(np.mean((dcaxis.real-np.polyval(np.polyfit(zvals**2,dcaxis.real,2),zvals**2))**2)))},
      "runtime_s":time.perf_counter()-tic,
      "limitations":["Finite rods terminate before grounded end caps; model is 3D but not an end-electrode design study.",
        "Linear tetrahedra; peak surface field is estimated from adjacent element gradients and under-resolves edge singularities.",
        "No breakdown claim; sharp ideal edges make the local FEM peak mesh-sensitive.",
        "This field solve does not replace direct trajectory integration in the FEM field."]}
    outpath=HERE/f"fem3d_{a.tag}.json"
    (HERE/f"fem3d_{a.tag}.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    np.savez_compressed(HERE/f"fem3d_{a.tag}_compact.npz",radii_R=np.array([x[0] for x in samples]),
      theta=np.arange(256)*2*np.pi/256,rf1=np.array([x[1] for x in samples]),rf2=np.array([x[2] for x in samples]),
      z_dc=zvals,phi_dc_axis=dcaxis,x_dc=xvals,phi_dc_radial=dcx)
    print(json.dumps({"out":str(outpath),"mesh":meshmeta,"max_E":out["field_magnitude_max_in_vacuum_per_unit_electrode_amplitude"],
      "dc_fit":out["DC_local_quadratic_fit"],"runtime_s":out["runtime_s"]},indent=2))

if __name__=="__main__":main()
