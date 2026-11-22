"""Coarse 3-D FEM cross-check for the finite 24-rod RF geometry.

Solves four real Laplace basis problems (real/imaginary parts of the k=+2
and k=-4 electrode phasors) in the finite cylindrical bore.  It does not
solve the segmented-DC electrode problem and is not an optimized full device.
Sharp rod ends are intentionally excluded from peak-field certification.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import gmsh
import meshio
import numpy as np
from scipy.sparse.linalg import splu
from skfem import Basis, MeshTet, asm
from skfem.element import ElementTetP1
from skfem.models.poisson import laplace

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"; RESULTS=ROOT/"results"
R=5e-3; D=R/.95; rod_r=.05*D; shield_r=1.10*D
L=50e-3; rod_L=48e-3
MESH_LEVELS=(0.55e-3,0.42e-3,0.32e-3)


def build_mesh(hmax: float, msh_path: Path):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal",1)
        gmsh.option.setNumber("General.Verbosity",2)
        # Resolve the 0.263 mm narrowest rod-to-shield gap with several cells.
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin",hmax/7)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax",hmax)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",1)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",1)
        gmsh.option.setNumber("Mesh.Algorithm3D",4)
        gmsh.model.add("finite_24rod_two_tone")
        occ=gmsh.model.occ
        cavity=occ.addCylinder(0,0,-L/2,0,0,L,shield_r)
        cutters=[]; centers=[]
        for j in range(24):
            theta=2*math.pi*j/24
            x=D*math.cos(theta); y=D*math.sin(theta)
            centers.append((x,y,theta))
            tag=occ.addCylinder(x,y,-rod_L/2,0,0,rod_L,rod_r)
            cutters.append((3,tag))
        occ.cut([(3,cavity)],cutters,removeObject=True,removeTool=True)
        occ.synchronize()
        gmsh.model.mesh.generate(3)
        gmsh.write(str(msh_path))
        nodes,coords,_=gmsh.model.mesh.getNodes()
        element_types,element_tags,element_nodes=gmsh.model.mesh.getElements(3)
        tet_nodes=None
        for typ,conn in zip(element_types,element_nodes):
            name,dim,order,num_nodes,*_=gmsh.model.mesh.getElementProperties(typ)
            if name.startswith("Tetrahedron") and num_nodes==4:
                tet_nodes=conn.reshape(-1,4); break
        if tet_nodes is None:
            raise RuntimeError("first-order tetrahedra were not generated")
        ids=np.asarray(nodes,dtype=np.int64)
        points=np.asarray(coords,dtype=float).reshape(-1,3)
        lut={int(tag):i for i,tag in enumerate(ids)}
        tets=np.vectorize(lut.__getitem__)(tet_nodes).astype(np.int32)
        return points,tets,np.asarray(centers,dtype=float)
    finally:
        gmsh.finalize()


def solve_mesh(hmax: float):
    msh=RESULTS/f"finite_24rod_rf_h{hmax*1e3:.2f}mm.msh"
    points,tets,centers=build_mesh(hmax,msh)
    mesh=MeshTet(points.T,tets.T,sort_t=False)
    basis=Basis(mesh,ElementTetP1())
    A=asm(laplace,basis).tocsc()
    xyz=mesh.p.T
    boundary=np.unique(mesh.facets[:,mesh.boundary_facets()])
    tol=max(2e-7,hmax*0.015)
    rho=np.hypot(xyz[:,0],xyz[:,1])
    shield_mask=np.abs(rho[boundary]-shield_r)<tol
    end_mask=np.abs(np.abs(xyz[boundary,2])-L/2)<tol
    grounded=np.zeros(len(xyz),dtype=bool)
    grounded[boundary[shield_mask|end_mask]]=True
    rod_id=np.full(len(xyz),-1,dtype=int)
    # Identify curved sides and planar end caps of each finite rod.
    for j,(cx,cy,theta) in enumerate(centers):
        dr=np.hypot(xyz[:,0]-cx,xyz[:,1]-cy)
        side=(np.abs(dr-rod_r)<tol)&(np.abs(xyz[:,2])<=rod_L/2+tol)
        cap=(np.abs(np.abs(xyz[:,2])-rod_L/2)<tol)&(dr<=rod_r+tol)
        rod_id[boundary[side[boundary]|cap[boundary]]]=j
    driven=rod_id>=0
    Dofs=np.unique(np.r_[boundary[grounded[boundary]],boundary[driven[boundary]]])
    if len(Dofs)<100 or np.count_nonzero(driven)<100:
        raise RuntimeError(f"boundary classification failed: {len(Dofs)} dofs; {np.count_nonzero(driven)} rod nodes")
    free=np.setdiff1d(np.arange(basis.N),Dofs,assume_unique=True)
    lu=splu(A[free][:,free].tocsc())
    # columns: Re/Im for m=+2, then Re/Im for m=-4
    vd=np.zeros((len(Dofs),4),dtype=float)
    pos={int(node):i for i,node in enumerate(Dofs)}
    dn=boundary[driven[boundary]]
    dj=rod_id[dn]
    phases=np.array([centers[j,2] for j in dj])
    for col,m in enumerate((2,-4)):
        val=np.exp(1j*m*phases)
        for node,v in zip(dn,val):
            vd[pos[int(node)],2*col]=v.real
            vd[pos[int(node)],2*col+1]=v.imag
    rhs=-A[free][:,Dofs]@vd
    U=np.zeros((basis.N,4),dtype=float)
    U[Dofs]=vd
    U[free]=lu.solve(np.asarray(rhs))
    r_sample=0.90*R
    theta=np.linspace(0,2*math.pi,128,endpoint=False)
    sample=np.vstack((r_sample*np.cos(theta),r_sample*np.sin(theta),np.zeros_like(theta)))
    Pmat=basis.probes(sample)
    phasors=[]; transfer={}
    for col,m in enumerate((2,-4)):
        vals=(Pmat@U[:,2*col])+1j*(Pmat@U[:,2*col+1])
        coef=np.mean(vals*np.exp(-1j*m*theta))
        phasors.append({"m":m,"complex_coefficient_V_per_Vrod":
                        [float(coef.real),float(coef.imag)],
                        "transfer_magnitude":float(abs(coef))})
        # Compare at r=0.9R with the corresponding infinite-2D leading mode
        # h_m*(r/R)^|m|. Do not sample r=R: in this geometry that circle is
        # tangent to the rods and includes Dirichlet boundary points.
        mfs_ref={2:0.8703168911,-4:0.7982172624}[m]*(r_sample/R)**abs(m)
        phasors[-1]["mfs_infinite_2d_expected_at_sample_radius_V_per_Vrod"]=float(mfs_ref)
        phasors[-1]["relative_difference_from_infinite_2d_at_sample_radius"]=float(abs(abs(coef)-mfs_ref)/mfs_ref)
        phasors[-1]["r_to_R_scaled_coefficient_2d_equivalent"]=float(abs(coef)/(r_sample/R)**abs(m))
        transfer[str(m)]=float(abs(coef)/(r_sample/R)**abs(m))
    out={"mesh_hmax_mm":hmax*1e3,"mesh_file":msh.name,"vertices":int(basis.N),
         "tetrahedra":int(mesh.nelements),"boundary_nodes":int(len(boundary)),
         "driven_rod_nodes":int(np.count_nonzero(driven[boundary])),
         "R_mm":R*1e3,"rod_center_radius_mm":D*1e3,"rod_radius_mm":rod_r*1e3,
         "shield_radius_mm":shield_r*1e3,"shield_length_mm":L*1e3,
         "rod_active_length_mm":rod_L*1e3,"endcap_rod_gap_mm":(L-rod_L)*.5e3,
         "end_boundary_grounded":True,
         "drive":"24 rod phasors exp(+i 2 theta_j) and exp(-i 4 theta_j); 1 V peak complex amplitude per rod; shield/end plates grounded",
         "midplane_leading_transfer_sample":phasors,
         "sample_radius_mm":r_sample*1e3,
         "sample_convention":"at r=0.90R, away from rod surfaces; normalized to r=R only under infinite-2D r^|m| scaling",
         "limitation":"linear P1 tetra FEM; preliminary two RF basis solves; no segmented-DC basis, no rounded ends, no validated surface-field maxima"}
    return out


def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    output_path=RESULTS/"finite_3d_rf_fem.json"
    # The earlier diagnostic used r=R, exactly tangent to electrodes in this
    # CAD definition; retain it under a non-result name and recompute probes.
    if output_path.exists():
        diagnostic=RESULTS/"finite_3d_rf_fem_boundary_sample_diagnostic.json"
        if not diagnostic.exists():
            diagnostic.write_text(output_path.read_text(encoding="utf-8"),encoding="utf-8")
        output_path.unlink()
    previous={}
    rows=previous.get("mesh_levels",[])
    existing={round(float(x["mesh_hmax_mm"]),5) for x in rows}
    rows.extend(solve_mesh(h) for h in MESH_LEVELS if round(h*1e3,5) not in existing)
    rows.sort(key=lambda x:float(x["mesh_hmax_mm"]),reverse=True)
    source={"source_mfs_infinite_rod_transfer":{"+2":0.8703168911,"-4":0.7982172624},
            "finite_geometry":"24 finite cylindrical rods in grounded shield; rod ends stop 1 mm short of grounded end plates; no axial rod segmentation in these RF solves",
            "mesh_levels":rows}
    changes=[]
    for coarse,fine in zip(rows[:-1],rows[1:]):
        for key,index in (("+2",0),("-4",1)):
            v0=coarse["midplane_leading_transfer_sample"][index]["r_to_R_scaled_coefficient_2d_equivalent"]
            v1=fine["midplane_leading_transfer_sample"][index]["r_to_R_scaled_coefficient_2d_equivalent"]
            changes.append({"mode":key,"coarse_hmax_mm":coarse["mesh_hmax_mm"],
                            "fine_hmax_mm":fine["mesh_hmax_mm"],
                            "relative_change_using_fine_denominator":abs(v1-v0)/abs(v1)})
    source["mesh_refinement_relative_change"]=changes
    output_path.write_text(json.dumps(source,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(source,indent=2))


if __name__=="__main__":
    main()
