"""Independent P1 FEM solve for G1/G2 and interior complex spectrum fit.

This extends the earlier V17C.2 coefficient audit: solve the 2D Laplace
boundary-value problem on three triangular meshes and recover the signed
harmonic spectrum used for pseudopotential/Floquet comparisons.
"""
from __future__ import annotations
import json,math
from pathlib import Path
import numpy as np
import triangle as tr
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.tri as mtri
from scipy.optimize import root
from rf_dc_core import RESULTS,integrate_rk4,load_geometry,pseudopotential,run_rk4
from g1_g2_translation_spectrum import grad,hessian

HERE=Path(__file__).resolve().parent
OUT=RESULTS/"g1_g2_joint"/"independent_fem_spectrum"
OUT.mkdir(parents=True,exist_ok=True)

def build_mesh(N,a,gap,hmax,nbound):
    B=1.0+a+gap
    vertices=[];segments=[];markers=[];holes=[]
    outerN=max(160,int(math.ceil(2*np.pi*B/hmax)))
    angles=np.arange(outerN)*2*np.pi/outerN
    start=len(vertices);vertices.extend([[B*np.cos(t),B*np.sin(t)] for t in angles])
    for j in range(outerN):
        segments.append([start+j,start+(j+1)%outerN]);markers.append(N+1)
    for j in range(N):
        th=2*np.pi*j/N;center=np.array([np.cos(th),np.sin(th)])
        q=np.arange(nbound)*2*np.pi/nbound;start=len(vertices)
        vertices.extend((center+a*np.column_stack([np.cos(q),np.sin(q)])).tolist())
        for k in range(nbound):
            segments.append([start+k,start+(k+1)%nbound]);markers.append(j+1)
        holes.append(center.tolist())
    inp={"vertices":np.asarray(vertices,float),"segments":np.asarray(segments,np.int32),
         "segment_markers":np.asarray(markers,np.int32)[:,None],"holes":np.asarray(holes,float)}
    maxarea=.5*hmax*hmax
    # Triangle's PSLG option parser does not reliably accept scientific
    # notation here (for example 7.2e-05); force a decimal literal.
    mesh=tr.triangulate(inp,f"pq28a{maxarea:.15f}Q")
    return mesh,B

def fem_solve(record,hmax,nbound,modes):
    N,m=int(record["N"]),int(record["m"])
    a,gap,R=float(record["a_D"]),float(record["gap_D"]),float(record["R_D"])
    mesh,B=build_mesh(N,a,gap,hmax,nbound)
    pts=mesh["vertices"];tri=mesh["triangles"];segs=mesh["segments"]
    marks=mesh["segment_markers"].ravel()
    p=pts[tri];x=p[:,:,0];y=p[:,:,1]
    det=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0])
    area=np.abs(det)*.5
    b=np.column_stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]])/(2*area[:,None])
    c=np.column_stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]])/(2*area[:,None])
    ke=area[:,None,None]*(b[:,:,None]*b[:,None,:]+c[:,:,None]*c[:,None,:])
    rows=np.repeat(tri,3,axis=1).ravel();cols=np.tile(tri,(1,3)).ravel()
    K=coo_matrix((ke.ravel(),(rows,cols)),shape=(len(pts),len(pts))).tocsr()
    fixed=np.unique(segs.ravel());bc=np.zeros(len(pts),dtype=np.complex128)
    for j in range(N):
        nodes=np.unique(segs[marks==j+1].ravel())
        bc[nodes]=np.exp(1j*m*2*np.pi*j/N)
    free=np.setdiff1d(np.arange(len(pts)),fixed)
    bc[free]=spsolve(K[free][:,free],-K[free][:,fixed]@bc[fixed])

    triang=mtri.Triangulation(pts[:,0],pts[:,1],tri)
    interp=mtri.LinearTriInterpolator(triang,bc.real)
    interpi=mtri.LinearTriInterpolator(triang,bc.imag)
    radii_u=np.array([.30,.42,.55,.66,.75,.83,.89])
    theta=np.arange(1024)*2*np.pi/1024
    projections={}
    for ru in radii_u:
        radius=ru*R
        xx=radius*np.cos(theta);yy=radius*np.sin(theta)
        values=np.asarray(interp(xx,yy))+1j*np.asarray(interpi(xx,yy))
        if np.any(np.ma.getmaskarray(values)):
            raise RuntimeError(f"FEM interpolation left mesh at r/R={ru}")
        values=np.asarray(values)
        for k in modes:
            projections.setdefault(int(k),[]).append(
                complex(np.mean(values*np.exp(-1j*int(k)*theta))))
    coeff={}
    for k,proj in projections.items():
        n=abs(k);xrad=radii_u**n
        # Least-squares fit to A_k (r/R)^|k| across interior circles.
        coeff[k]=complex(np.dot(xrad,np.asarray(proj))/np.dot(xrad,xrad))
    # Polygonal circle error and coefficient/field convergence diagnostics.
    circle_sagitta=float(a*(1-np.cos(np.pi/nbound)))
    return {"vertices":int(len(pts)),"triangles":int(len(tri)),
        "mesh_h":hmax,"nbound":nbound,"outer_radius":B,"R_D":R,
        "circle_sagitta_D":circle_sagitta,"coefficients":coeff,
        "radii_u":radii_u.tolist()}

def phase_normalize(coeff,lead):
    phase=np.angle(coeff[lead])
    rot=-phase/lead
    normalized={k:v/abs(coeff[lead])*np.exp(1j*k*rot) for k,v in coeff.items()}
    normalized[lead]=1+0j
    return normalized,rot

def field_difference(coeff_fem,coeff_mfs,lead,radii):
    rows=[]
    for r in radii:
        keys=set(coeff_fem)&set(coeff_mfs)
        ef=sum(abs(k*c)*r**(abs(k)-1) for k,c in coeff_fem.items()
               if k in keys)
        em=sum(abs(k*coeff_mfs[k])*r**(abs(k)-1) for k in keys)
        diff=sum(abs(k*(coeff_fem[k]-coeff_mfs[k]))*r**(abs(k)-1) for k in keys)
        rows.append({"r_over_R":r,"MFS_field_l1":em,"FEM_field_l1":ef,
                     "relative_field_l1_spectrum_delta":float(diff/max(em,1e-30))})
    return rows

def map_period(v,modes,coeff,alpha,beta,lam,steps=128,save=False):
    y0=np.zeros(13);y0[[0,1,3,4]]=v
    t,d,e,c=integrate_rk4(y0,1,steps,1 if save else steps,modes,coeff,
        np.array([alpha]),np.array([beta]),np.array([lam]),np.array([1.]),
        np.array([1.]),0.,0.,5.,5.)
    return d[-1,[0,1,3,4]],t,d

def floquet_fit(name,meta,modes,coeff,B,alpha=.01,lam=1e-5):
    eigdoc=json.loads((RESULTS/"g1_g2_joint"/"G1_G2_translation_spectrum_fit.json").read_text(encoding="utf-8"))
    seedpoint=eigdoc["stationary_points"][name]["selected_for_ringdown"]
    beta=alpha*alpha*B
    x0=np.array([seedpoint["x"],seedpoint["y"]])
    ps=root(lambda xy:grad(xy,B,modes,coeff),x0,tol=1e-11)
    H=hessian(ps.x,B,modes,coeff)
    peig=np.linalg.eigvalsh(H)
    geom={**meta,"modes":modes,"coeff":coeff}
    r=np.linalg.norm(ps.x);theta=np.arctan2(ps.x[1],ps.x[0])
    probe=run_rk4(geom,[r],[theta],[0.],[alpha],[beta],[lam],
        periods=1,steps_per_period=128,save_per_period=1,phase0=0.,
        add_micromotion=True,escape_radius=5.)
    guess=probe["state"][0,[0,1,3,4]]
    pr=root(lambda v:map_period(v,modes,coeff,alpha,beta,lam)[0]-v,
            guess,method="hybr",options={"xtol":1e-11,"maxfev":300})
    end,t,orbit=map_period(pr.x,modes,coeff,alpha,beta,lam,save=True)
    center=np.mean(orbit[:,0:2],axis=0)
    Hc=hessian(center,B,modes,coeff);ev=np.linalg.eigvalsh(Hc)
    predicted=alpha*np.sqrt(np.maximum(ev,0.))
    scales=np.array([1e-6,1e-6,1e-7,1e-7]);M=np.empty((4,4))
    for j,h in enumerate(scales):
        d=np.zeros(4);d[j]=h
        M[:,j]=(map_period(pr.x+d,modes,coeff,alpha,beta,lam)[0]-
                map_period(pr.x-d,modes,coeff,alpha,beta,lam)[0])/(2*h)
    mu=np.linalg.eigvals(M);ff=np.sort(np.abs(np.angle(mu))/(2*np.pi))
    uniq=[]
    for v in ff:
        if not uniq or abs(v-uniq[-1])>2e-5:uniq.append(float(v))
        else:uniq[-1]=.5*(uniq[-1]+float(v))
    full=sorted(uniq)[:2]
    match=[]
    for v in predicted:
        nearest=min(full,key=lambda x:abs(x-v))
        match.append({"pseudo_omega_tau":float(v),"full_rf_floquet_omega_tau":nearest,
                      "relative_error":float((nearest-v)/v) if v else None})
    return {"B":B,"beta":beta,"pseudopotential_stationary_root":ps.x.tolist(),
        "pseudopotential_hessian_initial":H.tolist(),
        "pseudopotential_frequencies_initial":(alpha*np.sqrt(np.maximum(peig,0.))).tolist(),
        "rf_periodic_orbit_root_converged":bool(pr.success),
        "poincare_residual_norm":float(np.linalg.norm(end-pr.x)),
        "guiding_center":center.tolist(),"hessian_at_rf_center":Hc.tolist(),
        "pseudopotential_frequencies_at_rf_center":predicted.tolist(),
        "floquet_multipliers_abs":[float(abs(v)) for v in mu],
        "floquet_frequencies_omega_tau":full,"matched_frequency_errors":match}

def main():
    source=json.loads((HERE/"input"/"harmonics.json").read_text(encoding="utf-8"))
    results={}
    specs={"G1":("6_ABC_moderate",[(.032,64),(.020,112),(.012,160)]),
           "G2":("6_ABC_thin",[(.032,72),(.020,112),(.012,160)])}
    for name,(recordname,levels) in specs.items():
        record=next(x for x in source if x["name"]==recordname)
        meta,mfs_modes,mfs_coeff=load_geometry(name)
        mode_keep=[]
        for k,c in zip(mfs_modes,mfs_coeff):
            if abs(k*c)*.89**(abs(k)-1) > 2e-6:
                mode_keep.append(int(k))
        # Also retain the leading/counter modes irrespective of this cut.
        for k in (2,-4):
            if k not in mode_keep:mode_keep.append(k)
        mode_keep=np.array(sorted(mode_keep),dtype=np.int64)
        mfs={int(k):complex(c) for k,c in zip(mfs_modes,mfs_coeff) if int(k) in mode_keep}
        rows=[]
        for h,nb in levels:
            sol=fem_solve(record,h,nb,mode_keep)
            normalized,rotation=phase_normalize(sol["coefficients"],2)
            carray=np.array([normalized[int(k)] for k in mode_keep],complex)
            marray=np.array([mfs[int(k)] for k in mode_keep],complex)
            amp_delta=np.abs(carray-marray)
            sol["rotation_gauge_rad"]=rotation
            sol["harmonic_modes"]=mode_keep.tolist()
            sol["complex_coefficients_normalized_to_A2"]=[
                [int(k),float(c.real),float(c.imag)] for k,c in zip(mode_keep,carray)]
            sol["max_complex_delta_vs_MFS"]=float(np.max(amp_delta))
            low=np.array([abs(int(k))<=16 for k in mode_keep])
            sol["max_low_order_complex_delta_vs_MFS"] = float(np.max(amp_delta[low]))
            sol["amplitude_ratio_minus4_over_plus2"]=float(abs(normalized[-4]))
            sol["MFS_amplitude_ratio_minus4_over_plus2"]=float(abs(mfs[-4]))
            sol["relative_field_spectrum_deltas"]=field_difference(normalized,mfs,2,[.4,.6,.8,.89])
            rows.append(sol)
            print(name,"h",h,"tri",sol["triangles"],"rho",sol["amplitude_ratio_minus4_over_plus2"],
                  "max|dC|",sol["max_complex_delta_vs_MFS"],flush=True)
        fine=rows[-1]
        femcoeff={int(k):complex(re,im) for k,re,im in fine["complex_coefficients_normalized_to_A2"]}
        Bdoc=json.loads((RESULTS/"g1_g2_joint"/"G1_G2_translation_spectrum_fit.json").read_text(encoding="utf-8"))
        B=float(Bdoc["stationary_points"][name]["selected_for_ringdown"]["B_beta_over_alpha2"])
        results[name]={"boundary_conditions":"complex electrode potentials exp(i*m*theta_j), grounded circular shield",
           "mesh_levels":[{k:v for k,v in r.items() if k not in ("coefficients","complex_coefficients_normalized_to_A2")} for r in rows],
           "fine_mesh_coefficients":fine["complex_coefficients_normalized_to_A2"],
           "fine_mesh_full_spectrum_floquet_fit":floquet_fit(name,meta,mode_keep,
                    np.array([femcoeff[int(k)] for k in mode_keep]),B)}
    (OUT/"G1_G2_full_P1_FEM_spectrum.json").write_text(
        json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,(name,rec) in zip(axes,results.items()):
        for level in rec["mesh_levels"]:
            rad=[x["r_over_R"] for x in level["relative_field_spectrum_deltas"]]
            err=[x["relative_field_l1_spectrum_delta"] for x in level["relative_field_spectrum_deltas"]]
            ax.plot(rad,err,"o-",label=f"{level['triangles']:,} элементов")
        ax.set_yscale("log");ax.set_xlabel("r / R")
        ax.set_ylabel("относительная разность сумм гармоник E")
        ax.set_title(f"{name}: FEM против MFS")
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle("Сходимость P1 FEM спектра по полю внутри апертуры")
    fig.savefig(OUT/"G1_G2_fem_spectrum_convergence.png",dpi=180)
    plt.close(fig)
    print(json.dumps({k:{"mesh_levels":[x["triangles"] for x in v["mesh_levels"]],
         "rho":v["mesh_levels"][-1]["amplitude_ratio_minus4_over_plus2"],
         "floquet_fit":v["fine_mesh_full_spectrum_floquet_fit"]["matched_frequency_errors"]}
         for k,v in results.items()},indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
