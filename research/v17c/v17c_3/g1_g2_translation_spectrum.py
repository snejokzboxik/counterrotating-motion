"""P1-FEM provenance + pseudopotential/direct-RF translational frequency fit.

The independent P1 FEM G1/G2 mesh audit is committed in V17C.2. This script
reuses its fine-mesh coefficients for provenance, scans full-MFS pseudopotential
stationary points and compares Hessian frequencies with full-RF ringdown FFTs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from rf_dc_core import RESULTS, load_geometry, pseudopotential, run_rk4

HERE = Path(__file__).resolve().parent
OUT = RESULTS / "g1_g2_joint"
OUT.mkdir(parents=True, exist_ok=True)
FEM_AUDIT = HERE.parent / "v17c_2" / "input" / "independent_fem_audit.json"


def grad(xy, B, modes, coeff, h=1e-6):
    x, y = xy
    r=np.hypot(x,y)
    if r<.005 or r>.975:
        return 1e2*np.asarray([x,y])/(r+1e-30)
    f = lambda a,b: pseudopotential(1.0, B, a, b, modes, coeff)
    return np.array([(f(x+h,y)-f(x-h,y))/(2*h),
                     (f(x,y+h)-f(x,y-h))/(2*h)])


def hessian(xy, B, modes, coeff, h=2e-4):
    x,y=xy
    return np.column_stack([(grad([x+h,y],B,modes,coeff)-grad([x-h,y],B,modes,coeff))/(2*h),
                            (grad([x,y+h],B,modes,coeff)-grad([x,y-h],B,modes,coeff))/(2*h)])


def stationary_scan(name):
    meta,modes,coeff=load_geometry(name)
    found=[]
    targets=np.linspace(.14,.90,20)
    for target in targets:
        # Set the DC/RF ratio to the full-spectrum radial-mean ring estimate.
        from rf_dc_core import radial_shape
        B=radial_shape(modes,coeff,float(target))
        for r0 in (max(.05,target-.12),target,min(.96,target+.12)):
            for theta in np.linspace(0,np.pi/3,16,endpoint=False):
                seed=np.array([r0*np.cos(theta),r0*np.sin(theta)])
                sol=root(lambda xy:grad(xy,B,modes,coeff),seed,tol=1e-10)
                if not sol.success: continue
                x,y=sol.x; r=float(np.hypot(x,y))
                if not .025<r<.975: continue
                if any(np.hypot(x-p["x"],y-p["y"])<2e-3 and
                       abs(B-p["B_beta_over_alpha2"])<1e-8 for p in found): continue
                H=hessian(sol.x,B,modes,coeff)
                ev,vec=np.linalg.eigh(H)
                found.append({"x":float(x),"y":float(y),"r":r,
                    "theta":float(np.arctan2(y,x)%(2*np.pi)),
                    "B_beta_over_alpha2":float(B),
                    "hessian":H.tolist(),"eigenvalues_scaled":ev.tolist(),
                    "eigenvectors":vec.tolist(),
                    "classification":"minimum" if np.all(ev>1e-7) else
                        ("maximum" if np.all(ev<-1e-7) else "saddle")})
    return meta,modes,coeff,found


def fft_peak(signal, discard=300):
    x=np.asarray(signal[discard:],float)
    x=x-np.mean(x)
    n=len(x)
    if n<256: return {"omega_tau":None,"peak_power":None,"samples":n}
    win=np.hanning(n)
    spec=np.fft.rfft(x*win)
    power=np.abs(spec)**2
    freq=np.fft.rfftfreq(n,d=1.0)  # cycles per RF period = omega in tau units
    allowed=(freq>.002)&(freq<.45)
    ix=np.flatnonzero(allowed)[np.argmax(power[allowed])]
    # Log-parabolic interpolation reduces FFT-bin quantization error.
    delta=0.0
    if 1<=ix<len(power)-1:
        yy=np.log(power[ix-1:ix+2]+1e-300)
        denom=yy[0]-2*yy[1]+yy[2]
        if abs(denom)>1e-15: delta=.5*(yy[0]-yy[2])/denom
    return {"omega_tau":float((ix+delta)/n),
            "peak_power_fraction":float(power[ix]/max(power.sum(),1e-300)),
            "samples":n}


def ringdown(name, meta, modes, coeff, minimum, mode_index, alpha=.01,
             periods=5000, steps=64, lam=1e-5):
    xy=np.array([minimum["x"],minimum["y"]])
    ev=np.array(minimum["eigenvalues_scaled"])
    vec=np.array(minimum["eigenvectors"])
    beta=alpha*alpha*minimum["B_beta_over_alpha2"]
    r=float(np.linalg.norm(xy));theta=float(np.arctan2(xy[1],xy[0]))
    direction=vec[:,mode_index]
    dxy=3e-4*direction
    out=run_rk4({**meta,"modes":modes,"coeff":coeff},
        [float(np.linalg.norm(xy+dxy))],
        [float(np.arctan2((xy+dxy)[1],(xy+dxy)[0]))],[0.0],
        [alpha],[beta],[lam],periods=periods,steps_per_period=steps,
        save_per_period=1,phase0=.37,add_micromotion=True,escape_radius=.995)
    state=out["state"]
    projection=(state[:,0:2]-xy)@direction
    pred=alpha*np.sqrt(max(ev[mode_index],0.0))
    measured=fft_peak(projection)
    result={"geometry":name,"mode_index":mode_index,"radius_equilibrium":r,
        "theta_equilibrium":theta,"alpha":alpha,"beta":beta,"B":minimum["B_beta_over_alpha2"],
        "lambda":lam,"pseudopotential_hessian_eigenvalue_scaled":float(ev[mode_index]),
        "pseudopotential_omega_tau":float(pred), "full_rf_fft":measured,
        "relative_frequency_error":None if measured["omega_tau"] is None else
            float((measured["omega_tau"]-pred)/pred),
        "periods":out["duration_rf_periods"],"escaped":out["escaped"],
        "radius_min":out["particles"][0]["radius_min"],
        "radius_max":out["particles"][0]["radius_max"],
        "turns":out["particles"][0]["turns"]}
    np.savez_compressed(OUT/f"{name}_translation_mode{mode_index}_ringdown.npz",
        t=out["t"],state=state,projection=projection,
        parameters=np.array([alpha,beta,lam,*xy,*direction]))
    return result


def main():
    fem=json.loads(FEM_AUDIT.read_text(encoding="utf-8"))
    fem_summary={}
    for record in fem:
        if record["name"] not in ("6_ABC_moderate","6_ABC_thin"): continue
        levels=record["levels"]
        final=levels[-1]
        lead=np.array(final["leading_FEM"],complex)
        counter=np.array(final["counter_FEM"],complex)
        leadc=complex(*final["leading_FEM"]); counterc=complex(*final["counter_FEM"])
        fem_summary[record["name"]]={
            "mesh_levels":[{"triangles":x["triangles"],"h":x["mesh_h"],
                "Aplus_abs_error":x["leading_abs_error"],
                "Aminus_abs_error":x["counter_abs_error"]} for x in levels],
            "fine_mesh_Aplus":final["leading_FEM"],
            "fine_mesh_Aminus":final["counter_FEM"],
            "fine_mesh_amplitude_ratio":float(abs(counterc/leadc)),
            "MFS_amplitude_ratio":float(abs(complex(*final["counter_MFS"])/complex(*final["leading_MFS"]))),
        }
    results={"independent_P1_FEM_from_V17C2":fem_summary,"stationary_points":{},"ringdowns":[]}
    for name in ("G1","G2"):
        meta,modes,coeff,points=stationary_scan(name)
        minima=[p for p in points if p["classification"]=="minimum"]
        # Prefer an interior stable minimum; select one away from the axis and wall.
        minima.sort(key=lambda p:abs(p["r"]-.35))
        results["stationary_points"][name]={
            "count":len(points),"minima":minima,
            "saddles":[p for p in points if p["classification"]=="saddle"]}
        if minima:
            chosen=minima[0]
            results["stationary_points"][name]["selected_for_ringdown"]=chosen
            for mode in range(2):
                results["ringdowns"].append(ringdown(name,meta,modes,coeff,chosen,mode))
            # Independent axial DC translation frequency: omega_z=sqrt(2 beta).
            beta=.01**2*chosen["B_beta_over_alpha2"]
            axial=run_rk4({**meta,"modes":modes,"coeff":coeff},
                [chosen["r"]],[chosen["theta"]],[0.0],[.01],[beta],[1e-5],
                z0=[1e-3],periods=5000,steps_per_period=64,save_per_period=1,
                phase0=.37,escape_radius=.995)
            zfit=fft_peak(axial["state"][:,2])
            expected=float(np.sqrt(2*beta))
            results["stationary_points"][name]["axial_fit"]={
                "beta":beta,"pseudopotential_omega_tau":expected,"full_rf_fft":zfit,
                "relative_frequency_error":None if zfit["omega_tau"] is None else
                    float((zfit["omega_tau"]-expected)/expected),
                "escaped":axial["escaped"]}
            np.savez_compressed(OUT/f"{name}_translation_axial_ringdown.npz",
                t=axial["t"],state=axial["state"],parameters=np.array([.01,beta,1e-5]))
    (OUT/"G1_G2_translation_spectrum_fit.json").write_text(
        json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"fem":fem_summary,
        "minima":{k:len(v["minima"]) for k,v in results["stationary_points"].items()},
        "ringdowns":results["ringdowns"]},indent=2,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
