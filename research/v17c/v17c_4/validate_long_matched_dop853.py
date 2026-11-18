"""Long-horizon DOP853 check for the closest ADCFEB matched pair controls."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"validation_dop853"))
from independent_dop853_check import read_spectrum,rhs,energy_and_torque
SRC=HERE/"results"/"phase_controls"/"ADCFEB_AFCBED_matched_000228"
OUT=HERE/"results"/"numerical_validation"/"long_pair_dop853"
OUT.mkdir(parents=True,exist_ok=True)

def main():
    table=list(csv.DictReader((SRC/"matched_phase_comparison.csv").open(newline="",encoding="utf-8")))
    modes,coeff=read_spectrum()
    results=[]
    for i,row in enumerate([r for r in table if r["optimized_schedule"]=="ADCFEB"]):
        fname=f"ADCFEB_chi_{float(row['chi']):.6g}.npz"
        with np.load(SRC/fname) as z:
            t=z["t"]; rk=z["state"]
        alpha=np.array([float(row["alpha1"]),float(row["alpha2"])])
        beta=np.array([float(row["beta1"]),float(row["beta2"])])
        lam=np.array([float(row["lambda1"]),float(row["lambda2"])])
        mass=np.array([1.0,float(row["mass2_over_mass1"])])
        charge=np.array([1.0,float(row["charge2_over_charge1"])])
        chi=float(row["chi"]); phase0=float(row["phase0"])
        dop=solve_ivp(rhs,(float(t[0]),float(t[-1])),rk[0].reshape(-1),method="DOP853",t_eval=t,
            rtol=1e-10,atol=1e-12,max_step=2*np.pi/96,
            args=(modes,coeff,alpha,beta,lam,mass,charge,chi,phase0))
        if not dop.success: raise RuntimeError(dop.message)
        d=dop.y.T.reshape(-1,2,13)
        err=float(np.max(np.abs(d[:,:,:6]-rk[:,:12].reshape(-1,2,6))))
        energy0=energy_and_torque(dop.y[:,0],beta,mass,charge,chi)[0]
        energy1,lz,sep=energy_and_torque(dop.y[:,-1],beta,mass,charge,chi)
        workrf=float(sum(mass[j]*d[-1,j,6] for j in range(2)))
        workdrag=float(sum(mass[j]*d[-1,j,9] for j in range(2)))
        torqC=np.array([mass[j]*d[-1,j,11] for j in range(2)])
        p1=d[:,0,:6];p2=d[:,1,:6]
        th1=np.unwrap(np.arctan2(p1[:,1],p1[:,0]));th2=np.unwrap(np.arctan2(p2[:,1],p2[:,0]))
        trp=t/(2*np.pi); relp=5/(2*np.pi*min(lam))
        jrel=min(int(np.searchsorted(trp,relp)),len(t)-1)
        turns=[float((th1[-1]-th1[jrel])/(2*np.pi)),float((th2[-1]-th2[jrel])/(2*np.pi))]
        res={"source":fname,"periods":float(t[-1]/(2*np.pi)),"rtol":1e-10,"atol":1e-12,
            "max_step_RF_period_fraction":1/96,"nfev":int(dop.nfev),"max_state_error_vs_RK4_192":err,
            "turns_after_5_drag_times":turns,"lambda":lam.tolist(),"chi":chi,
            "energy_balance_residual":float(energy1-energy0-workrf-workdrag),
            "relative_energy_balance_residual":float(abs(energy1-energy0-workrf-workdrag)/max(abs(energy0),abs(energy1),1e-30)),
            "integrated_coulomb_torque":torqC.tolist(),"sum_coulomb_torque":float(torqC.sum()),
            "minimum_separation":float(np.min(np.linalg.norm(d[:,0,:3]-d[:,1,:3],axis=1))),
            "final_separation":float(sep)}
        results.append(res)
        np.savez_compressed(OUT/f"long_dop853_{i}.npz",t=t,state=dop.y.T)
        print(json.dumps(res,ensure_ascii=False),flush=True)
    (OUT/"summary.json").write_text(json.dumps({"method":"independent SciPy DOP853 full-spectrum RHS",
        "comparisons":results},indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__":main()
