"""Reconcile energy and angular momentum from independent DOP853 trajectories."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import verify_strict_leaders_dop853 as verify
import basin_scan

HERE=Path(__file__).resolve().parent
DATA=HERE/"strict_long_validation"


def main():
    pdoc=json.loads((HERE/"local15_parameters.json").read_text(encoding="utf-8"))
    p=np.asarray(pdoc["candidate"]["p"],float);qratio=float(pdoc["candidate"]["q2_over_q1"])
    _p0,family,modes,coeff=basin_scan.load_model()
    records=[json.loads(s) for s in (DATA/"dop853_energy_validation.jsonl").read_text().splitlines() if s.strip()]
    out=[]
    for rec in records:
        z=np.load(DATA/rec["trajectory_file"])
        tau=z["tau"];states=z["state"];work=z["work"];explicit=z["explicit_work"];torque=z["torque"]
        protocol=z["protocol"];phase=np.asarray(rec["rf_phase"],float)
        trans=int(np.ceil(verify.ld.protocol_duration_cycles(protocol)))
        skip=trans+verify.SKIP_RELAX;ix=min(skip,len(tau)-1)
        H0=verify.energy_at(float(tau[0]),np.r_[states[0],np.zeros(25)],p,protocol,phase,family,modes,coeff,qratio)
        Hend=verify.energy_at(float(tau[-1]),np.r_[states[-1],np.zeros(25)],p,protocol,phase,family,modes,coeff,qratio)
        Hskip=verify.energy_at(float(tau[ix]),np.r_[states[ix],np.zeros(25)],p,protocol,phase,family,modes,coeff,qratio)
        explicit_full=(explicit[-1]-explicit[0]).tolist()
        explicit_post=(explicit[-1]-explicit[ix]).tolist()
        drag_full=float(work[-1,:,4].sum()-work[0,:,4].sum())
        drag_post=float(work[-1,:,4].sum()-work[ix,:,4].sum())
        full_res=(Hend["H_total"]-H0["H_total"])-sum(explicit_full)-drag_full
        post_res=(Hend["H_total"]-Hskip["H_total"])-sum(explicit_post)-drag_post
        masses=np.asarray([1.,1./p[4]])
        vel=states[:,[3,4,5,9,10,11]].reshape(-1,2,3)
        K=.5*masses[None,:]*np.sum(vel*vel,axis=2)
        per={}
        for i in range(2):
            dl=float(masses[i]*(states[-1,6*i]*states[-1,6*i+4]-states[-1,6*i+1]*states[-1,6*i+3])-
                     masses[i]*(states[ix,6*i]*states[ix,6*i+4]-states[ix,6*i+1]*states[ix,6*i+3]))
            td=torque[-1,i]-torque[ix,i]
            dw=work[-1,i]-work[ix,i]
            dk=float(K[-1,i]-K[ix,i])
            per[str(i+1)]={"post_relaxation_delta_K":dk,"post_relaxation_work_by_source_RF1_RF2_DC_Coulomb_drag":dw.tolist(),
               "post_relaxation_kinetic_residual":float(dk-dw.sum()),"post_relaxation_delta_Lz":dl,
               "post_relaxation_integrated_torque_RF1_RF2_DC_Coulomb_drag":td.tolist(),
               "post_relaxation_angular_residual":float(dl-td.sum()),
               "full_interval_delta_K":float(K[-1,i]-K[0,i]),
               "full_interval_work_by_source":(work[-1,i]-work[0,i]).tolist()}
        L=masses[None,:]*(states[:,[0,6]]*states[:,[4,10]]-states[:,[1,7]]*states[:,[3,9]])
        dL=float(L[-1].sum()-L[ix].sum());td=torque[-1].sum(axis=0)-torque[ix].sum(axis=0)
        per_summary={"protocol_id":rec["protocol_id"],"representative":f"{rec['ensemble']}:{rec['start_id']}",
          "strict_category_DOP853":rec["strict_dop853"]["category"],"RK4_category":rec["rk4_category"],
          "RK4_DOP853_final_max_abs_difference":rec["rk4_final_max_abs_delta_vs_DOP853"],
          "DOP853_turns":rec["strict_dop853"]["turns_after_relaxation"],
          "DOP853_frequencies_Hz":rec["frequency_Hz"],"nfev":rec["nfev"],"elapsed_seconds":rec["elapsed_seconds"],
          "first_to_last_system_energy":{"H_initial":H0,"H_final":Hend,"delta_H":Hend["H_total"]-H0["H_total"],
            "explicit_work_RF1_RF2_DC":explicit_full,"drag_work_total":drag_full,"balance_residual":float(full_res)},
          "after_transition_plus_200_cycles":{"H_start":Hskip,"H_end":Hend,
            "delta_H":Hend["H_total"]-Hskip["H_total"],"explicit_work_RF1_RF2_DC":explicit_post,
            "drag_work_total":drag_post,"balance_residual":float(post_res),
            "integrated_torque_system_RF1_RF2_DC_Coulomb_drag":td.tolist(),
            "delta_Lz_system":dL,"angular_balance_residual":float(dL-td.sum())},
          "per_particle_ledger":per,"trajectory_file":rec["trajectory_file"]}
        out.append(per_summary)

    (DATA/"dop853_energy_balance_summary.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    for r in out:
        print(r["protocol_id"],"full energy residual",r["first_to_last_system_energy"]["balance_residual"],
              "post energy residual",r["after_transition_plus_200_cycles"]["balance_residual"],
              "angular residual",r["after_transition_plus_200_cycles"]["angular_balance_residual"])

    leader=next(r for r in records if r["protocol_id"]==505)
    z=np.load(DATA/leader["trajectory_file"]);tau=z["tau"];y=z["state"]
    cyc=tau/(2*np.pi);pos=y[:,[0,1,2,6,7,8]].reshape(-1,2,3)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)/(2*np.pi)
    rad=np.linalg.norm(pos[:,:,:2],axis=2)
    fig=plt.figure(figsize=(14,9))
    ax=fig.add_subplot(221,projection="3d")
    ax.plot(pos[:,0,0],pos[:,0,1],pos[:,0,2],lw=.6,label="particle 1 (+)")
    ax.plot(pos[:,1,0],pos[:,1,1],pos[:,1,2],lw=.6,label="particle 2 (−)")
    ax.set(xlabel="x/R",ylabel="y/R",zlabel="z/R",title="DOP853 stroboscopic 3D paths")
    ax.legend(loc="upper left")
    ax2=fig.add_subplot(222)
    ax2.plot(pos[:,0,0],pos[:,0,1],lw=.45,label="particle 1")
    ax2.plot(pos[:,1,0],pos[:,1,1],lw=.45,label="particle 2")
    ax2.set_aspect("equal",adjustable="datalim");ax2.set(xlabel="x/R",ylabel="y/R",title="XY projection, one sample per RF1 period");ax2.legend()
    ax3=fig.add_subplot(223)
    ax3.plot(cyc,theta[:,0]-theta[0,0],label="particle 1")
    ax3.plot(cyc,theta[:,1]-theta[0,1],label="particle 2")
    ax3.axvline(3,color="black",ls="--",lw=.8,label="load complete")
    ax3.axvline(203,color="grey",ls=":",lw=.8,label="relaxation complete")
    ax3.set(xlabel="Primary RF periods",ylabel="Accumulated turns",title="Unwrapped azimuth")
    ax3.legend(ncol=2)
    ax4=fig.add_subplot(224)
    ax4.plot(cyc,rad[:,0],label="particle 1")
    ax4.plot(cyc,rad[:,1],label="particle 2")
    ax4.axhline(1-.00189,color="black",ls="--",lw=.7,label="particle 2 wall center limit")
    ax4.set(xlabel="Primary RF periods",ylabel="r/R",title="Radial histories")
    ax4.legend()
    fig.suptitle("Protocol #505, independent DOP853 representative; stroboscopic output once per RF1 period")
    fig.tight_layout();fig.savefig(DATA/"protocol505_dop853_trajectory.png",dpi=180);plt.close(fig)

if __name__=="__main__":main()
