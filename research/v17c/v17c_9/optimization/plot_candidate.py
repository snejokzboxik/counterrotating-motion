"""Plot a checked DOP853 trajectory: orbit, unwrapped turns and radii."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config-id",type=int,required=True);ap.add_argument("--ensemble",default="rest");ap.add_argument("--index",type=int,required=True);a=ap.parse_args()
    p=HERE/"stage3"/"runs"/f"dop_cfg{a.config_id}_{a.ensemble}_{a.index}.npz"
    z=np.load(p); tau=z["tau"]; state=z["state"]
    pos=state[:,[0,1,2,6,7,8]].reshape(-1,2,3)
    cyc=tau/(2*np.pi); radius=np.linalg.norm(pos[:,:,:2],axis=2)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)/(2*np.pi)
    turns=theta-theta[min(200,len(theta)-1)]
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    colors=("#1769aa","#d1495b")
    for i in range(2):
        axs[0].plot(pos[:,i,0],pos[:,i,1],lw=.55,alpha=.82,color=colors[i],label=f"particle {i+1}")
        axs[1].plot(cyc,turns[:,i],lw=1.1,color=colors[i],label=f"particle {i+1}")
        axs[2].plot(cyc,radius[:,i],lw=.8,color=colors[i],label=f"particle {i+1}")
    axs[0].add_patch(plt.Circle((0,0),1,fill=False,ls="--",color="#555",lw=.8))
    axs[0].set_aspect("equal");axs[0].set(xlabel="x/R",ylabel="y/R",title="Stroboscopic XY trajectories")
    axs[1].set(xlabel="primary RF cycles",ylabel="unwrapped turns",title="Accumulated orbital turns")
    axs[2].set(xlabel="primary RF cycles",ylabel="r/R",title="Radial localization")
    for ax in axs: ax.grid(alpha=.22);ax.legend(loc="best")
    fig.suptitle(f"V17C.9 P1 candidate {a.config_id} — independent DOP853, {len(cyc)-1:,} cycles")
    out=HERE/f"candidate_{a.config_id}_trajectory_validation.png";fig.savefig(out,dpi=180)
    print(out)
if __name__=="__main__":main()
