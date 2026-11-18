"""Summarize completed RK trajectories and draw the single V17C.5 figure.
Optional positional argument: original input ZIP for an independent reference comparison.
"""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CUT = 2000
RUNDIR = ROOT/"rk"/"results"
CFG = json.loads((ROOT/"input"/"config.json").read_text(encoding="utf-8"))
F1 = CFG["f1_Hz"]


def energies(states, coupling):
    positions = states[:, :, :3]
    speed2 = np.sum(states[:, :, 3:]**2,axis=-1)
    beta,s = CFG["parameters"][3:5]
    kinetic = .5*speed2*np.array([1,1/s])
    dc = beta*(positions[:,:,2]**2 - .5*np.sum(positions[:,:,:2]**2,axis=-1))
    coul = coupling/np.linalg.norm(positions[:,0]-positions[:,1],axis=1)
    return kinetic,dc,coul


def evaluate(file, coupling):
    with np.load(file) as data:
        period = data["tau"]/(2*np.pi)
        states = data["state"].reshape(-1,2,6)
        turns = data["turns"]-data["turns"][0]
        work = data["work"]
    assert np.allclose(period,np.arange(len(period)),rtol=0,atol=1e-8)
    start = CUT
    windows = np.linspace(start,len(period)-1,4,dtype=int)
    wturns = np.diff(turns[windows],axis=0)
    z = states[:,:,0] + 1j*states[:,:,1]
    unwrapped = np.unwrap(np.angle(z),axis=0)/(2*np.pi)
    unwrapped -= unwrapped[0].copy()
    kinetic,dc,coul = energies(states,coupling)
    energy = (kinetic+dc).sum(axis=1)+coul
    late_work = work[-1]-work[start]
    resid_system = energy-energy[0]-((work[:,:,0]+work[:,:,3])-(work[0,:,0]+work[0,:,3])).sum(axis=1)
    resid_kinetic = (kinetic-kinetic[0])-(work-work[0]).sum(axis=-1)
    turns_late = turns[-1]-turns[start]
    x = states[start:,:,:3]
    late_d = np.linalg.norm(x[:,0]-x[:,1],axis=1)
    return dict(periods=len(period)-1,windows=windows.tolist(),turns_late=turns_late.tolist(),
                freq_Hz=(F1*turns_late/(period[-1]-period[start])).tolist(),
                window_turns=wturns.tolist(),window_f_Hz=(F1*wturns/np.diff(period[windows])[:,None]).tolist(),
                radius_strobe_mean=np.abs(z[start:]).mean(axis=0).tolist(),
                radius_strobe_min=np.abs(z[start:]).min(axis=0).tolist(),
                radius_strobe_max=np.abs(z[start:]).max(axis=0).tolist(),
                position_abs_z_max_all=np.max(np.abs(states[:,:,2]),axis=0).tolist(),
                position_abs_z_max_late=np.max(np.abs(states[start:,:,2]),axis=0).tolist(),
                late_distance_min_mean=[late_d.min(),late_d.mean()],
                late_mean_coulomb_energy=float((coupling/late_d).mean()),
                phase_unwrap_vs_step_turns_max=np.abs(turns-unwrapped).max(axis=0).tolist(),
                energy_units="m1*(R*Omega1)^2", power_channel_order=["RF","DC","Coulomb","drag"],
                work_late=late_work.tolist(), kinetic_initial=kinetic[0].tolist(),
                energy_change_late=float(energy[-1]-energy[start]),
                energy_residual_late=float(resid_system[-1]-resid_system[start]),
                energy_residual_max_all=float(np.max(abs(resid_system))),
                per_particle_kinetic_residual_max=np.max(np.abs(resid_kinetic),axis=0).tolist())


def figure(file):
    with np.load(file) as data:
        p = data["tau"]/(2*np.pi)
        st = data["state"].reshape(-1,2,6)
        turns = data["turns"]-data["turns"][CUT]
    colors = ["#165a87","#b3491e"]
    fig, ax = plt.subplots(1,3,figsize=(13,4.3),gridspec_kw={"width_ratios":[1,1.2,1.2]},layout="constrained")
    ax[0].add_patch(plt.Circle((0,0),1,fc="none",ec="0.65",lw=.8,ls="--"))
    ax[0].scatter([0],[0],s=12,c="black",marker="+")
    for i,color in enumerate(colors):
        tail = st[-451:,i]
        ax[0].scatter(tail[:,0],tail[:,1],s=4,c=color,alpha=.6)
        ax[1].plot(p[CUT:],turns[CUT:,i],color=color,lw=1.6,label=f"Частица {i+1}")
        rr = np.hypot(st[:-1,i,0],st[:-1,i,1]).reshape(-1,100)
        pp = p[:-1].reshape(-1,100).mean(axis=1)
        ax[2].fill_between(pp,rr.min(axis=1),rr.max(axis=1),color=color,alpha=.20,lw=0)
        ax[2].plot(pp,rr.mean(axis=1),color=color,lw=1.4)
    ax[0].set(title="Проекция XY · последние 450 T₁",aspect="equal",xlim=(-1.04,1.04),ylim=(-1.04,1.04),xlabel="x/R",ylabel="y/R")
    ax[1].set(title="Обороты после 2 000 T₁",xlabel="Время, T₁",ylabel="Накопленное число оборотов")
    ax[1].axhline(0,lw=.6,color=".55")
    ax[1].legend(loc="upper left")
    ax[2].set(title="Радиус · окна по 100 T₁",xlabel="Время, T₁",ylabel="r/R",ylim=(0,.75))
    ax[2].axvline(CUT,ls="--",lw=.9,color=".45")
    for a in ax:
        a.grid(alpha=.18)
    fig.suptitle("24 электрода, две RF-частоты · C = 10⁻⁵ · RK4, 128 шагов/T₁",fontsize=14)
    fig.savefig(ROOT/"counterrotation.png",dpi=170)
    plt.close(fig)


def main(archive):
    strengths = {"nom64":1e-5,"nom128":1e-5,"c0":0.,"c5e6":5e-6,"c2e5":2e-5,
                 "oseen":1e-5,"rplus1":1e-5,"combined":1e-5}
    results = {name:evaluate(RUNDIR/f"{name}.npz",c) for name,c in strengths.items() if (RUNDIR/f"{name}.npz").is_file()}
    if archive:
        with ZipFile(archive) as z:
            for name,filename in [("nom64","base64"),("nom128","base128")]:
                with np.load(BytesIO(z.read(f"v17c_dualfreq/{filename}.npz"))) as source, np.load(RUNDIR/f"{name}.npz") as data:
                    archived = source["trajectory"][:,1:13].reshape(-1,2,6)
                    theta = np.unwrap(np.arctan2(archived[:,:,1],archived[:,:,0]),axis=0)
                    archival_freq = F1*(theta[-1]-theta[CUT])/(2*np.pi*(len(theta)-1-CUT))
                    results[name]["archived_freq_Hz"] = archival_freq.tolist()
                    results[name]["vs_archive_freq_abs_Hz"] = np.abs(np.array(results[name]["freq_Hz"])-archival_freq).tolist()
                    results[name]["vs_archive_max_state_difference"] = float(np.max(abs(data["state"].reshape(-1,2,6)-archived)))
    (ROOT/"comparison.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    figure(RUNDIR/"nom128.npz")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source_zip",nargs="?")
    main(ap.parse_args().source_zip)
