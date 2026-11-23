"""Publication-style spectrum of the short phase-continuous V20B replay."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE/"results"
OUT = BASE/"figures"


def main():
    meta = json.loads((RESULTS/"micromotion_spectrum_audit.json").read_text(encoding="utf-8"))
    data = np.load(RESULTS/"micromotion_spectrum_audit.npz")
    tau = data["tau"]
    sample_periods = float(np.median(np.diff(tau)))/(2*np.pi)
    freq = np.fft.fftfreq(len(tau), d=sample_periods)
    band = (np.abs(freq) > 0.05) & (np.abs(freq) < 3.8)
    win = np.hanning(len(tau))
    spectra = []
    for i in range(2):
        disp = data[f"particle{i+1}_rotating_frame_displacement"]
        z = disp[:,0]+1j*disp[:,1]
        amp = 2*np.abs(np.fft.fft((z-np.mean(z))*win))/np.sum(win)
        spectra.append(amp)

    plt.rcParams.update({
        "font.family":"DejaVu Sans","font.size":9,"axes.labelsize":10,
        "axes.titlesize":10,"legend.fontsize":8,"mathtext.fontset":"dejavusans",
        "axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none",
        "pdf.fonttype":42,
    })
    fig, axes = plt.subplots(2,1,figsize=(8.8,6.0),sharex=True)
    colors = ["#356F9F","#D17A22"]
    with (RESULTS/"micromotion_spectrum.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["particle","signed_frequency_cycles_per_RF1_period","amplitude_R"])
        for i,ax in enumerate(axes):
            ax.semilogy(freq[band],np.maximum(spectra[i][band],1e-10),".",markersize=2.0,color="#333333",
                        rasterized=False,label="Полный RK4")
            row=meta["rows"][i]
            for fam,color in ((1,colors[0]),(2,colors[1])):
                offsets=[m["signed_rotating_cycles_per_RF1_period"]
                         for m in row["M3_linear_RF_carrier_offsets"] if m["family"]==fam]
                for offset in offsets:
                    ax.axvline(offset,color=color,linewidth=0.8,alpha=0.65,
                               linestyle="--" if fam==1 else ":")
            ax.set_ylabel("Амплитуда, R (лог.)")
            ax.set_title(
                f"Частица {i+1}: высокочастотный RMS RK4 "
                f"{row['full_RK4_highpass_micromotion_RMS_R']:.5f} R; "
                f"M3 {row['M3_phase_torus_micromotion_RMS_R']:.5f} R")
            ax.grid(axis="both",color="#dddddd",linewidth=0.5,which="both")
            ax.set_axisbelow(True)
            for x,y in zip(freq[band],spectra[i][band]):
                writer.writerow([i+1,float(x),float(y)])
    axes[-1].set_xlabel("Знаковая частота в циклах на период RF1")
    axes[-1].set_xlim(-3.8,3.8)
    fig.text(0.5,0.045,"Пунктирные линии: знаковые M3 carrier offsets; синий RF1, оранжевый RF2",
             ha="center",va="bottom",fontsize=8)
    fig.suptitle("V17C.21 — Микродвижение и RF-спектр",fontsize=12)
    fig.text(0.5,0.012,
             "1000 RF1-периодов, RK4 64 шага/период; старт из позднего V20B состояния с непрерывными RF-фазами. "
             "Показан конечновременной спектр в сопутствующей системе.",
             ha="center",va="bottom",fontsize=8)
    fig.tight_layout(rect=(0,0.075,1,0.94))
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ("pdf","svg","png"):
        out=OUT/f"FIG02_micromotion_spectrum.{ext}"
        fig.savefig(out,
                    dpi=600 if ext=="png" else None,bbox_inches="tight")
        if ext=="svg":
            svg=out.read_text(encoding="utf-8")
            out.write_text("\n".join(line.rstrip() for line in svg.splitlines())+"\n",
                           encoding="utf-8")
    plt.close(fig)


if __name__=="__main__":
    main()
