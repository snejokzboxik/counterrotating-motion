from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RES=ROOT/'results'
FIG=ROOT/'figures'
FIG.mkdir(exist_ok=True)

def loadj(name): return json.loads((RES/name).read_text(encoding='utf-8'))

def trajectories():
    meta=loadj('baseline_reproduction.json')['rows']
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),constrained_layout=True)
    colors=('#1769aa','#d1495b')
    for ax,row in zip(axs,meta):
        name=row['candidate'].replace('scaled_','').replace('_f45hz','').replace('_f50hz','')
        z=np.load(RES/f"baseline_{row['candidate']}_15000p_rk464.npz")
        s=z['states']; ix=z['step_index']/z['steps']
        for i,j in enumerate((0,6)):
            ax.plot(s[:,j],s[:,j+1],lw=.65,color=colors[i],alpha=.72,label=f'частица {i+1}')
            ax.scatter(s[0,j],s[0,j+1],s=16,color=colors[i],marker='o',zorder=4)
        ax.add_patch(plt.Circle((0,0),1,fill=False,ls='--',lw=.8,color='0.45'))
        ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='x/R',ylabel='y/R',title=name)
        ax.grid(alpha=.2)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Общие XY-траектории, RK4/64, 15 000 периодов; пунктир — радиальная граница R')
    fig.savefig(FIG/'matched_trajectories_xy.png',dpi=180);plt.close(fig)

    fig,axs=plt.subplots(2,1,figsize=(10,7),sharex=True,constrained_layout=True)
    for row in meta:
        name=row['candidate'].replace('scaled_','').replace('_f45hz','').replace('_f50hz','')
        z=np.load(RES/f"baseline_{row['candidate']}_15000p_rk464.npz")
        s=z['states']; cyc=z['step_index']/z['steps']
        # decoded angle is already continuous in stored trace, reconstructed at dense-enough 10-cycle cadence
        for k,j in enumerate((0,6)):
            theta=np.unwrap(np.arctan2(s[:,j+1],s[:,j]))/(2*np.pi)
            axs[0].plot(cyc,theta,label=f'{name}, {k+1}',lw=1)
            axs[1].plot(cyc,np.hypot(s[:,j],s[:,j+1]),label=f'{name}, {k+1}',lw=1)
    axs[0].set_ylabel('накопленные обороты');axs[1].set_ylabel('r/R');axs[1].set_xlabel('периоды RF₁')
    for ax in axs: ax.grid(alpha=.25);ax.legend(ncol=3,fontsize=7,frameon=False)
    fig.suptitle('Частоты и радиусы на полной 15k контрольной траектории')
    fig.savefig(FIG/'turns_and_radius.png',dpi=180);plt.close(fig)

def parameter_maps():
    rows=[json.loads(x) for x in (RES/'parameter_grid_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    axes=('alpha1','alpha2','lambda1','lambda2','mu','C12')
    pairs=(('alpha1','alpha2'),('lambda1','lambda2'),('alpha1','C12'))
    codes={'SCREEN_CR':1,'SCREEN_CO':-1,'PARTICLE_CONTACT':-2,'WALL_CONTACT':-3,'ESCAPE':-4,'UNRESOLVED':0,'NONROTATING_OR_SLOW':0}
    fig,axs=plt.subplots(3,3,figsize=(14,12),constrained_layout=True);im=None
    for ri,pair in enumerate(pairs):
      for ci,candidate in enumerate(sorted({r['candidate'] for r in rows})):
        ax=axs[ri,ci];sel=[r for r in rows if r['candidate']==candidate and r['slice']==f'{pair[0]}__{pair[1]}']
        xs=sorted({r['multiplier_x'] for r in sel});ys=sorted({r['multiplier_y'] for r in sel})
        arr=np.full((len(ys),len(xs)),np.nan)
        for r in sel: arr[ys.index(r['multiplier_y']),xs.index(r['multiplier_x'])]=codes.get(r['screen_class'],0)
        # The nominal centre is supplied by the separate 15k baseline run, not the off-centre grid file.
        arr[ys.index(1.0),xs.index(1.0)]=1
        im=ax.imshow(arr,origin='lower',vmin=-4,vmax=1,cmap='RdYlBu',aspect='auto')
        ax.set_xticks(range(len(xs)),[f'{v:.1f}' for v in xs]);ax.set_yticks(range(len(ys)),[f'{v:.1f}' for v in ys])
        ax.set_xlabel(f'{pair[0]} × nominal');ax.set_ylabel(f'{pair[1]} × nominal')
        ax.set_title(candidate.replace('scaled_','').replace('_f45hz','').replace('_f50hz',''))
    fig.suptitle('Короткий математический экран: статус знаков/удержания после 3 000 периодов\n1=CR, −1=CO; не является картой аттракторов')
    cb=fig.colorbar(im,ax=axs.ravel().tolist(),ticks=[-4,-3,-2,-1,0,1],shrink=.76,pad=.02)
    cb.ax.set_yticklabels(['escape','wall','particle','CO','unresolved','CR'])
    fig.savefig(FIG/'parameter_slices_screen.png',dpi=180);plt.close(fig)

def energy_and_spectra():
    s=loadj('refine12_240k_corrected_drag_summary.json')['energy_angular_momentum']
    z=np.load(RES/'refine12_240k_corrected_drag_trajectory.npz');wledger=z['work_history'][-1,:20]
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    labels=['RF','DC','Coulomb','drag','gravity'];x=np.arange(len(labels));w=.36
    work=np.array([[wledger[k],wledger[k+5]] for k in range(5)]);work=work/np.maximum(1.,np.abs(work[0]))
    torque=np.array([[wledger[10+k],wledger[15+k]] for k in range(5)]);torque=torque/np.maximum(1.,np.abs(torque[0]))
    for ax,arr,title,yl,lt in ((axs[0],work,'Работа каждой силы / |работа RF|','относительная работа',1e-8),(axs[1],torque,'Момент каждой силы / |момент RF|','относительный момент',1e-6)):
        ax.bar(x-w/2,arr[:,0],w,label='частица 1');ax.bar(x+w/2,arr[:,1],w,label='частица 2')
        ax.set_yscale('symlog',linthresh=lt);ax.set_xticks(x,labels,rotation=18);ax.axhline(0,color='black',lw=.6);ax.set_ylabel(yl+' (symlog)');ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
    fig.suptitle('refine12_sobol17, corrected drag, 240 000 RF-периодов')
    fig.savefig(FIG/'energy_and_torque_ledger.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
    for fn,label in [('lyapunov_candidate52_2000p.json','candidate52'),('lyapunov_local15_2000p.json','Local15'),('lyapunov_refine12_sobol17_2000p.json','refine12')]:
        d=loadj(fn);ax.plot([r['periods'] for r in d['rows']],[abs(r['top_per_tau']) for r in d['rows']],label=label)
    ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('RF-периоды накопления QR');ax.set_ylabel('|верхнего условного механического показателя|, 1/τ')
    ax.set_title('Медленное сближение верхней оценки к нулю');ax.grid(which='both',alpha=.25);ax.legend(frameon=False)
    fig.savefig(FIG/'conditional_lyapunov_convergence.png',dpi=180);plt.close(fig)

    rows=[json.loads(x) for x in (RES/'coulomb_sensitivity_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    fig,axs=plt.subplots(1,3,figsize=(13,4),sharex=True,constrained_layout=True)
    for ax,candidate in zip(axs,sorted({r['candidate'] for r in rows})):
        g=sorted((r for r in rows if r['candidate']==candidate),key=lambda r:r['sC']);sc=[r['sC'] for r in g]
        ax.plot(sc,[r['frequency_Hz'][0] for r in g],'-o',ms=3,label='частица 1')
        ax.plot(sc,[r['frequency_Hz'][1] for r in g],'-o',ms=3,label='частица 2')
        ax.set_xscale('symlog',linthresh=.1);ax.set_xlabel('диагностический множитель Coulomb');ax.set_ylabel('частота обращения, Гц');ax.set_title(candidate.replace('scaled_','').replace('_f45hz','').replace('_f50hz',''));ax.grid(alpha=.2)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle('5 000-периодный Coulomb screen: движение остаётся CR при C=0…5C₀')
    fig.savefig(FIG/'coulomb_sensitivity.png',dpi=180);plt.close(fig)

def basin_panel():
    d=loadj('basin_ensemble_summary.json');cands=sorted({x['candidate'] for x in d['candidate_ensemble_results']});ens=['rest','weak','wide','edge']
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True);arr=np.full((len(cands),len(ens)),np.nan)
    for r in d['candidate_ensemble_results']:arr[cands.index(r['candidate']),ens.index(r['ensemble'])]=r['strict_cr_fraction']
    im=ax.imshow(arr,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    for i in range(len(cands)):
      for j in range(len(ens)):
        x=d['candidate_ensemble_results'][i*4+j]
        ax.text(j,i,f"{x['counts']['STRICT_CR']}/8\nwall {x['counts']['WALL_CONTACT']}",ha='center',va='center',color='white' if arr[i,j]<.6 else 'black',fontsize=9)
    ax.set_xticks(range(4),ens);ax.set_yticks(range(3),[x.replace('scaled_','').replace('_f45hz','').replace('_f50hz','') for x in cands])
    ax.set_title('Детерминированный поднабор frozen starts; denominator = все N=8')
    fig.colorbar(im,ax=ax,label='доля strict CR среди всех восьми стартов')
    fig.savefig(FIG/'basin_subsample.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    trajectories();parameter_maps();energy_and_spectra();basin_panel()
