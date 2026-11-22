from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; register 3-D projection
from scipy.signal import find_peaks, periodogram

HERE=Path(__file__).resolve().parent
RES=HERE/'results'; TRAJ=RES/'trajectories'; FIG=RES/'figures'; FIG.mkdir(parents=True,exist_ok=True)
inp=json.loads((RES/'input_configuration.json').read_text(encoding='utf-8'))
scan=[json.loads(x) for x in (RES/'coulomb_scan_corrected_drag.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
field_family=np.asarray(inp['field_family'],int); modes=np.asarray(inp['field_k'],int)
coeffraw=np.asarray(inp['field_coeff'],float); coeff=coeffraw[:,0]+1j*coeffraw[:,1]
phase=np.asarray(inp['phase'],float); twopi=2*np.pi

def field_xy(pos,t,p):
    u=pos[:,0]+1j*pos[:,1]; ex=np.zeros(len(pos)); ey=np.zeros(len(pos))
    c1=p[0]*np.exp(-1j*(t+phase[0])); c2=p[1]*np.exp(-1j*(p[2]*t+phase[1]))
    for fam,k,c in zip(field_family,modes,coeff):
        n=abs(int(k)); base=u if k>0 else np.conjugate(u); carrier=c1 if fam==0 else c2
        grad=carrier*c*n*base**(n-1); ex-=grad.real; ey+=grad.imag if k>0 else -grad.imag
    return np.column_stack((ex,ey))

def read_series(config,scale=1.0):
    stem=f'{config}_C{scale:g}_rk464_15000p.npz'
    z=np.load(TRAJ/stem); states=z['states']; cycles=z['step_index']/int(z['steps']); work_history=z['work_history']
    case=next(c for c in inp['cases'] if c['id']==config); p=np.asarray(case['p_corrected'],float).copy(); p[5]*=scale
    qratio=float(case['qratio']); massfac=np.array([1.0,1.0/p[4]])
    qfac=np.array([1.0,qratio]); efac=np.array([1.0,p[4]*qratio]); f1=float(case['physical']['frequency_f1_Hz'])
    xy=states[:,[0,1,6,7]].reshape(-1,2,2); vv=states[:,[3,4,9,10]].reshape(-1,2,2)
    zz=states[:,[2,8]]; vz=states[:,[5,11]]; rr=np.linalg.norm(xy,axis=2)
    theta=np.unwrap(np.arctan2(xy[:,:,1],xy[:,:,0]),axis=0); theta_rel=theta[:,0]-theta[:,1]
    delta=states[:,:3]-states[:,6:9]; d=np.linalg.norm(delta,axis=1); dxy=np.linalg.norm(delta[:,:2],axis=1)
    vrel=np.linalg.norm(states[:,3:6]-states[:,9:12],axis=1)
    K=np.array([0.5*massfac[0]*np.sum(states[:,3:6]**2,axis=1),0.5*massfac[1]*np.sum(states[:,9:12]**2,axis=1)]).T
    Uc=p[5]/d
    L=massfac[None,:]*(xy[:,:,0]*vv[:,:,1]-xy[:,:,1]*vv[:,:,0])
    torque_rf=np.zeros((len(states),2)); torque_c=np.zeros_like(torque_rf); torque_drag=np.zeros_like(torque_rf)
    c_force_radial=np.zeros_like(torque_rf); c_force_tangential=np.zeros_like(torque_rf); coulomb_to_rf=np.zeros_like(torque_rf)
    rfract=0.15
    re=np.asarray([case['physical']['particle1']['Re_per_dimensionless_speed'],case['physical']['particle2']['Re_per_dimensionless_speed']])
    for k,tcyc in enumerate(cycles):
        t=tcyc*twopi; E=field_xy(xy[k],t,p)
        torque_rf[k]=massfac*efac*(xy[k,:,0]*E[:,1]-xy[k,:,1]*E[:,0])
        dc=delta[k]; invd3=1.0/max(np.dot(dc,dc)**1.5,1e-30)
        ac=np.array([1.0,-p[4]])[:,None]*p[5]*dc[None,:]*invd3
        torque_c[k]=massfac*(xy[k,:,0]*ac[:,1]-xy[k,:,1]*ac[:,0])
        fc=np.array([1.0,-1.0])[:,None]*p[5]*dc[None,:]*invd3
        for i in range(2):
            rad_hat=xy[k,i]/max(rr[k,i],1e-30);tan_hat=np.array([-rad_hat[1],rad_hat[0]])
            c_force_radial[k,i]=fc[i,:2]@rad_hat;c_force_tangential[k,i]=fc[i,:2]@tan_hat
            coulomb_to_rf[k,i]=np.linalg.norm(ac[i])/max(1e-30,np.linalg.norm(efac[i]*E[i]))
        speed=np.sqrt(np.sum(vv[k]**2,axis=1)+vz[k]**2)
        lam=np.array([p[6],p[7]])
        drag=lam*(1+rfract*(re*speed)**0.687)
        torque_drag[k]=-massfac*drag*(xy[k,:,0]*vv[k,:,1]-xy[k,:,1]*vv[k,:,0])
    s={
      'config':config,'scale':scale,'cycle':cycles,'time_s':cycles/f1,'states':states,'xy':xy,'vxy':vv,'z':zz,'vz':vz,
      'r':rr,'theta':theta,'theta_rel':theta_rel,'d':d,'dxy':dxy,'vrel':vrel,'K':K,'Uc':Uc,'L':L,
      'torque_rf':torque_rf,'torque_c':torque_c,'torque_drag':torque_drag,'c_force_radial':c_force_radial,
      'c_force_tangential':c_force_tangential,'coulomb_to_rf':coulomb_to_rf,'work_history':work_history,'case':case,'p':p,'frequency_f1_Hz':f1
    }
    return s

def save_csv(s):
    path=RES/f"timeseries_{s['config']}_C{s['scale']:g}.csv"
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['RF_cycle','time_s','r1_R','r2_R','z1_R','z2_R','theta1_rad_unwrapped','theta2_rad_unwrapped','relative_angle_rad_unwrapped','distance_R','xy_distance_R','relative_speed_over_ROmega','K1_over_m1R2Omega2','K2_over_m1R2Omega2','Uc_over_m1R2Omega2','Lz1_over_m1R2Omega','Lz2_over_m1R2Omega','RF_torque1','RF_torque2','C_torque1','C_torque2','drag_torque1','drag_torque2','C_force_radial1','C_force_radial2','C_force_tangential1','C_force_tangential2','C_over_RF_force1','C_over_RF_force2'])
        for i in range(len(s['cycle'])):
            w.writerow([s['cycle'][i],s['time_s'][i],*s['r'][i],*s['z'][i],*s['theta'][i],s['theta_rel'][i],s['d'][i],s['dxy'][i],s['vrel'][i],*s['K'][i],s['Uc'][i],*s['L'][i],*s['torque_rf'][i],*s['torque_c'][i],*s['torque_drag'][i],*s['c_force_radial'][i],*s['c_force_tangential'][i],*s['coulomb_to_rf'][i]])
    return path

def encounter_list(s):
    # One minimum-distance window per crossing of relative azimuth 2*pi*k;
    # this filters RF micromotion ripples that create many false local minima.
    phase_cycles=s['theta_rel']/twopi; ks=np.arange(np.ceil(phase_cycles[0]),np.floor(phase_cycles[-1])+1,dtype=int); rows=[]
    for k in ks:
        center=int(np.searchsorted(phase_cycles,k)); lo=max(0,center-50); hi=min(len(phase_cycles),center+51)
        i=lo+int(np.argmin(s['d'][lo:hi]))
        rows.append({'configuration':s['config'],'scale':s['scale'],'relative_turn':int(k),'cycle':float(s['cycle'][i]),'time_s':float(s['time_s'][i]),'distance_R':float(s['d'][i]),'xy_distance_R':float(s['dxy'][i]),'relative_angle_rad':float(s['theta_rel'][i]),'relative_speed_over_ROmega':float(s['vrel'][i]),'Uc_over_m1R2Omega2':float(s['Uc'][i]),'C_torque1':float(s['torque_c'][i,0]),'C_torque2':float(s['torque_c'][i,1])})
    return rows

def plot_orbits(s0,s1):
    fig,ax=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for a,s,title in zip(ax,[s0,s1],[f"{s0['config']} — C=0",f"{s1['config']} — C=C₀"]):
        for i,color,label in ((0,'#1769aa','частица 1'),(1,'#d1495b','частица 2')):
            a.plot(s['xy'][:,i,0],s['xy'][:,i,1],lw=.55,alpha=.85,color=color,label=label)
            a.scatter(*s['xy'][0,i],s=18,marker='o',color=color)
        a.add_patch(plt.Circle((0,0),1,fill=False,color='black',lw=1,ls='--'))
        a.set(xlabel='x/R',ylabel='y/R',title='Стробоскопическая XY-орбита: '+title);a.set_aspect('equal');a.grid(alpha=.2);a.legend()
    fig.savefig(FIG/'joint_xy_trajectories_phys0570.png',dpi=180);plt.close(fig)

def plot_3d(s):
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    for i,color,label in ((0,'#1769aa','частица 1'),(1,'#d1495b','частица 2')):
        ax.plot(s['xy'][:,i,0],s['xy'][:,i,1],s['z'][:,i],lw=.55,color=color,label=label)
        ax.scatter(*s['xy'][0,i],s['z'][0,i],s=18,color=color)
    ax.set(xlabel='x/R',ylabel='y/R',zlabel='z/R',title='Стробоскопические 3D-траектории phys0570, sC=1')
    ax.legend();fig.savefig(FIG/'joint_3d_trajectory_phys0570_C1.png',dpi=180);plt.close(fig)

def plot_close_passages(s):
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True,constrained_layout=True)
    sel=s['cycle']>=s['cycle'][-1]-2500
    ax[0].plot(s['cycle'][sel],s['d'][sel],lw=.8,color='#414487');ax[0].set_ylabel('d/R');ax[0].grid(alpha=.25)
    ax[1].plot(s['cycle'][sel],np.rad2deg(np.unwrap(s['theta_rel'][sel])),lw=.8,color='#f28e2b');ax[1].set_ylabel('разность углов, град');ax[1].set_xlabel('период первой RF');ax[1].grid(alpha=.25)
    ax[0].set_title(f"Сближения и относительная фаза: {s['config']}, sC={s['scale']:g}")
    fig.savefig(FIG/'distance_and_relative_phase_phys0570_C1.png',dpi=180);plt.close(fig)

def plot_moments(s):
    sel=s['cycle']>=s['cycle'][-1]-2000; cyc=s['cycle'][sel]
    fig,ax=plt.subplots(3,1,figsize=(12,10),constrained_layout=True)
    for torque,label,color in ((s['torque_rf'],'RF','#1769aa'),(s['torque_drag'],'drag','#d1495b')):
        ax[0].plot(cyc,torque[sel,0],label=f'{label}, 1',lw=.8,color=color)
        ax[0].plot(cyc,torque[sel,1],label=f'{label}, 2',lw=.8,color=color,ls='--')
    ax[0].set_ylabel('момент на RF-стробе');ax[0].set_title('RF подкачивает вращение, сопротивление снимает момент');ax[0].grid(alpha=.2);ax[0].legend(ncol=2,fontsize=8)
    ax[1].plot(cyc,s['torque_c'][sel,0],label='Coulomb, частица 1',lw=.8,color='#59a14f')
    ax[1].plot(cyc,s['torque_c'][sel,1],label='Coulomb, частица 2',lw=.8,color='#edc948',ls='--')
    ax[1].set_ylabel('кулоновский момент');ax[1].set_title('Внутренние моменты равны и противоположны');ax[1].grid(alpha=.2);ax[1].legend(fontsize=8)
    wh=s['work_history']
    for ix,label,color in ((10,'RF','#1769aa'),(13,'drag','#d1495b')):
        ax[2].plot(s['cycle'],wh[:,ix],label=f'{label}, 1',color=color,lw=.9)
        ax[2].plot(s['cycle'],wh[:,ix+5],label=f'{label}, 2',color=color,lw=.9,ls='--')
    ax[2].plot(s['cycle'],wh[:,12],label='Coulomb, 1',color='#59a14f',lw=1.3)
    ax[2].plot(s['cycle'],wh[:,17],label='Coulomb, 2',color='#edc948',lw=1.3,ls='--')
    ax[2].set_xlabel('период первой RF');ax[2].set_ylabel('накопленный угловой импульс');ax[2].set_title('Накопленные интегралы моментов; Coulomb ≈ 0 на шкале RF/drag');ax[2].grid(alpha=.2);ax[2].legend(ncol=3,fontsize=8)
    fig.savefig(FIG/'torque_balance_phys0570_C1.png',dpi=180);plt.close(fig)

def plot_coulomb_phase(s):
    wrapped=(s['theta_rel']+np.pi)%(2*np.pi)-np.pi
    fig,ax=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    sc=ax[0].scatter(wrapped,s['torque_c'][:,0],c=s['d'],s=5,cmap='viridis',alpha=.7)
    ax[0].set(xlabel='относительный угол, рад',ylabel='момент Coulomb на частице 1',title='Импульс связи и относительная фаза');ax[0].grid(alpha=.2);fig.colorbar(sc,ax=ax[0],label='d/R')
    ax[1].scatter(wrapped,s['d'],c=s['torque_c'][:,0],s=5,cmap='coolwarm',alpha=.65)
    ax[1].set(xlabel='относительный угол, рад',ylabel='d/R',title='Расстояние при сближениях');ax[1].grid(alpha=.2)
    fig.savefig(FIG/'coulomb_torque_vs_relative_phase_phys0570_C1.png',dpi=180);plt.close(fig)

def plot_poincare(s):
    r=np.maximum(s['r'],1e-30);vr=np.sum(s['xy']*s['vxy'],axis=2)/r
    vt=(s['xy'][:,:,0]*s['vxy'][:,:,1]-s['xy'][:,:,1]*s['vxy'][:,:,0])/r
    fig,ax=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for i,color,label in ((0,'#1769aa','частица 1'),(1,'#d1495b','частица 2')):
        ax[0].scatter(s['r'][:,i],vr[:,i],s=2,alpha=.28,color=color,label=label)
        ax[1].scatter(s['r'][:,i],vt[:,i],s=2,alpha=.28,color=color,label=label)
    ax[0].set(xlabel='r/R',ylabel='радиальная скорость / RΩ₁',title='Радиальное сечение Пуанкаре')
    ax[1].set(xlabel='r/R',ylabel='тангенциальная скорость / RΩ₁',title='Радиус и вращательное движение')
    for a in ax:a.grid(alpha=.25);a.legend()
    fig.suptitle('Стробирование при фиксированной фазе RF₁; RF₂-фаза продолжает вращаться')
    fig.savefig(FIG/'poincare_stroboscopic_phase_space_phys0570_C1.png',dpi=180);plt.close(fig)

def plot_scale():
    fig,ax=plt.subplots(2,1,figsize=(9,8),sharex=True,constrained_layout=True)
    for config,color in [('scaled_refine12_sobol17_f50hz','#4e79a7'),('phys0570','#e15759')]:
        rows=sorted([r for r in scan if r['configuration']==config],key=lambda r:r['coulomb_scale'])
        x=np.array([r['coulomb_scale'] for r in rows]);xplot=np.where(x==0,0.06,x)
        f=np.array([r['frequency_Hz_corrected_from_cycles'] for r in rows]);d=np.array([r['min_separation_over_R'] for r in rows])
        ax[0].plot(xplot,f[:,0],'-o',color=color,label=f'{config}: част. 1');ax[0].plot(xplot,f[:,1],'--s',color=color,label=f'{config}: част. 2')
        ax[1].plot(xplot,d,'-o',color=color,label=config)
    ax[0].axhline(0,color='black',lw=.7);ax[0].set_xscale('log');ax[0].set_ylabel('средняя орбитальная частота, Hz');ax[0].grid(alpha=.25);ax[0].legend(fontsize=8,ncol=2)
    ax[1].set_xscale('log');ax[1].set_xlabel('множитель кулоновской связи sC (0 показан у 0,06)');ax[1].set_ylabel('минимальное расстояние, R');ax[1].grid(alpha=.25);ax[1].legend(fontsize=8)
    fig.savefig(FIG/'coulomb_scale_regime_map.png',dpi=180);plt.close(fig)

def main():
    allenc=[]
    for c in inp['cases']:
        for scale in (0.0,1.0):
            s=read_series(c['id'],scale); save_csv(s); allenc.extend(encounter_list(s))
            if c['id']=='phys0570' and scale==0.0: phys0=s
            if c['id']=='phys0570' and scale==1.0: phys1=s
    with (RES/'stroboscopic_minima.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(allenc[0]) if allenc else ['configuration']);w.writeheader();w.writerows(allenc)
    plot_orbits(phys0,phys1);plot_3d(phys1);plot_close_passages(phys1);plot_moments(phys1);plot_scale();plot_coulomb_phase(phys1);plot_poincare(phys1)
    # Dominant spectral peaks in the late distance and the detrended relative angle.
    sel=phys1['cycle']>=phys1['cycle'][-1]-8192; d=phys1['d'][sel]; rel=phys1['theta_rel'][sel]
    f_d,p_d=periodogram(d-np.mean(d)); t=phys1['cycle'][sel]; rel_res=rel-np.polyval(np.polyfit(t,rel,1),t); f_r,p_r=periodogram(rel_res-np.mean(rel_res))
    with (RES/'relative_motion_spectrum.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['cycles^-1','distance_power','detrended_relative_angle_power']);w.writerows(zip(f_d,p_d,np.interp(f_d,f_r,p_r)))
    fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
    ax.semilogy(f_d[1:],p_d[1:],label='d(t), demeaned');ax.semilogy(f_r[1:],p_r[1:],label='relative angle after linear trend removal')
    ax.set(xlabel='частота на RF-строб (циклов⁻¹)',ylabel='мощность',title='Спектр стробоскопического относительного движения phys0570, sC=1');ax.grid(alpha=.25);ax.legend()
    fig.savefig(FIG/'relative_motion_spectrum_phys0570_C1.png',dpi=180);plt.close(fig)
    print('saved',len(allenc),'relative-angle passage windows; plots:',*[p.name for p in FIG.glob('*.png')])

if __name__=='__main__': main()
