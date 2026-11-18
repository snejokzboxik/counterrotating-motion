"""Generate compact figures for the independent geometry and dynamics audit."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from single_particle import RESULTS

def plot_torque():
    with (RESULTS/'averaged_torque_map.csv').open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    names=['6_ABC_moderate','6_ABC_thin','9_ABC_moderate','12_ABC_moderate']
    labels=['G1: 6 электродов','G2: 6 тонких','G3: 9 электродов','G5: 12 электродов']
    for n,l in zip(names,labels):
        q=[x for x in rows if x['name']==n];xx=np.array([float(x['r_over_R']) for x in q]);yy=np.array([float(x['torque_dimensionless']) for x in q]);
        scale=np.max(np.abs(yy));ax.plot(xx,yy/scale,label=l)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='r/R',ylabel='нормированный момент центра',title='Усреднённое азимутальное воздействие полного спектра')
    ax.legend(frameon=False);fig.savefig(RESULTS/'averaged_torque_sign_map.png',dpi=180)

def plot_fem():
    dat=json.loads((RESULTS.parent/'input'/'independent_fem_audit.json').read_text(encoding='utf-8'))
    fig,ax=plt.subplots(figsize=(8,4.8),constrained_layout=True)
    for g in dat:
        n=g['name'];h=np.array([x['mesh_h'] for x in g['levels']]);
        e=np.array([x['counter_abs_error']/abs(complex(*x['counter_MFS'])) for x in g['levels']])
        ax.loglog(h,e,'o-',label=n.replace('_ABC_',' '))
    ax.set(xlabel='характерный размер FEM-элемента h/D',ylabel='относительная ошибка |A−n|',title='Сходимость независимой P1 FEM-проверки')
    ax.invert_xaxis();ax.legend(frameon=False);fig.savefig(RESULTS/'fem_coefficient_convergence.png',dpi=180)

def plot_rejected_orbits():
    fig,axs=plt.subplots(3,2,figsize=(10,11),constrained_layout=True)
    cases=[('G1 внешний старт, вращение −','G1_outer_negative_8000RF.npz'),
           ('G1 внутренний старт, вращение +','G1_inner_positive_8000RF.npz')]
    for j,(lab,file) in enumerate(cases):
        d=np.load(RESULTS/file);s=d['state'];t=d['rf_periods'];r=np.hypot(s[:,0],s[:,1]);ang=d['theta']
        ax=axs[0,j];use=t<=200
        ax.plot(s[use,0],s[use,1],lw=.35);ax.set_aspect('equal');ax.set(xlabel='x/R',ylabel='y/R');ax.set_title(lab+' · 0–200 RF',fontsize=10)
        ax=axs[1,j];ax.plot(t,r,lw=.8);ax.axhline(.05,color='red',ls='--',lw=.8,label='r/R = 0.05')
        ax.set(xlabel='RF-период',ylabel='r/R',title='Радиус за 8000 RF');ax.legend(frameon=False,fontsize=8)
        ax=axs[2,j];cut=int(np.flatnonzero(r<.05)[0]) if np.any(r<.05) else len(r)
        ax.plot(t[:cut],(ang[:cut]-ang[0])/(2*np.pi),lw=.8);ax.axvline(t[cut-1],color='red',ls='--',lw=.8)
        ax.set(xlabel='RF-период',ylabel='развёрнутые обороты',title='Угол до r/R < 0.05')
    fig.savefig(RESULTS/'rejected_counterrotation_candidates.png',dpi=180)

def plot_positive_robustness():
    fig,axs=plt.subplots(3,1,figsize=(8,10),constrained_layout=True)
    nom=np.load(RESULTS/'G2_positive_50000RF.npz');sn=nom['state'][:8001];tn=nom['rf_period'][:8001];rn=np.hypot(sn[:,0],sn[:,1])
    angles=[nom['theta'][:8001]]
    cases=[('номинал',tn,rn,angles[0],sn)]
    for label,file in [('r0 +1%','G2_radius_plus_1pct_8000RF.npz'),('alpha +0,5%','G2_alpha_plus_0p5pct_8000RF.npz')]:
        d=np.load(RESULTS/file);s=d['state'];r=np.hypot(s[:,0],s[:,1]);cases.append((label,d['rf_period'],r,d['theta'],s))
    for label,t,r,ang,s in cases:
        axs[1].plot(t,r,lw=.8,label=label)
        cut=int(np.flatnonzero(r<.05)[0]) if np.any(r<.05) else len(r)
        axs[2].plot(t[:cut],(ang[:cut]-ang[0])/(2*np.pi),lw=.8,label=label)
    axs[1].axhline(.05,color='black',ls='--',lw=.8);axs[1].set(xlim=(0,8000),xlabel='RF-период',ylabel='r/R',title='Номинал и малые возмущения');axs[1].legend(frameon=False)
    for label,t,r,ang,s in cases:
        use=t<=200;axs[0].plot(s[use,0],s[use,1],lw=.5,label=label)
    axs[0].set_aspect('equal');axs[0].set(xlabel='x/R',ylabel='y/R',title='Траектория XY, первые 200 RF');axs[0].legend(frameon=False)
    axs[2].set(xlabel='RF-период',ylabel='развёрнутые обороты',title='Угол обрезан при r/R < 0,05');axs[2].legend(frameon=False)
    fig.savefig(RESULTS/'G2_positive_robustness.png',dpi=180)

def main():
    plot_torque();plot_fem();plot_rejected_orbits();plot_positive_robustness()
    print('wrote three figures to',RESULTS)

if __name__=='__main__':main()
