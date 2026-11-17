"""Publication-style V14 figures from inspectable trajectory and orbit tables."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_ideal_pilot import OUT,ALPHA,BETA,KAPPA
from three_phase_trap.v14_dynamics import effective_potential

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'figure.dpi':150,'savefig.dpi':220})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def compare():
    ideal=[json.loads(s) for s in (OUT/'ideal_pilot_checkpoint.jsonl').read_text().splitlines() if s.strip()]
    fem=read(OUT/'fem_so_paired_summary.csv')
    names=['Q','SO_equal_harmonic','SO_equal_rms','RO']
    labels=['Q','Q+SO\nравная гармоника','Q+SO\nравное ⟨E²⟩','Q+RO']
    vals=[]
    for field in ('Идеальная гармоника','Полное FEM h50'):
        v=[]
        for name in names:
            if field.startswith('Идеальная'):
                v.append(sum(r['model']==name and r['status']=='retained' for r in ideal)/256)
            else:
                v.append(float(next(r['P_ret'] for r in fem if r['phase_deg']=='average' and r['model']==name)))
        vals.append(v)
    x=np.arange(4);fig,ax=plt.subplots(figsize=(8.4,4.6))
    ax.bar(x-.19,vals[0],.36,label='Идеальная гармоника',color='#5a829d')
    ax.bar(x+.19,vals[1],.36,label='Полное FEM h50 P4',color='#ba7255')
    ax.set_xticks(x,labels);ax.set_ylim(0,.4);ax.set_ylabel('Доля удержания на 100 RF')
    ax.set_title('A9, μ=1: общий физический 4D ансамбль, 256 стартов на модель')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,'q_so_ro_ideal_fem_N100')

def phase():
    rows=read(OUT/'phase_ideal_summary.csv')
    x=np.array([int(r['delta_deg']) for r in rows]);y=np.array([float(r['P_ret']) for r in rows])
    lo=np.array([float(r['ci95_low']) for r in rows]);hi=np.array([float(r['ci95_high']) for r in rows])
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,color='#3f6d8b',label='Идеальная, 4×16×4 RF-фазы')
    femfile=OUT/'phase_fem_summary.csv'
    if femfile.exists():
        fem=read(femfile)
        ax.plot([int(r['delta_deg']) for r in fem],[float(r['P_ret']) for r in fem],
                's-',color='#b85e44',label='Полное FEM h50, те же старты')
    ax.set(xticks=x,xlabel='Относительная фаза δ, градусы',ylabel='Доля удержания на 100 RF',
           title='Фазовая чувствительность вращающегося октуполя')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,'ro_phase_retention')

def potential():
    grid=np.linspace(-.98,.98,251);X,Y=np.meshgrid(grid,grid)
    fig,axes=plt.subplots(1,3,figsize=(11.3,3.8),sharex=True,sharey=True)
    orbits=json.loads((OUT/'phase_ideal_orbits.json').read_text())
    for ax,deg in zip(axes,(0,45,90)):
        V=effective_potential(X,Y,ALPHA,BETA,KAPPA,'RO',math.radians(deg))
        V=np.ma.masked_where(X*X+Y*Y>1,V)
        ax.contourf(X,Y,V,levels=np.linspace(-.0001,.015,40),cmap='viridis',extend='both')
        ax.contour(X,Y,V,levels=[-.00002,0,.0001,.0005,.002],colors='white',linewidths=.5,alpha=.7)
        th=np.linspace(0,2*math.pi,401);ax.plot(np.cos(th),np.sin(th),'k',lw=.8)
        for r in orbits:
            if r['delta_deg']==deg:ax.plot(r['state'][0],r['state'][1],'r*',markersize=9)
        ax.set_aspect('equal');ax.set_title(f'δ={deg}°');ax.set_xlabel('x/R');ax.set_xlim(-1,1);ax.set_ylim(-1,1)
    axes[0].set_ylabel('y/R')
    fig.suptitle('Усреднённый потенциал A9; звёзды — найденные точные 1-RF орбиты')
    save(fig,'effective_potential_phase')

def strobe():
    path=OUT/'stroboscopic_states.csv'
    if not path.exists():return
    rows=read(path)
    fig,axes=plt.subplots(2,3,figsize=(10.8,6.3),sharex='col')
    for ci,model in enumerate(('Q','SO','RO')):
      for ri,field in enumerate(('ideal','fem_h50')):
        ax=axes[ri,ci]
        selected=[r for r in rows if r['field']==field and r['model']==model and r['start_id']=='near_RO_ideal']
        ax.plot([float(r['y']) for r in selected],[float(r['vy']) for r in selected],'.',ms=2,color='#4f7592')
        ax.set_title(model+' / '+field);ax.grid(alpha=.15)
        if ri==1:ax.set_xlabel('y/R')
        if ci==0:ax.set_ylabel("v′y")
    fig.suptitle('Стробоскопическая проекция (y, v′y), одна точка на RF-период; общая начальная 4D точка')
    save(fig,'stroboscopic_projections')

def lifetime():
    path=OUT/'fem_long_checkpoint.jsonl'
    if not path.exists():return
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    if not any(r['horizon']==1000 for r in rows):return
    names=('Q','RO','SO_equal_harmonic','SO_equal_rms')
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for model in names:
      val=[]
      for h in (100,300,1000):
        if h==100:
            rec=read(OUT/'fem_so_paired_summary.csv')
            val.append(float(next(r['P_ret'] for r in rec if r['phase_deg']=='average' and r['model']==model)))
        else:
            part=[r for r in rows if r['horizon']==h and r['model']==model]
            val.append(sum(r['status']=='retained' for r in part)/len(part))
      ax.plot([100,300,1000],val,'o-',label=model)
    ax.set(xscale='log',xlabel='Горизонт, RF-периоды',ylabel='Доля удержания',title='Продолжение общих h50 FEM траекторий без сброса RF-фазы')
    ax.grid(alpha=.2);ax.legend(frameon=False)
    save(fig,'fem_survival_vs_time')

def mass():
    rows=read(OUT/'fem_mass_so_summary.csv')
    base=read(OUT/'fem_so_paired_summary.csv')
    for model in ('Q','SO_equal_harmonic','RO'):
        p=float(next(r['P_ret'] for r in base if r['phase_deg']=='average' and r['model']==model))
        rows.append(dict(mu=1.,model=model,P_ret=p,A4_m4_s2=p*167.38913929608813))
    fig,axes=plt.subplots(1,2,figsize=(9.8,4.2))
    colors={'Q':'#777777','SO_equal_harmonic':'#4d8b68','RO':'#b85e44'}
    labels={'Q':'Q','SO_equal_harmonic':'Q+SO','RO':'Q+RO'}
    for model in ('Q','SO_equal_harmonic','RO'):
        part=sorted((r for r in rows if r['model']==model),key=lambda r:float(r['mu']))
        x=[float(r['mu']) for r in part]
        axes[0].plot(x,[float(r['P_ret']) for r in part],'o-',label=labels[model],color=colors[model])
        axes[1].plot(x,[float(r['A4_m4_s2']) for r in part],'o-',label=labels[model],color=colors[model])
    axes[0].set(xlabel='Нормированная масса μ',ylabel='P удержания, 100 RF',title='Полный h50 P4 FEM')
    axes[1].set(xlabel='Нормированная масса μ',ylabel='A₄, м⁴·с⁻²',title='A₄ = P × Vol(D)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle('A9: выборочные массовые срезы; соединение служит только ориентиром')
    save(fig,'mass_retention_and_A4')

if __name__=='__main__':
    compare();phase();potential();strobe();lifetime()
    if (OUT/'fem_mass_so_summary.csv').exists():mass()
