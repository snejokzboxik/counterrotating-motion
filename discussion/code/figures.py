from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
R=Path(__file__).resolve().parents[2]; O=R/'discussion'; D=lambda p:json.loads((R/p).read_text(encoding='utf-8'))
t=D('data/m3_nominal_torque_decomposition.json'); v=D('data/three_parameter_criterion_validation.json'); rad=D('data/radial_localization_validation.json'); drag=D('data/m3_schiller_naumann_vs_linear.json')
for p in ['data/m3_nominal_torque_decomposition.json','data/three_parameter_criterion_validation.json','data/radial_localization_validation.json','data/m3_schiller_naumann_vs_linear.json']:
 (O/'data'/Path(p).name).write_bytes((R/p).read_bytes())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600,'svg.fonttype':'none'})
def save(f,n):
 for e in ['pdf','svg','png']: f.savefig(O/'figures'/f'FIG{n:02d}.{e}',bbox_inches='tight')
 plt.close(f)
f,ax=plt.subplots(figsize=(6.8,3.5)); th=np.linspace(0,2*np.pi,400)
for r,c,sg in [(.555,'#167d91',1),(.751,'#b64d36',-1)]:
 ax.plot(r*np.cos(th),r*np.sin(th),color=c); a=.65 if sg>0 else 3.8; b=a+sg*.3; ax.annotate('',xy=(r*np.cos(b),r*np.sin(b)),xytext=(r*np.cos(a),r*np.sin(a)),arrowprops=dict(arrowstyle='->',color=c,lw=2))
ax.text(.08,.3,r'$\bar\omega_1>0$'); ax.text(-.72,-.55,r'$\bar\omega_2<0$'); ax.plot(0,0,'ko'); ax.text(.03,.03,'trap axis z'); ax.text(-.92,-.94,'Common RF: k=+2 and k=-4',fontsize=9); ax.set(xlim=(-1,1),ylim=(-1,1),aspect='equal',xlabel='x/R',ylabel='y/R',title='Schematic mean orbits (not RF trajectory)');save(f,1)
A,B,nu,beta,*_=t['parameters']; rr=np.linspace(.03,.9,450); tables={}
f,axs=plt.subplots(1,2,figsize=(7,3.2),sharey=True)
for ax,p in zip(axs,t['particles']):
 a=p['alpha']; l=p['lambda']; k1=1+a*beta;k2=nu**2+a*beta;d1=k1*k1+l*l;d2=k2*k2+(l*nu)**2
 dc=a*beta*rr; rf1=-4*a*a*A*A*k1/d1*rr;rf2=-48*a*a*B*B*k2/d2*rr**5
 for ar,lab in [(dc,'DC'),(rf1,'RF +2'),(rf2,'RF -4'),(dc+rf1+rf2,'Sum')]:ax.plot(rr,ar,label=lab)
 ax.axhline(0,c='gray',lw=.6); ax.axvline(v['nominal']['particles'][p['index']-1]['M1_radius_R'],c='black',ls=':',label='M1 root');ax.set(xlabel='r/R',title=f"Particle {p['index']}",ylim=(-.018,.016)); tables[str(p['index'])]={'r_R':rr.tolist(),'DC':dc.tolist(),'RF_plus2':rf1.tolist(),'RF_minus4':rf2.tolist()}
axs[0].set_ylabel('Radial acceleration / (R Omega1^2)');axs[1].legend(fontsize=8);f.tight_layout();save(f,2)
f,axs=plt.subplots(2,2,figsize=(7,5),sharex=True)
for col,p in enumerate(t['particles']):
 a=p['alpha'];l=p['lambda'];d1=(1+a*beta)**2+l*l;d2=(nu**2+a*beta)**2+(l*nu)**2
 for row,(c1,c2,label) in enumerate([(4,48,'Net slow torque, omega=0'),(8,64,'Total RF torque, omega=0')]):
  ax=axs[row,col];pos=c1*a*a*A*A*l*rr**2/d1;neg=-c2*a*a*B*B*l*nu*rr**6/d2
  ax.plot(rr,pos,label='+2');ax.plot(rr,neg,label='-4');ax.plot(rr,pos+neg,c='black',label='Sum');ax.axhline(0,c='gray',lw=.6);ax.set(title=f"{label}\nParticle {p['index']}",ylim=(-.0013,.001));ax.axvline(p['leading_critical_radii']['zero_net_slow_torque_r_over_R' if row==0 else 'zero_total_RF_torque_r_over_R'],c='gray',ls=':');ax.set_xlabel('r/R')
axs[0,0].set_ylabel('Torque / (mi R^2 Omega1^2)');axs[1,0].set_ylabel('Torque / (mi R^2 Omega1^2)');axs[0,1].legend(fontsize=8);f.tight_layout();save(f,3)
f,axs=plt.subplots(1,2,figsize=(7,3.6)); x=np.linspace(.05,.98,300); n=v['nominal'];
for ax,mode in zip(axs,['alpha','nu']):
 xx=x if mode=='alpha' else np.linspace(1.8,4.5,300);low=np.maximum(1,xx*(1+n['nu'])) if mode=='alpha' else np.maximum(1,n['alpha2']*(1+xx));hi=np.full_like(xx,1+n['nu']) if mode=='alpha' else 1+xx
 ax.fill_between(xx,low,hi,color='#dcefe5',label='M0 torque-sign band');ax.plot(xx,low,c='#28664a');ax.plot(xx,hi,c='#28664a');ax.set(xlabel='alpha2' if mode=='alpha' else 'nu',ylabel='Pi',ylim=(.8,6.6));ax.scatter(n['alpha2'] if mode=='alpha' else n['nu'],n['Pi'],marker='*',s=100,c='black',label='Nominal')
 for row in v['sparse_M3_continuation_comparison']['rows']:
  if mode=='alpha' and abs(row['nu']-n['nu'])>1e-8:continue
  mismatch=not row['M0_sign_match_M3'];ax.scatter(row['alpha2'] if mode=='alpha' else row['nu'],row['Pi'],marker='x' if mismatch else 'o',s=55 if mismatch else 12,c='red' if mismatch else '#337599',zorder=4)
axs[1].plot([],[],'rx',label='M0/M3 mismatch');axs[0].legend(fontsize=8);axs[1].legend(fontsize=8);f.tight_layout();save(f,4)
f,axs=plt.subplots(1,2,figsize=(7,3.4));xx=np.arange(2)
for j,(key,lab) in enumerate([('leading_radius_R','M1'),('M3_radius_R','M3'),('full_RF_RK4_240k_radius_R','Full RF')]): axs[0].scatter(xx+(j-1)*.12,[p[key] for p in rad['nominal']],label=lab)
axs[0].set(xticks=xx,xticklabels=['Particle 1','Particle 2'],ylabel='Mean radius / R');axs[0].legend(fontsize=8)
for run in drag['paired_runs']:axs[1].scatter(xx,run['root']['omega_over_Omega1'],label='SN drag' if run['sn_scale'] else 'Linear drag')
axs[1].axhline(0,c='gray',lw=.6);axs[1].set(xticks=xx,xticklabels=['Particle 1','Particle 2'],ylabel='omega / Omega1');axs[1].legend(fontsize=8);f.tight_layout();save(f,5)
(O/'data/radial_curves.json').write_text(json.dumps(tables,allow_nan=False),encoding='utf-8')
