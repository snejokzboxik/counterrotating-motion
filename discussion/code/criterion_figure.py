from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
root=Path(__file__).resolve().parents[2]
v=json.loads((root/'discussion/data/three_parameter_criterion_validation.json').read_text(encoding='utf-8'));n=v['nominal']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':600,'svg.fonttype':'none'})
f,axs=plt.subplots(1,2,figsize=(8,5.1))
for ax,mode in zip(axs,['alpha','nu']):
 x=np.linspace(.05,.98,300) if mode=='alpha' else np.linspace(1.8,4.5,300)
 lo=np.maximum(1,x*(1+n['nu'])) if mode=='alpha' else np.maximum(1,n['alpha2']*(1+x))
 hi=np.full_like(x,1+n['nu']) if mode=='alpha' else 1+x
 ax.fill_between(x,lo,hi,color='#dcefe5');ax.plot(x,lo,c='#28664a');ax.plot(x,hi,c='#28664a')
 ax.set_ylim(.8,6.6)
 ax.set_ylabel(r'$\Pi=\beta/(4A^2)$'+'\nDC / RF1 confinement parameter')
 ax.set_xlabel((r'$\alpha_2=(q_2/m_2)/(q_1/m_1)$'+'\nRelative charge-to-mass ratio') if mode=='alpha' else (r'$\nu=f_{\mathrm{RF2}}/f_{\mathrm{RF1}}$'+'\nRF frequency ratio'))
 ax.set_title((r'(a) Fixed $\nu=3.21968$') if mode=='alpha' else r'(b) Fixed $\alpha_2=0.378146$',fontsize=11)
 for row in v['sparse_M3_continuation_comparison']['rows']:
  if mode=='alpha' and abs(row['nu']-n['nu'])>1e-8:continue
  mismatch=not row['M0_sign_match_M3']
  ax.scatter(row['alpha2'] if mode=='alpha' else row['nu'],row['Pi'],marker='x' if mismatch else 'o',s=52 if mismatch else 16,c='#b32323' if mismatch else '#337599',zorder=4)
 ax.scatter(n['alpha2'] if mode=='alpha' else n['nu'],n['Pi'],marker='*',s=110,c='black',zorder=5)
 ax.grid(alpha=.15)
handles=[Patch(facecolor='#dcefe5',label='Opposite slow-torque signs (M0)'),Line2D([],[],marker='o',ls='',color='#337599',label='Archived M3 sample'),Line2D([],[],marker='*',ls='',color='black',label='Nominal pair'),Line2D([],[],marker='x',ls='',color='#b32323',label='M0 / M3 sign disagreement')]
f.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False,fontsize=9)
f.text(.5,.025,'Larger Pi: stronger DC or weaker RF1. Shading is not a full-RF stability region.',ha='center',fontsize=9)
f.subplots_adjust(left=.12,right=.98,top=.91,bottom=.35,wspace=.43)
for ext in ['pdf','svg','png']:f.savefig(root/f'discussion/figures/FIG04.{ext}',bbox_inches='tight')
