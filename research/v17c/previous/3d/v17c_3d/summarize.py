import numpy as np,csv,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
rows=[]
def add(stage,data,inputs,idx):
 st=np.atleast_1d(data['st']);s=data['s'];ct=data['ct'];mi=data['mi'];rs=data['rs'];periods=int(data['periods']);steps=int(data['steps']);n=len(inputs['ms'][idx[0]])
 if s.ndim==3:s=s[None];ct=ct[None];mi=mi[None];rs=rs[None]
 for k,i in enumerate(idx):
  p=inputs['ps'][i];turns=s[k,:,:,0]/(2*np.pi);freq=turns*100000*steps/np.maximum(ct[k,:,None],1);good=mi[k].min(0)>.015;pos=(turns.min(0)>1)&good;neg=(turns.max(0)<-1)&good
  rows.append([stage,i,n,periods,steps,*p,*inputs['meta'][i],int(st[k]),*rs[k],float(freq[-1,:n//2].mean()),float(freq[-1,n//2:].mean()),int(st[k] and pos.any() and neg.any())])
D=np.load('inputs.npz')
for i in range(0,96,12):add('scout',np.load(f'scout_{i}.npz'),D,list(range(i,i+12)))
for steps in [96,192]:
 a=np.load(f'confirm_{steps}.npz');add('confirm',a,D,a['idx'])
for i in [59,83,80,81,82]:add('trace',np.load(f'trace_{i}.npz'),D,[i])
E=np.load('extra_inputs.npz')
for i in range(0,32,8):add('extra',np.load(f'extra_{i}.npz'),E,list(range(i,i+8)))
with open('summary.csv','w',newline='') as f:
 w=csv.writer(f);w.writerow('stage index N periods steps k C eps axial_mode wz d chirality_mode Re_star mass_ratio charge_ratio drag_ratio seed retained max_r max_abs_z min_pair max_Re group_1_frequency_Hz_last group_2_frequency_Hz_last opposite_each_window_gt_1_turn'.split());w.writerows(rows)
a=np.load('trace_83.npz');tr=a['tr'];fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d');
for j in range(6):
 c='tab:blue' if j<3 else 'tab:orange';ax.plot(*tr[-300:,j,:3].T,color=c,alpha=.65,lw=1);ax.scatter(*tr[-1,j,:3],color=c,s=35,label=('Lower group' if j==0 else 'Upper group' if j==3 else None))
ax.set(xlabel='x/R',ylabel='y/R',zlabel='z/R',xlim=(-.4,.4),ylim=(-.4,.4),zlim=(-.8,.8),title='Axially separated counter-rotating groups (case 83)');ax.legend();ax.set_box_aspect((1,1,1.5));fig.tight_layout();fig.savefig('trajectory_3d.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5));ang=np.unwrap(np.arctan2(tr[:,:,1],tr[:,:,0]),axis=0);turns=(ang-ang[0])/(2*np.pi);t=np.arange(len(tr))*5;
for j in range(6):ax.plot(t,turns[:,j],color='tab:blue' if j<3 else 'tab:orange',alpha=.8,label='Lower group' if j==0 else 'Upper group' if j==3 else None)
ax.set(xlabel='RF periods',ylabel='Accumulated revolutions',title='Counter-rotation in axial chirality model');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig('winding.png',dpi=170);plt.close(fig)
print('rows',len(rows),'retained',sum(r[17] for r in rows))
