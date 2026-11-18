from pathlib import Path
import numpy as np,csv
root=Path(__file__).parent
with (root/'summary.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['file','row','N','steps_per_RF','RF_periods','retained','mode','C','epsilon','lambda_base','Re_star','mass_ratio','charge_ratio','drag_ratio','max_radius','min_pair_distance','max_Re','mixed_three_windows','last_window_freq_Hz','last_window_mean_r','min_r_all_windows'])
 for path in sorted(root.glob('*.npz')):
  d=np.load(path);t=d['o'][:,:,:,0]/(2*np.pi);n=d['ys'].shape[1];steps=int(d['steps']);periods=int(d['periods']);good=d['mi'].min(1)>.02
  mixed=(d['st']==1)&((t.min(1)>.05)&good).any(1)&((t.max(1)<-.05)&good).any(1)
  for i,p in enumerate(d['p']):
   if path.name=='targeted.npz': c,eps,gr,re,spin=p;mode=0;g=.003;mr=qr=1
   else:c,g,mode,re,eps,phase,spin,mr,qr,gr=p
   ok=int(d['st'][i]);freq=(t[i,-1]*1e5/(periods*.2)).tolist() if ok else [];r=(d['o'][i,-1,:,1]/(steps*periods*.2)).tolist() if ok else [];mi=d['mi'][i].min(0).tolist() if ok else []
   w.writerow([path.name,i,n,steps,periods,ok,int(mode),c,eps,g,re,mr,qr,gr,*d['rs'][i],int(mixed[i]),freq,r,mi])
