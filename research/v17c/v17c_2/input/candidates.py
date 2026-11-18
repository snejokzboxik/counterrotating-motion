from scan import model, write, OUT
import numpy as np
from scipy.linalg import solve
from scipy.optimize import brentq
import json

def detail(name,N,m,a,gap,p=64):
 B=1+a+gap; R=1-a; n=N-m
 row=model(N,m,a,B,p); row['name']=name
 th=np.arange(N)*2*np.pi/N; ct=np.exp(1j*th); ang=th[:,None]+np.arange(p)*2*np.pi/p
 z=(ct[:,None]+a*np.exp(1j*ang)).ravel(); s=(ct[:,None]+.65*a*np.exp(1j*ang)).ravel()
 G=np.log(np.abs(B*B-z[:,None]*s[None,:].conj())/(B*np.abs(z[:,None]-s[None,:])))
 q=solve(G,np.repeat(np.exp(1j*m*th),p))
 def c(k):
  f=(s**(-abs(k))-(s.conj()/B**2)**abs(k))*R**abs(k)/(2*abs(k)); return np.dot(q,f if k>0 else f.conj())
 modes=[m+N*k for k in range(-25,26) if abs(m+N*k)<=160]; vals=np.array([c(k) for k in modes]); modes=np.array(modes)
 row['torque_cross_pair_R']=(np.sqrt((m-1)/(n-1))/row['F_R'])**(1/(n-m))
 def torque(x):
  return np.sum(np.sign(modes)*modes**2*(abs(modes)-1)*abs(vals)**2*x**(2*abs(modes)-2))
 grid=np.linspace(.1,.98,177); roots=[]
 for x,y in zip(grid[:-1],grid[1:]):
  if torque(x)*torque(y)<0: roots.append(brentq(torque,x,y))
 row['torque_cross_all_R']=roots[0] if roots else float('nan')
 # exact phasor-field on rod surfaces, sampling at 512 angles per electrode
 zz=(ct[:,None]+a*np.exp(1j*(th[:,None]+np.arange(512)*2*np.pi/512))).ravel()
 gmax=0.
 for zi in np.array_split(zz,32):
  g=1/np.conj(zi[:,None]-B**2/s[None,:].conj())-1/np.conj(zi[:,None]-s[None,:])
  ex=-(g.real@q);ey=-(g.imag@q)
  peak=np.sqrt((abs(ex)**2+abs(ey)**2+abs(ex**2+ey**2))/2)
  gmax=max(gmax,float(peak.max()))
 row['surface_Emax_U0_over_R']=gmax*R
 row['surface_Emax_V_per_m_at_250V_R8mm']=gmax*R*250/.008
 return row,{'name':name,'N':N,'m':m,'a_D':a,'gap_D':gap,'R_D':R,'coefficients':[[int(k),float(v.real),float(v.imag)] for k,v in zip(modes,vals)]}

if __name__=='__main__':
 specs=[('6_ABC_moderate',6,2,.05,.05),('6_ABC_thin',6,2,.01,.02),('9_ABC_moderate',9,3,.05,.05),('9_ABC_thin',9,3,.01,.02),('12_ABC_moderate',12,4,.05,.05),('12_ABC_thin',12,4,.01,.02),('12_ABC_extreme',12,4,.002,.003),('9_phase160_moderate',9,4,.05,.05),('9_phase160_thin',9,4,.01,.02),('10_phase144_moderate',10,4,.05,.05),('10_phase144_thin',10,4,.01,.02),('12_phase150_moderate',12,5,.05,.05)]
 rows=[]; coeff=[]
 for spec in specs:
  row,cs=detail(*spec);rows.append(row);coeff.append(cs)
 write('candidates.csv',rows)
 (OUT/'harmonics.json').write_text(json.dumps(coeff,indent=2))
 for r in rows:print(r['name'], 'A-/A+',round(r['ratio'],5),'A+',round(r['Aplus'],5),'F cross',round(r['cross_R'],4),'torque pair',round(r['torque_cross_pair_R'],4),'all torque',round(r['torque_cross_all_R'],4),'Emax V/m',round(r['surface_Emax_V_per_m_at_250V_R8mm']),'err',r['boundary_error'])
