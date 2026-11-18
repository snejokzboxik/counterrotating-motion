import numpy as np,json
from scipy.integrate import solve_ivp
from model import rhs
D=np.load('inputs.npz');A=np.load('confirm_192.npz');i=83;p,m,q,g=[D[k][i] for k in ['ps','ms','qs','gs']];y=A['fin'][list(A['idx']).index(i)]
def energy(y,p,m,q,t):
 k,C,e,az,w,d,mode,re=p;z=y[:,2];u=y[:,0]+1j*y[:,1];r2=np.abs(u)**2;a=np.exp(-1j*t);b=a.conjugate();aa=a if mode==0 else (a.real if mode==1 else (1+z/d)*a/2+(1-z/d)*b/2);E=np.sum(q*k*np.real(u**4*aa+e*u**8*b));
 if az==0:E+=np.sum(q*w*w/2*(z*z-r2/2))
 elif az==1:E+=np.sum(m*w*w*(z*z-d*d)**2/(8*d*d))
 else:E+=np.sum(q*w*w/(8*d*d)*(z**4-3*z*z*r2+3/8*r2**2-2*d*d*(z*z-r2/2)))
 for i in range(len(y)):
  for j in range(i):E+=C*q[i]*q[j]/np.linalg.norm(y[i,:3]-y[j,:3])
 return E
rng=np.random.default_rng(878);err=[]
for az in range(3):
 for mode in range(3):
  pp=p.copy();pp[3]=az;pp[6]=mode;pp[2]=.32945;yy=y.copy();yy[:,:3]+=rng.normal(0,.01,yy[:,:3].shape);f=rhs(yy,.7,pp,m,q,np.zeros_like(g))[:,3:]*m[:,None];grad=np.zeros_like(f)
  for ii in range(len(yy)):
   for jj in range(3):
    yp=yy.copy();ym=yy.copy();yp[ii,jj]+=1e-6;ym[ii,jj]-=1e-6;grad[ii,jj]=(energy(yp,pp,m,q,.7)-energy(ym,pp,m,q,.7))/2e-6
  err.append(float(np.max(np.abs(f+grad))))
print('max force error',max(err),flush=True)
# independently integrate continuation of candidate; sample at 4 points per RF for winding and retain checks
T=2*np.pi*1000;ts=np.linspace(0,T,4001);sol=solve_ivp(lambda t,Y:rhs(Y.reshape(-1,6),t,p,m,q,g).ravel(),[0,T],y.ravel(),method='DOP853',rtol=1e-10,atol=1e-12,max_step=.35,t_eval=ts);tr=sol.y.T.reshape(-1,len(y),6);ang=np.unwrap(np.arctan2(tr[:,:,1],tr[:,:,0]),axis=0);turns=(ang[-1]-ang[0])/(2*np.pi);r=np.linalg.norm(tr[:,:,:2],axis=2);print('DOP',sol.success,turns, 'rmax',r.max(), 'zmax',abs(tr[:,:,2]).max(),flush=True);np.savez('dop853_83.npz',t=sol.t,tr=tr,turns=turns,success=sol.success,idx=i);json.dump({'max_force_gradient_abs_error':max(err),'DOP853_success':bool(sol.success),'turns':turns.tolist(),'rmax':float(r.max()),'zmax':float(abs(tr[:,:,2]).max())},open('validation.json','w'),indent=2)
