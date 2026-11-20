from search import *
from mfs import transfer

def geom(N,p=32):
 k,T,E=transfer(N,.05,1.10,p,48);js=[];ks=[];cs=[];data=[]
 for j,k0 in enumerate([2,-4]):
  v=np.exp(2j*np.pi*k0*np.arange(N)/N);h=T@v;lead=h[k==k0][0];print('geom',N,p,k0,lead,max(abs(E@v)),flush=True)
  for kk,c in zip(k,h):
   if abs(c)>1e-10:
    js.append(j);ks.append(kk);cs.append(c/lead);data.append((N,p,j,kk,c.real,c.imag,abs(c),abs(lead),max(abs(E@v))))
 return np.array(js),np.array(ks),np.array(cs),data
if __name__=='__main__':
 a=np.load(P/'ideal.npz');cases=[24,25,26,30,31,33,34,35];allg={};data=[]
 for N in [24,36]:
  j,k,c,d=geom(N);allg[N]=(j,k,c);data+=d
 with open(P/'mfs_coefficients.csv','w') as f:
  w=csv.writer(f);w.writerow(['N','p','family','k','re','im','abs','leading_abs','boundary_error']);w.writerows(data)
 def go(task):
  N,i=task;y=a['init'][i];p=a['param'][i];j,k,c=allg[N];o,ok=run(y,p,j,k,c,3500,48,2);w,m,M=stats(o);print('result',N,i,ok,np.round(w,3).tolist(),np.round(m,3),np.round(M,3),flush=True);return (N,i,o,ok)
 with ThreadPoolExecutor(max_workers=4) as ex: out=list(ex.map(go,[(N,i) for N in allg for i in cases]))
 np.savez_compressed(P/'multielectrode.npz',**{f'N{N}_case{i}':o for N,i,o,ok in out})
