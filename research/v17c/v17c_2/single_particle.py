"""Full-spectrum 2D single-particle RF dynamics for archived geometries."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent
INPUT=HERE/'input'/'harmonics.json'
RESULTS=HERE/'results'

@njit(cache=True)
def acceleration(x,y,vx,vy,t,modes,coeff,amp,lam):
    u=x+1j*y; maxn=0
    for k in modes:
        if abs(k)>maxn:maxn=abs(k)
    powers=np.empty(maxn+1,np.complex128);powers[0]=1.+0j
    for n in range(1,maxn+1):powers[n]=powers[n-1]*u
    phase=np.cos(t)-1j*np.sin(t); ax=0.;ay=0.
    for j in range(len(modes)):
        k=modes[j]; n=abs(k);w=n*coeff[j]*(powers[n-1] if k>0 else powers[n-1].conjugate())
        wp=w*phase
        ax += -wp.real
        ay += (-1. if k>0 else 1.)*(1j*wp).real
    return amp*ax-lam*vx,amp*ay-lam*vy

@njit(cache=True)
def micro_initial(x,y,t,modes,coeff,amp,lam):
    u=x+1j*y; maxn=0
    for k in modes:
        if abs(k)>maxn:maxn=abs(k)
    powers=np.empty(maxn+1,np.complex128);powers[0]=1.+0j
    for n in range(1,maxn+1):powers[n]=powers[n-1]*u
    fx=0j;fy=0j
    for j in range(len(modes)):
        k=modes[j];n=abs(k);w=n*coeff[j]*(powers[n-1] if k>0 else powers[n-1].conjugate())
        if k>0: fx-=w;fy-=1j*w
        else: fx-=w;fy+=1j*w
    phase=np.cos(t)-1j*np.sin(t);den=-1.-1j*lam
    dx=amp*fx*phase/den;dy=amp*fy*phase/den
    vx=-1j*dx;vy=-1j*dy
    return dx.real,dy.real,vx.real,vy.real

@njit(cache=True)
def integrate(r0,theta0,vr0,omega0,phase0,modes,coeff,amp,lam,periods,steps,micro=True,sample_factor=1):
    dt=2*np.pi/steps; x=r0*np.cos(theta0);y=r0*np.sin(theta0)
    vx=vr0*np.cos(theta0)-omega0*r0*np.sin(theta0)
    vy=vr0*np.sin(theta0)+omega0*r0*np.cos(theta0)
    if micro:
        dx,dy,dvx,dvy=micro_initial(x,y,phase0,modes,coeff,amp,lam)
        x+=dx;y+=dy;vx+=dvx;vy+=dvy
    stride=steps//sample_factor
    nsave=periods*sample_factor+1; tr=np.empty((nsave,4)); tr[0]=[x,y,vx,vy]
    rmax=np.hypot(x,y); escaped=False;count=1
    for it in range(periods*steps):
        t=phase0+it*dt
        ax1,ay1=acceleration(x,y,vx,vy,t,modes,coeff,amp,lam)
        k1=np.array([vx,vy,ax1,ay1])
        ax2,ay2=acceleration(x+dt*k1[0]/2,y+dt*k1[1]/2,vx+dt*k1[2]/2,vy+dt*k1[3]/2,t+dt/2,modes,coeff,amp,lam)
        k2=np.array([vx+dt*k1[2]/2,vy+dt*k1[3]/2,ax2,ay2])
        ax3,ay3=acceleration(x+dt*k2[0]/2,y+dt*k2[1]/2,vx+dt*k2[2]/2,vy+dt*k2[3]/2,t+dt/2,modes,coeff,amp,lam)
        k3=np.array([vx+dt*k2[2]/2,vy+dt*k2[3]/2,ax3,ay3])
        ax4,ay4=acceleration(x+dt*k3[0],y+dt*k3[1],vx+dt*k3[2],vy+dt*k3[3],t+dt,modes,coeff,amp,lam)
        k4=np.array([vx+dt*k3[2],vy+dt*k3[3],ax4,ay4])
        x+=dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        y+=dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        vx+=dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
        vy+=dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
        r=np.hypot(x,y);rmax=max(rmax,r)
        if not np.isfinite(r+vx+vy) or r>=.995:
            escaped=True;break
        if (it+1)%stride==0:
            tr[count]=[x,y,vx,vy];count+=1
    return tr[:count],rmax,escaped

def classify(tr,periods,escaped):
    ang=np.unwrap(np.arctan2(tr[:,1],tr[:,0]))
    turns=(ang-ang[0])/(2*np.pi)
    rad=np.hypot(tr[:,0],tr[:,1]); windows=[]
    for lo,hi in ((.4,.6),(.6,.8),(.8,1.)):
        a=min(len(tr)-1,int(periods*lo));b=min(len(tr)-1,int(periods*hi))
        windows.append(float(turns[b]-turns[a]))
    late=np.asarray(windows); active=np.min(rad[int(.4*len(rad)):])>.015
    state='escape' if escaped else ('uncertain' if not active else
      ('positive' if np.all(late>.05) else ('negative' if np.all(late<-.05) else
       ('stationary' if np.max(abs(late))<.02 else 'transient'))))
    return dict(turns_total=float(turns[-1]),turns_windows=windows,min_radius=float(rad.min()),state=state)

def load_geometry(name):
    data=json.loads(INPUT.read_text(encoding='utf-8'))
    d=next(x for x in data if x['name']==name)
    arr=np.array(d['coefficients'],float); modes=arr[:,0].astype(np.int64)
    coeff=arr[:,1]+1j*arr[:,2]
    # Discard orders that are below 1e-9 in potential at 0.95R; high orders are
    # still preserved explicitly for the boundary layer near the aperture.
    keep=np.abs(coeff)*(.95**np.abs(modes))**1>1e-9
    return d,modes[keep],coeff[keep]

def run_screen(names=('6_ABC_moderate','6_ABC_thin','9_ABC_moderate','12_ABC_moderate'),
               periods=1600,steps=64):
    RESULTS.mkdir(parents=True,exist_ok=True); rows=[]; rgrid=[.12,.25,.4,.55,.68,.8,.9]
    for name in names:
        geom,modes,coeff=load_geometry(name); lead=abs(coeff[np.where(modes==geom['m'])[0][0]])
        amp=.1044/lead
        for ri,r0 in enumerate(rgrid):
            for spin in (-.04,0.,.04):
                tr,rmax,escaped=integrate(r0,.371,0.,spin,0.,modes,coeff,amp,.00477464829,periods,steps,True)
                row=dict(name=name,r0=r0,spin0=spin,amp=amp,lambda_=.00477464829,
                         periods=periods,steps=steps,max_radius=float(rmax),**classify(tr,periods,escaped))
                rows.append(row)
        np.savez_compressed(RESULTS/f'{name}_screen.npz',rows=np.array([(r['r0'],r['spin0'],r['turns_total'],r['min_radius']) for r in rows if r['name']==name]))
        (RESULTS/'single_screen_checkpoint.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(name,'done',flush=True)
    (RESULTS/'single_screen.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return rows

if __name__=='__main__':run_screen()
