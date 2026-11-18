"""Independent FMM, complex-harmonic, and field-gradient audit."""
import json
import numpy as np
from pathlib import Path
from scipy.linalg import solve

HERE=Path(__file__).resolve().parent
NAMES=['6_ABC_moderate','6_ABC_thin','9_ABC_moderate','12_ABC_moderate']

def reconstruct(N,m,a,gap,p):
    D=1.; B=1+a+gap; R=1-a
    theta=np.arange(N)*2*np.pi/N
    centers=np.exp(1j*theta)
    phi=theta[:,None]+np.arange(p)*2*np.pi/p
    boundary=(centers[:,None]+a*np.exp(1j*phi)).ravel()
    src=(centers[:,None]+.65*a*np.exp(1j*phi)).ravel()
    def greens(z):
        return np.log(np.abs(B*B-z[:,None]*src.conj()[None,:])/(B*np.abs(z[:,None]-src[None,:])))
    q=solve(greens(boundary),np.repeat(np.exp(1j*m*theta),p),assume_a='gen')
    # Evaluate boundary values at interleaved angles, independently of collocation.
    test=(centers[:,None]+a*np.exp(1j*(phi+np.pi/p))).ravel()
    berr=float(np.max(np.abs(greens(test)@q-np.repeat(np.exp(1j*m*theta),p))))
    def coef(k):
        n=abs(k); f=(src**(-n)-(src.conj()/B**2)**n)*R**n/(2*n)
        return np.dot(q,f if k>0 else f.conj())
    modes=np.array([m+N*k for k in range(-25,26) if abs(m+N*k)<=160])
    cs=np.array([coef(int(k)) for k in modes])
    return B,R,modes,cs,berr,src,q

def pot_harm(r,theta,modes,cs):
    zz=(r*np.exp(1j*theta))
    powers=np.array([zz**k if k>0 else zz.conjugate()**(-k) for k in modes])
    return float(np.real(np.sum(cs*powers)))

def field_harm(r,theta,modes,cs):
    # Cartesian field from the exact derivative of each signed solid harmonic.
    x,y=r*np.cos(theta),r*np.sin(theta); u=x+1j*y; force=np.zeros(2)
    for k,c in zip(modes,cs):
        n=abs(int(k))
        if k>0:
            w=n*c*u**(n-1); force += [-w.real,w.imag]
        else:
            w=n*c*u.conjugate()**(n-1); force += [-w.real,-w.imag]
    return force

def main():
    input_dir=HERE
    saved=json.loads((input_dir/'harmonics.json').read_text(encoding='utf-8'))
    rows=[]
    for record in saved:
        if record['name'] not in NAMES: continue
        N,m=record['N'],record['m'];a=record['a_D'];gap=record['gap_D']
        for p in (24,40,64):
            B,R,modes,cs,berr,src,q=reconstruct(N,m,a,gap,p)
            # Compare every stored signed coefficient.
            stored={int(k):complex(re,im) for k,re,im in record['coefficients']}
            cerr=max(abs(c-stored[int(k)]) for k,c in zip(modes,cs))
            r=.61; th=.37; h=1e-6
            # Independent field from FMM Green-function derivative.
            z=r*R*np.exp(1j*th)
            dG=1/(z-B**2/src.conj())-1/(z-src)
            fx=-np.real(dG.real@q)*R; fy=np.real(dG.imag@q)*R
            # Solid harmonics differentiate in physical coordinates via normalized r.
            Fh=field_harm(r,th,modes,cs)
            # Direct centered difference of FMM potential; rescale physical x by R.
            xy=np.array([z.real,z.imag]); fd=[]
            def gpoint(point):
                return np.log(np.abs(B*B-point*src.conj())/(B*np.abs(point-src)))
            for j in range(2):
                dp=np.zeros(2);dp[j]=h*R
                vp=float(np.real(gpoint(complex(xy[0]+dp[0],xy[1]+dp[1]))@q))
                vm=float(np.real(gpoint(complex(xy[0]-dp[0],xy[1]-dp[1]))@q))
                fd.append(-(vp-vm)/(2*h))
            # Expansion error sampled at .61R, compare potential and force.
            fmm=np.array([fx,fy]); rows.append(dict(name=record['name'],p=p,
                boundary_error=berr,max_coefficient_delta=float(cerr),
                field_vs_fd_max_abs=float(np.max(abs(fmm-fd))),
                harmonic_field_vs_fmm_max_abs=float(np.max(abs(Fh-fmm))),
                potential_abs=float(pot_harm(r,th,modes,cs))))
            # Recompute sampled peak surface field from both full complex quadratures.
            theta=np.arange(N)*2*np.pi/N; centers=np.exp(1j*theta)
            angles=theta[:,None]+np.arange(512)*2*np.pi/512
            surface=(centers[:,None]+a*np.exp(1j*angles)).ravel()
            peaks=[]
            for slab in np.array_split(surface,32):
                deriv=1/(slab[:,None]-B**2/src.conj()[None,:])-1/(slab[:,None]-src[None,:])
                ex=-(deriv.real@q); ey=deriv.imag@q
                peak=np.sqrt((abs(ex)**2+abs(ey)**2+abs(ex**2+ey**2))/2)
                peaks.append(float(peak.max()))
            rows[-1]['surface_Emax_corrected_U0_over_R']=max(peaks)*R
    (input_dir/'independent_audit.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
