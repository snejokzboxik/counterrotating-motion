"""Long-horizon and independent-method controls for the two-particle screen."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from two_particle_search import OUT, coefficients, diagnostics, integrate, rhs


def full_metrics(t, tr, cf, row):
    m=np.asarray(row['masses']); q=np.asarray(row['charges']); g=np.asarray(row['gamma'])
    C=row['C']; wz=row['wz']; phase=row['phase']
    u=tr[:,:,0]+1j*tr[:,:,1]; r2=abs(u)**2
    pot_rf=np.zeros(len(t)); rf_work=np.zeros((len(t),2)); dc=np.zeros(len(t))
    power_drag=np.zeros((len(t),2)); torque=np.zeros((len(t),2)); rf_force=np.zeros((len(t),2,3))
    for j,tt in enumerate(t):
        ct=np.cos(tt+phase); st=np.sin(tt+phase)
        poly=cf[0]*ct+cf[1]*st; poly8=cf[2]*ct+cf[3]*st
        D=4*u[j]**3*poly+8*u[j]**7*poly8
        rf_force[j,:,0]=-q*D.real; rf_force[j,:,1]=q*D.imag
        pot_rf[j]=np.sum(q*np.real(u[j]**4*poly+u[j]**8*poly8))
        dphi_dt=np.real(u[j]**4*(-cf[0]*st+cf[1]*ct)+u[j]**8*(-cf[2]*st+cf[3]*ct))
        rf_work[j]=q*dphi_dt
        dc[j]=np.sum(q*wz*wz/2*(tr[j,:,2]**2-r2[j]/2))
        power_drag[j]=-m*g*np.sum(tr[j,:,3:]**2,axis=1)
        torque[j]=tr[j,:,0]*rf_force[j,:,1]-tr[j,:,1]*rf_force[j,:,0]
    sep=np.linalg.norm(tr[:,0,:3]-tr[:,1,:3],axis=1)
    coul=C*q[0]*q[1]/sep
    kin=np.sum(.5*m[None,:]*np.sum(tr[:,:,3:]**2,axis=2),axis=1)
    energy=kin+pot_rf+dc+coul
    dr=tr[:,0,:3]-tr[:,1,:3]
    fc=(C*q[0]*q[1]/sep**3)[:,None]*dr
    coul_torque=np.column_stack((tr[:,0,0]*fc[:,1]-tr[:,0,1]*fc[:,0],
                                  -tr[:,1,0]*fc[:,1]+tr[:,1,1]*fc[:,0]))
    L=m[None,:]*(tr[:,:,0]*tr[:,:,4]-tr[:,:,1]*tr[:,:,3])
    return dict(energy=energy,coulomb=coul,rf_work=rf_work,power_drag=power_drag,
                rf_torque=torque,coulomb_torque=coul_torque,L=L,sep=sep,
                force_ratio=np.linalg.norm(fc,axis=1)/(np.linalg.norm(rf_force[:,0],axis=1)+1e-12))


def make_plot(index,t,tr):
    fig=plt.figure(figsize=(11,8))
    axes=[fig.add_subplot(2,3,1,projection='3d')]+[fig.add_subplot(2,3,k) for k in range(2,6)]
    for i,col in enumerate(('tab:blue','tab:orange')):
        x,y,z=tr[:,i,0],tr[:,i,1],tr[:,i,2]
        axes[0].plot(x,y,z,color=col,lw=.6)
        for ax,a,b in ((axes[1],x,y),(axes[2],x,z),(axes[3],y,z)): ax.plot(a,b,color=col,lw=.6)
        turns=(np.unwrap(np.arctan2(y,x))-np.arctan2(y[0],x[0]))/(2*np.pi)
        axes[4].plot(t/(2*np.pi),turns,color=col,label=f'particle {i+1}')
    for ax,xlab,ylab in ((axes[1],'x/R','y/R'),(axes[2],'x/R','z/R'),(axes[3],'y/R','z/R'),(axes[4],'RF periods','turns')):
        ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.grid(alpha=.3)
    axes[0].set_xlabel('x/R');axes[0].set_ylabel('y/R');axes[0].set_zlabel('z/R')
    axes[4].legend();fig.tight_layout();fig.savefig(OUT/f'case_{index}_trajectories.png',dpi=140);plt.close(fig)


def main():
    data=json.loads((OUT/'summary.json').read_text()); starts=np.load(OUT/'initial_states.npz')['starts']
    cf=coefficients(); results=[]
    for i in (16,39,53):
        row=data['records'][i]; y=starts[i]
        m=np.asarray(row['masses']);q=np.asarray(row['charges']);g=np.asarray(row['gamma'])
        for kind in ('full','pure_ro4','no_coulomb'):
            cc=cf.copy(); C=row['C'];
            if kind=='pure_ro4': cc[2:]=0
            if kind=='no_coulomb': C=0
            t0=time.perf_counter()
            t,tr,sep,maxr,escaped=integrate(y,cc,C,row['wz'],g,m,q,row['phase'],2400,48,C!=0,48)
            d=diagnostics(t,tr,2400,escaped)
            rr=dict(index=i,kind=kind,min_separation=sep,max_radius=maxr,elapsed_s=time.perf_counter()-t0,**d)
            if kind=='full' and not escaped:
                metric=full_metrics(t,tr,cc,row)
                rr.update(coulomb_energy_range=[float(metric['coulomb'].min()),float(metric['coulomb'].max())],
                          mean_separation=float(metric['sep'].mean()),
                          force_ratio_median=float(np.median(metric['force_ratio'])),
                          energy_change=float(metric['energy'][-1]-metric['energy'][0]),
                          angular_momentum_change=(metric['L'][-1]-metric['L'][0]).tolist())
                np.savez_compressed(OUT/f'case_{i}_full.npz',t=t,tr=tr,angle=np.unwrap(np.arctan2(tr[:,:,1],tr[:,:,0]),axis=0),
                                    energy=metric['energy'],L=metric['L'],separation=metric['sep'],
                                    coulomb_energy=metric['coulomb'],rf_work=metric['rf_work'],drag_power=metric['power_drag'])
                make_plot(i,t,tr)
            results.append(rr); print(i,kind,escaped,d['windows_turns'],flush=True)
    (OUT/'long_controls.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__': main()
