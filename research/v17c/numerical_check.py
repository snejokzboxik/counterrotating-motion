"""Resolution, DOP853, force-gradient, and energy checks of a retained case."""
import json
import time
import numpy as np
from scipy.integrate import solve_ivp
from two_particle_search import OUT, coefficients, integrate, rhs
from validate_candidates import full_metrics


def potential(y,t,cf,C,wz,q):
    u=y[:,0]+1j*y[:,1]
    poly=cf[0]*np.cos(t)+cf[1]*np.sin(t)
    poly8=cf[2]*np.cos(t)+cf[3]*np.sin(t)
    rf=np.sum(q*np.real(u**4*poly+u**8*poly8))
    dc=np.sum(q*wz*wz/2*(y[:,2]**2-abs(u)**2/2))
    coul=C*q[0]*q[1]/np.linalg.norm(y[0,:3]-y[1,:3])
    return rf+dc+coul


def main():
    records=json.loads((OUT/'summary.json').read_text())['records']
    row=records[39]; y=np.load(OUT/'initial_states.npz')['starts'][39]
    cf=coefficients();m=np.asarray(row['masses']);q=np.asarray(row['charges']);g=np.asarray(row['gamma'])
    p=row['phase'];C=row['C'];w=row['wz']; period=2400
    result={}
    force=rhs(y,p,cf,C,w,np.zeros(2),m,q,True)[:,3:]*m[:,None]
    fd=np.empty_like(force); h=1e-6
    for i in range(2):
        for j in range(3):
            yp=y.copy();ym=y.copy();yp[i,j]+=h;ym[i,j]-=h
            fd[i,j]=-(potential(yp,p,cf,C,w,q)-potential(ym,p,cf,C,w,q))/(2*h)
    result['force_gradient_max_abs_error']=float(np.max(abs(force-fd)))
    t0=time.perf_counter();t48,tr48,_,_,_=integrate(y,cf,C,w,g,m,q,p,period,48,True,4)
    result['rk4_48_elapsed_s']=time.perf_counter()-t0
    t0=time.perf_counter();t96,tr96,_,_,_=integrate(y,cf,C,w,g,m,q,p,period,96,True,96)
    result['rk4_96_elapsed_s']=time.perf_counter()-t0
    result['rk4_final_state_max_abs_48_96']=float(np.max(abs(tr48[-1]-tr96[-1])))
    result['rk4_turns_48']=((np.unwrap(np.arctan2(tr48[:,:,1],tr48[:,:,0]),axis=0)[-1]-np.arctan2(y[:,1],y[:,0]))/(2*np.pi)).tolist()
    result['rk4_turns_96']=((np.unwrap(np.arctan2(tr96[:,:,1],tr96[:,:,0]),axis=0)[-1]-np.arctan2(y[:,1],y[:,0]))/(2*np.pi)).tolist()
    met=full_metrics(t48,tr48,cf,row)
    work=np.trapezoid(met['rf_work'].sum(axis=1)+met['power_drag'].sum(axis=1),t48)
    result['energy_change']=float(met['energy'][-1]-met['energy'][0])
    result['external_plus_drag_work']=float(work)
    result['energy_balance_residual']=float(result['energy_change']-work)
    Ldrag=-m[None,:]*g[None,:]*(tr48[:,:,0]*tr48[:,:,4]-tr48[:,:,1]*tr48[:,:,3])
    torque=np.trapezoid(met['rf_torque']+met['coulomb_torque']+Ldrag,t48,axis=0)
    result['angular_momentum_change']=(met['L'][-1]-met['L'][0]).tolist()
    result['integrated_torque']=torque.tolist()
    result['angular_balance_residual']=(met['L'][-1]-met['L'][0]-torque).tolist()
    # DOP853 independently integrates the same physical 3D equations over 400 RF.
    end=400*2*np.pi
    t0=time.perf_counter()
    sol=solve_ivp(lambda tt,Y:rhs(Y.reshape(2,6),tt+p,cf,C,w,g,m,q,True).ravel(),
                  (0,end),y.ravel(),method='DOP853',rtol=1e-10,atol=1e-12,
                  max_step=2*np.pi/24,t_eval=np.linspace(0,end,1601))
    result['dop853_elapsed_s']=time.perf_counter()-t0
    result['dop853_success']=bool(sol.success)
    trd=sol.y.T.reshape(-1,2,6)
    result['dop853_turns_400']=((np.unwrap(np.arctan2(trd[:,:,1],trd[:,:,0]),axis=0)[-1]-np.arctan2(y[:,1],y[:,0]))/(2*np.pi)).tolist()
    t400,tr400,_,_,_=integrate(y,cf,C,w,g,m,q,p,400,96,True,96)
    result['rk4_96_vs_dop853_max_abs_at_400']=float(np.max(abs(tr400[-1]-trd[-1])))
    (OUT/'numerical_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
