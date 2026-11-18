"""Independent high-order solver check with integrated work/torque balances."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from single_particle import RESULTS,load_geometry,micro_initial,acceleration
from full_rf_balances import integrate_balance

def run_case(name,r,theta,omega,alpha,lam,periods,steps=128):
    d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]]);amp=alpha/lead
    x=r*np.cos(theta);y=r*np.sin(theta);vx=-omega*r*np.sin(theta);vy=omega*r*np.cos(theta)
    dx,dy,dvx,dvy=micro_initial(x,y,0.,modes,coeff,amp,lam)
    y0=np.array([x+dx,y+dy,vx+dvx,vy+dvy,0.,0.,0.,0.])
    def rhs(t,z):
        x,y,vx,vy=z[:4];ax,ay=acceleration(x,y,vx,vy,t,modes,coeff,amp,lam)
        axrf=ax+lam*vx;ayrf=ay+lam*vy
        return [vx,vy,ax,ay,vx*axrf+vy*ayrf,lam*(vx*vx+vy*vy),x*ayrf-y*axrf,-lam*(x*vy-y*vx)]
    ts=2*np.pi*np.arange(periods+1)
    sol=solve_ivp(rhs,(0.,2*np.pi*periods),y0,method='DOP853',t_eval=ts,
                  rtol=2e-11,atol=2e-13,max_step=2*np.pi/steps,dense_output=True)
    rk,E,D,T,TD,angle=integrate_balance(r,theta,omega,0.,modes,coeff,amp,lam,periods,steps)
    # solver columns are state+quadrature; RK includes state only and four cumulative integrals.
    delta=np.max(np.abs(sol.y[:4].T-rk))
    # Accumulate angle in short pieces; the stroboscopic RF samples can alias
    # rapid angular sweeps when a trajectory approaches the axis.
    theta_rf=np.zeros(periods+1);minr_rf=np.full(periods,np.inf);theta_now=float(np.arctan2(sol.y[1,0],sol.y[0,0]))
    angle_grid=np.linspace(0.,1.,2049)
    for j in range(periods):
        yy=sol.sol(2*np.pi*(j+angle_grid))
        rr=np.hypot(yy[0],yy[1]);minr_rf[j]=float(rr.min())
        aa=np.arctan2(yy[1],yy[0]);dd=np.diff(aa);dd=np.where(dd>np.pi,dd-2*np.pi,dd);dd=np.where(dd<-np.pi,dd+2*np.pi,dd)
        theta_now+=float(dd.sum());theta_rf[j+1]=theta_now
    turns_d=float((theta_rf[-1]-theta_rf[0])/(2*np.pi))
    K=.5*(sol.y[2]**2+sol.y[3]**2);L=sol.y[0]*sol.y[3]-sol.y[1]*sol.y[2]
    er=K-K[0]-sol.y[4]+sol.y[5]
    lr=L-L[0]-sol.y[6]-sol.y[7]
    row=dict(geometry=name,r0=r,theta0=theta,omega0=omega,alpha=alpha,lambda_=lam,periods=periods,
      DOP853_nfev=int(sol.nfev),DOP853_status=int(sol.status),max_state_difference_vs_RK4=float(delta),
      DOP853_turns=turns_d,RK4_turns=float((angle[-1]-angle[0])/(2*np.pi)),
      DOP853_rf_work=float(sol.y[4,-1]),DOP853_drag_loss=float(sol.y[5,-1]),
      DOP853_energy_residual=float(er[-1]),DOP853_rf_angular_impulse=float(sol.y[6,-1]),
      DOP853_drag_angular_impulse=float(sol.y[7,-1]),DOP853_angular_residual=float(lr[-1]),
      final_radius_DOP853=float(np.hypot(sol.y[0,-1],sol.y[1,-1])),
      final_radius_RK4=float(np.hypot(rk[-1,0],rk[-1,1])))
    np.savez_compressed(RESULTS/f'dop853_{name}_{"pos" if omega>0 else "neg"}_{periods}RF.npz',
        time=ts,state=sol.y[:4].T,work_rf=sol.y[4],drag_loss=sol.y[5],rf_torque=sol.y[6],drag_torque=sol.y[7],
        theta=theta_rf,min_radius_per_period=minr_rf)
    print(row,flush=True)
    return row

def main():
    cases=[('6_ABC_moderate',.6397,np.pi/2,-.098075855,.05,.0005,800),
           ('6_ABC_thin',.34,.371,.12470596599090433,.07308,.0005,1200)]
    rows=[run_case(*c) for c in cases]
    (RESULTS/'dop853_validation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':main()
