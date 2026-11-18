"""Direct RK4 work/drag and angular-momentum balances for full RF motion."""
import json
import numpy as np
from numba import njit
from single_particle import acceleration, micro_initial, load_geometry, RESULTS

@njit(cache=True)
def eval_stage(x,y,vx,vy,t,modes,coeff,amp,lam):
    ax,ay=acceleration(x,y,vx,vy,t,modes,coeff,amp,lam)
    axrf=ax+lam*vx; ayrf=ay+lam*vy
    rhs=np.array([vx,vy,ax,ay])
    power=vx*axrf+vy*ayrf
    diss=lam*(vx*vx+vy*vy)
    tau=x*ayrf-y*axrf
    L=x*vy-y*vx
    taudrag=-lam*L
    return rhs,power,diss,tau,taudrag

@njit(cache=True)
def integrate_balance(r0,theta0,omega0,phase,modes,coeff,amp,lam,periods,steps):
    x=r0*np.cos(theta0); y=r0*np.sin(theta0)
    vx=-omega0*r0*np.sin(theta0); vy=omega0*r0*np.cos(theta0)
    dx,dy,dvx,dvy=micro_initial(x,y,phase,modes,coeff,amp,lam)
    x+=dx; y+=dy; vx+=dvx; vy+=dvy
    tr=np.empty((periods+1,4));tr[0]=[x,y,vx,vy]
    E=np.zeros(periods+1);D=np.zeros(periods+1);T=np.zeros(periods+1);TD=np.zeros(periods+1)
    theta=np.zeros(periods+1); theta_now=np.arctan2(y,x)
    h=2*np.pi/steps; ew=0.;dw=0.;tw=0.;tdw=0.;count=0
    for it in range(periods*steps):
        t=phase+it*h
        k1,p1,d1,t1,td1=eval_stage(x,y,vx,vy,t,modes,coeff,amp,lam)
        k2,p2,d2,t2,td2=eval_stage(x+h*k1[0]/2,y+h*k1[1]/2,vx+h*k1[2]/2,vy+h*k1[3]/2,t+h/2,modes,coeff,amp,lam)
        k3,p3,d3,t3,td3=eval_stage(x+h*k2[0]/2,y+h*k2[1]/2,vx+h*k2[2]/2,vy+h*k2[3]/2,t+h/2,modes,coeff,amp,lam)
        k4,p4,d4,t4,td4=eval_stage(x+h*k3[0],y+h*k3[1],vx+h*k3[2],vy+h*k3[3],t+h,modes,coeff,amp,lam)
        x0=x;y0=y
        x+=h*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
        y+=h*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
        vx+=h*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
        vy+=h*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
        ew+=h*(p1+2*p2+2*p3+p4)/6
        dw+=h*(d1+2*d2+2*d3+d4)/6
        tw+=h*(t1+2*t2+2*t3+t4)/6
        tdw+=h*(td1+2*td2+2*td3+td4)/6
        a0=np.arctan2(y0,x0);a1=np.arctan2(y,x);da=a1-a0
        if da>np.pi: da-=2*np.pi
        elif da<-np.pi: da+=2*np.pi
        theta_now+=da
        if (it+1)%steps==0:
            count+=1;tr[count]=[x,y,vx,vy]
            E[count]=ew;D[count]=dw;T[count]=tw;TD[count]=tdw;theta[count]=theta_now
    return tr,E,D,T,TD,theta

def main():
    d,modes,coeff=load_geometry('6_ABC_moderate');lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
    cases=[('inner_positive',.45216,.1,.425,0.),('outer_negative',.6397,.05,-.098075855,np.pi/2)]
    summary=[]
    for name,r,alpha,omega,theta0 in cases:
        amp=alpha/lead;periods=1400;steps=128
        tr,E,D,T,TD,th=integrate_balance(r,theta0,omega,0.,modes,coeff,amp,.0005,periods,steps)
        K=.5*(tr[:,2]**2+tr[:,3]**2);L=tr[:,0]*tr[:,3]-tr[:,1]*tr[:,2]
        er=K-K[0]-E+D;lr=L-L[0]-T-TD
        row=dict(name=name,r0=r,alpha=alpha,omega0=omega,lambda_=.0005,periods=periods,steps=steps,
                 initial_kinetic=float(K[0]),final_kinetic=float(K[-1]),rf_work=float(E[-1]),drag_loss=float(D[-1]),
                 energy_residual=float(er[-1]),energy_residual_rel=float(abs(er[-1])/(abs(K[-1]-K[0])+abs(E[-1])+abs(D[-1])+1e-30)),
                 initial_angular_momentum=float(L[0]),final_angular_momentum=float(L[-1]),rf_angular_impulse=float(T[-1]),
                 drag_angular_impulse=float(TD[-1]),angular_residual=float(lr[-1]),
                 turns=float((th[-1]-th[0])/(2*np.pi)),min_radius=float(np.min(np.hypot(tr[:,0],tr[:,1]))),
                 max_radius=float(np.max(np.hypot(tr[:,0],tr[:,1]))))
        summary.append(row)
        np.savez_compressed(RESULTS/f'G1_{name}_balance_1400RF.npz',rf_period=np.arange(periods+1),state=tr,
                            theta=th,K=K,RF_work=E,drag_loss=D,RF_angular_impulse=T,drag_angular_impulse=TD,
                            energy_residual=er,angular_residual=lr)
        print(row,flush=True)
    (RESULTS/'full_rf_balance_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

if __name__=='__main__':main()
