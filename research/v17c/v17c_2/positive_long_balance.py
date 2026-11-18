"""Energy and angular-momentum accounting for the longest nominal positive orbit."""
import json
import numpy as np
from full_rf_balances import integrate_balance
from single_particle import RESULTS,load_geometry

def main():
    name='6_ABC_thin';r=.34;theta=.371;omega=.12470596599090433;alpha=.07308;lam=.0005;periods=8000;steps=96
    d,modes,coeff=load_geometry(name);lead=abs(coeff[np.where(modes==int(d['m']))[0][0]])
    tr,E,D,T,TD,ang=integrate_balance(r,theta,omega,0.,modes,coeff,alpha/lead,lam,periods,steps)
    rad=np.hypot(tr[:,0],tr[:,1]);K=.5*(tr[:,2]**2+tr[:,3]**2);L=tr[:,0]*tr[:,3]-tr[:,1]*tr[:,2]
    er=K-K[0]-E+D;lr=L-L[0]-T-TD
    row=dict(geometry=name,r0=r,theta0=theta,omega0=omega,alpha=alpha,lambda_=lam,periods=periods,steps=steps,
      turns=float((ang[-1]-ang[0])/(2*np.pi)),min_radius=float(rad.min()),max_radius=float(rad.max()),
      initial_K=float(K[0]),final_K=float(K[-1]),rf_work=float(E[-1]),drag_loss=float(D[-1]),
      energy_residual=float(er[-1]),initial_L=float(L[0]),final_L=float(L[-1]),
      rf_angular_impulse=float(T[-1]),drag_angular_impulse=float(TD[-1]),angular_residual=float(lr[-1]),
      windows=[dict(periods=[a,b],rf_work=float(E[b]-E[a]),drag_loss=float(D[b]-D[a]),
                    rf_torque=float(T[b]-T[a]),drag_torque=float(TD[b]-TD[a]),
                    turns=float((ang[b]-ang[a])/(2*np.pi)),r_mean=float(rad[a:b].mean()),r_min=float(rad[a:b].min()),r_max=float(rad[a:b].max()))
               for a,b in [(0,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000)]])
    np.savez_compressed(RESULTS/'G2_positive_long_balance_8000RF.npz',rf_period=np.arange(periods+1),state=tr,
      theta=ang,K=K,RF_work=E,drag_loss=D,RF_angular_impulse=T,drag_angular_impulse=TD,
      energy_residual=er,angular_residual=lr)
    (RESULTS/'G2_positive_long_balance_8000RF.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    print(json.dumps(row,indent=2),flush=True)

if __name__=='__main__':main()
