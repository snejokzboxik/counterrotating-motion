"""Independent adaptive-DOP853 checks of selected V14 fixed-step pilot outcomes."""
from __future__ import annotations
import csv,json,math,sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_v14_ideal_pilot import OUT,ALPHA,BETA
from three_phase_trap.v14_dynamics import rhs

def main():
    rows=[json.loads(s) for s in (OUT/'ideal_pilot_checkpoint.jsonl').read_text().splitlines() if s.strip()]
    selected=[]
    for name in ('Q','RO','SO_equal_harmonic','SO_equal_rms'):
        part=[r for r in rows if r['model']==name]
        for status in ('retained','escaped'):
            candidates=[r for r in part if r['status']==status]
            selected.extend(candidates[::max(1,len(candidates)//6)][:6])
    checked=[]
    for r in selected:
        model='SO' if r['model'].startswith('SO') else r['model']
        z0=np.asarray(r['initial']);tau0=r['tau0_rad'];k=r['kappa']
        def fun(t,z):return rhs(t,z,ALPHA,BETA,k,model,tau0=tau0)
        def ev(t,z):return math.hypot(z[0],z[1])-1
        ev.direction=1;ev.terminal=True
        sol=solve_ivp(fun,(0,200*math.pi),z0,method='DOP853',rtol=2e-11,atol=2e-13,
                      max_step=2*math.pi/128,events=ev)
        status='unresolved' if not sol.success else ('escaped' if sol.status==1 else 'retained')
        exitp=sol.t_events[0][0]/(2*math.pi) if sol.status==1 else math.nan
        checked.append(dict(key=r['key'],model=r['model'],RK4_status=r['status'],DOP853_status=status,
                            agree=status==r['status'],RK4_exit_periods=r['exit_periods'],
                            DOP853_exit_periods=exitp,
                            exit_difference=None if r['exit_periods'] is None else exitp-r['exit_periods']))
    with (OUT/'ideal_pilot_dop853_validation.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=checked[0]);w.writeheader();w.writerows(checked)
    print('checked',len(checked),'agree',sum(r['agree'] for r in checked),
          'max_exit_difference_RF',max(abs(r['exit_difference']) for r in checked if r['exit_difference'] is not None))

if __name__=='__main__':main()
