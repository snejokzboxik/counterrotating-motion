"""Write exact peak electrode-voltage values for the A9 comparison."""
from __future__ import annotations
import csv,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys_path=ROOT/'src'
import sys;sys.path[:0]=[str(ROOT),str(sys_path)]
from scripts.run_v15_chiral_pilot import POINTS,M0_AMU,AMU,OMEGA,R_M,E_CHARGE,C2,D2,C4

def main():
 alpha,beta,kappa=POINTS['A9'];scale=M0_AMU*AMU*OMEGA**2*R_M**2/E_CHARGE
 u2=beta*scale/C2;dc=alpha*scale/D2;u4=kappa*scale/C4
 rows=[]
 for delta in (0.,math.pi/2):
  for j in range(12):
   th=math.pi*j/6;q=math.cos(2*th);c=math.cos(4*th);s=math.sin(4*th)
   for config,mult in [('Q',0.),('SO_equal_harmonic',1.),('SO_equal_rms',math.sqrt(2)),('RO_PLUS',1.),('RO_MINUS',1.)]:
    v4=u4*mult
    if config=='Q':a=u2*q;b=0.
    elif config.startswith('SO'):a=u2*q+v4*c*math.cos(delta);b=v4*c*math.sin(delta)
    else:
     sig=1 if config=='RO_PLUS' else -1
     a=u2*q+v4*(c*math.cos(delta)-sig*s*math.sin(delta))
     b=v4*(c*math.sin(delta)+sig*s*math.cos(delta))
    peak=abs(dc*q)+math.hypot(a,b)
    rows.append({'delta_deg':math.degrees(delta),'electrode_index':j,'theta_deg':30*j,'configuration':config,
      'U2_V':u2,'VDC_V':dc,'U4_V':v4,'peak_abs_voltage_V':peak,'within_250V':peak<=250})
 out=ROOT/'outputs/octupole_comparison_home/mass_filter_v15/electrode_voltages_A9.csv'
 with out.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('U2',u2,'VDC',dc,'U4',u4,'max',max(r['peak_abs_voltage_V'] for r in rows))
if __name__=='__main__':main()
