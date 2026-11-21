"""Resolve RF micromotion and the exact torque budget from terminal Local15 states."""
import csv,json,math
from pathlib import Path
import numpy as np
from local15_exact_validation import (ROOT,PANEL,OUT,PARAM,load_panel,
                                      acceleration,rk4_step)

ROOT=Path(__file__).resolve().parents[1]
RUN_DIR=ROOT/"results/local15_exact_rf"
outdir=RUN_DIR/"micromotion"
outdir.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader(PANEL.open(newline="",encoding="utf-8")))
row=rows[0]  # preregistered S01, at rest initially
phase1=np.array([float(row["phase1"])])
phase2=np.array([float(row["phase2"])])
state0=np.array([[float(row["x0"]),float(row["y0"]),
                  float(row["vx0"]),float(row["vy0"])]])
periods=1000
steps_per_period=96
dt=2*np.pi/steps_per_period
nsteps=periods*steps_per_period
tau0=2*np.pi*15000
summary={"start_id":row["start_id"],"source_panel_sha256":
 "9aeefa9546dc320e5158a18ffdecd3a7d5fe55a8deaa4fe7e876015d188f8761",
 "initial_state_reused_from_terminal_checkpoint":True,
 "sampling":{"extra_periods":periods,"steps_per_primary_period":steps_per_period,
             "sample_interval_tau":dt,"secondary_cycles_in_window":PARAM["nu"]*periods},
 "species":{}}
for name in ("species1","species2"):
 cp=np.load(RUN_DIR/f"{name}_checkpoint.npz")
 idx=0
 y=cp["state"][idx:idx+1].copy()
 if not bool(cp["active"][idx]):
  raise RuntimeError(f"{name} S01 is not retained")
 alpha=PARAM["species"][name]["alpha"]; lam=PARAM["species"][name]["lambda"]
 active=np.ones(1,dtype=bool)
 rho=[]; omega=[]; tau1=[]; tau2=[]; drag=[]; times=[]
 angle=[]
 for step in range(nsteps+1):
  t=tau0+step*dt
  x,yc,vx,vy=y[0]
  r=math.hypot(x,yc)
  rho.append(r); times.append(t)
  omega.append((x*vy-yc*vx)/(r*r))
  th=math.atan2(yc,x)
  angle.append(th)
  ph1=2*th-t+float(phase1[0])
  ph2=-4*th-PARAM["nu"]*t+float(phase2[0])
  tau1.append(alpha*2*PARAM["A"]*r**2*math.sin(ph1))
  tau2.append(alpha*(-4)*PARAM["B"]*r**4*math.sin(ph2))
  drag.append(-lam*r*r*omega[-1])
  if step<nsteps:
   y=rk4_step(t,y,dt,phase1,phase2,alpha,lam,active)
 angle=np.unwrap(np.asarray(angle))
 rho=np.asarray(rho); omega=np.asarray(omega)
 tau1=np.asarray(tau1); tau2=np.asarray(tau2); drag=np.asarray(drag)
 times=np.asarray(times)
 # Trapezoidal phase-time averages over the explicitly resolved window.
 mean=lambda v:float(np.trapezoid(v,times)/(times[-1]-times[0]))
 doc={
  "sample_count":len(times),"mean_rho":mean(rho),"std_rho_rf_resolved":float(np.std(rho)),
  "min_rho":float(rho.min()),"max_rho":float(rho.max()),
  "mean_omega_instantaneous":mean(omega),"omega_from_unwrapped_angle":
      float((angle[-1]-angle[0])/(times[-1]-times[0])),
  "std_omega_instantaneous":float(np.std(omega)),
  "mean_rf_torque_mode1":mean(tau1),"mean_rf_torque_mode2":mean(tau2),
  "mean_rf_torque_total":mean(tau1+tau2),"mean_drag_torque":mean(drag),
  "mean_torque_residual":mean(tau1+tau2+drag),
  "rho_reference_phase1_sampled_period_std_from_full_run":float(
      cp["sample_rho"][-500:,idx].std())
 }
 summary["species"][name]=doc
 np.savez_compressed(outdir/f"{name}_S01_1000periods.npz",tau=times,rho=rho,
   omega=omega,theta_unwrapped=angle,torque_mode1=tau1,
   torque_mode2=tau2,drag_torque=drag)
( outdir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
print(json.dumps(summary,indent=2))
