"""One self-consistent illustrative Stokes/air scale for candidate 48."""
import json, math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
pz=np.load(HERE/"stage2_speed"/"configurations.npz")
p=np.asarray(pz["p"][48],float); qratio=float(pz["qratio"][48])
cfg=json.loads((HERE.parent.parent/"v17c_5"/"input"/"config.json").read_text(encoding="utf-8-sig"))
f1_reference=float(cfg["f1_Hz"])
A,B,nu,beta,s,C,lam1,lam2=map(float,p)

# Pick a millimetre tungsten sphere for particle 1. The other radius and
# density are solved from the fitted mass and Stokes-damping ratios.
a1=1e-3; rho1=19300.; eta=1.81e-5; epsilon0=8.8541878128e-12
R=a1/2e-4; m1=4*math.pi*rho1*a1**3/3; mass_ratio=1/s
a2=a1*(lam2/lam1)*mass_ratio; m2=m1*mass_ratio
rho2=m2/(4*math.pi*a2**3/3)
omega1=9*eta/(2*rho1*a1*a1*lam1); f1=omega1/(2*math.pi)
q1=math.sqrt(C*4*math.pi*epsilon0*m1*R**3*omega1**2/qratio); q2=qratio*q1
U1=A*m1*R**2*omega1**2/q1
electrode_U2_U1=3.22721397
U2=electrode_U2_U1*U1; Vdc=beta/A*U1

# Set gas density so chi=Re/|v*| is 0.2 for species 1. The size ratio
# then gives chi2. This yields one common gas, not two drag laws.
chi1=.2; rho_gas=chi1*eta/(2*a1*R*omega1)
temperature=293.15; R_air=287.05; pressure=rho_gas*R_air*temperature
mfp=68e-9*101325/pressure
chi2=chi1*a2/a1

# 2-D MFS geometry scale; local edge enhancement is not represented here.
D=R/(1-.05); a_e=.05*D; Bscreen=1.10*D
center_chord=2*D*math.sin(math.pi/24); interelectrode_gap=center_chord-2*a_e
# theta step is 2*pi/24. The two drive harmonics advance by 2 and 4
# times that angle between adjacent rods; the difference amplitude is
# 2 U |sin(phase_step/2)|.
delta_u1=2*U1*math.sin(math.pi/12)
delta_u2=2*U2*math.sin(math.pi/6)

speeds={}
for ensemble,index in (("rest",0),("weak",1),("wide",2)):
    data=np.load(HERE/"stage3"/"runs"/f"dop_cfg48_{ensemble}_{index}.npz")
    state=data["state"]; v=state[:,[3,4,5,9,10,11]].reshape(-1,2,3)
    vm=np.linalg.norm(v,axis=2)
    speeds[ensemble]={"mean_dimensionless_speed":vm.mean(axis=0).tolist(),
      "max_dimensionless_speed":vm.max(axis=0).tolist(),
      "max_Re_Oseen":(np.array([chi1,chi2])*vm.max(axis=0)).tolist()}

freq_dop=json.loads((HERE/"stage3"/"runs"/"dop_cfg48_rest_results.json").read_text())[0]["frequency_Hz"]
out={"model":"illustrative candidate-48 dimensional mapping; not an engineering design",
 "reference_dimensionless_parameters":{"A":A,"B":B,"nu":nu,"beta":beta,"mass_ratio_m2_m1":mass_ratio,
   "charge_ratio_q2_q1":qratio,"C":C,"lambda1":lam1,"lambda2":lam2,
   "electrode_voltage_ratio_U2_U1":electrode_U2_U1},
 "particles":{"a1_m":a1,"a2_m":a2,"rho1_kg_m3":rho1,"rho2_kg_m3":rho2,
   "m1_kg":m1,"m2_kg":m2,"q1_C":q1,"q2_C":q2,"q1_e":q1/1.602176634e-19,
   "q2_e":q2/1.602176634e-19,"isolated_surface_field1_V_m":q1/(4*math.pi*epsilon0*a1*a1),
   "isolated_surface_field2_V_m":q2/(4*math.pi*epsilon0*a2*a2)},
 "trap":{"R_m":R,"D_m":D,"electrode_radius_m":a_e,"screen_inner_radius_m":Bscreen,
   "adjacent_rod_surface_gap_m":interelectrode_gap,"f1_Hz":f1,"f2_Hz":nu*f1,
   "U1_peak_V":U1,"U2_peak_V":U2,"ideal_quadrupole_Vdc_V":Vdc,
   "upper_bound_adjacent_RF_difference_V":delta_u1+delta_u2,
   "gap_average_RF_field_bound_V_m":(delta_u1+delta_u2)/interelectrode_gap},
 "fluid":{"description":"air-like ideal gas at 293.15 K; viscosity fixed to 1.81e-5 Pa s",
   "density_kg_m3":rho_gas,"ideal_pressure_Pa":pressure,"mean_free_path_estimate_m":mfp,
   "Kn_a1":mfp/a1,"Kn_a2":mfp/a2,"chi1_Re_per_dimensionless_speed":chi1,
   "chi2_Re_per_dimensionless_speed":chi2,"dimensionless_speed_samples":speeds},
 "DOP853_rotation":{"reference_frequency_Hz_at_f1_100kHz":freq_dop,
   "physical_frequency_Hz_at_mapped_f1":[f/f1_reference*f1 for f in freq_dop]},
 "caveats":["Stokes damping solves this scale; Oseen is then a small correction, not an independently chosen lambda.",
   "a1/R=0.0002 and a2/R≈0.00189; both particle and trap scales are large for a conventional RF trap.",
   "Knudsen values are small for this illustrative pressure, but slip corrections and charge retention need measurement.",
   "Surface fields are isolated-sphere estimates; no finite 3-D electrode edge field or breakdown threshold is asserted.",
   "The ideal DC quadrupole voltage is not a derived endcap design."]}
(HERE/"physical_scale_mapping.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
print(json.dumps(out,indent=2))
