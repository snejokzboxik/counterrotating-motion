"""Matched-initial-condition RF--DC hydrodynamic model comparison.

Models: linear Stokes self drag, isolated-sphere Oseen correction, pair Oseen
mobility converted to resistance, and unequal-sphere generalized RPY mobility.
The RF, DC, Coulomb terms and initial Cartesian conditions are identical.
"""
from __future__ import annotations

import csv, json, math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
BASE = V4 / "results" / "full_dynamics" / "ADCFEB_sobol_800p_n512_s20261118" / "all_runs.csv"
SPEC = V4 / "results" / "fem_sixphase" / "G1_G2_12electrode_6phase_P1_FEM.json"
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
N, STRIDE = 2, 13
EPS0 = 8.8541878128e-12
E_CHARGE = 1.602176634e-19

# One consistent illustrative continuum-gas mapping for all matched records.
# Gas parameters are stated as model assumptions; this is not a claim that a
# real six-phase electrode assembly has these exact fluid boundary conditions.
PHYS = {"trap_radius_m": 2.0e-4, "omega_rf_rad_s": 2.0e4,
        "U0_reference_V": None, "fluid_name": "air, 293 K, approximately 1 atm",
        "fluid_density_kg_m3": 1.204, "dynamic_viscosity_Pa_s": 1.81e-5,
        "mean_free_path_m": 68.0e-9, "particle1_material_density_kg_m3": 2000.0,
        "hydrodynamic_wall_corrections": "omitted; unbounded-fluid two-sphere tensors"}
CASE_IDS = ["ADCFEB_000151", "ADCFEB_000444", "ADCFEB_000473",
            "ADCFEB_000453", "ADCFEB_000509", "ADCFEB_000083"]
MODELS = ("stokes", "oseen_self", "oseen_pair", "rpy_pair")


def load_spectrum():
    doc = json.loads(SPEC.read_text(encoding="utf-8"))
    rows = doc["geometries"]["ADCFEBADCFEB"]["mesh_levels"][-1]["mfs_coefficients"]
    vals = {int(k): complex(re, im) for k, re, im in rows}
    vals = {k: (v if k % 2 == 0 else 0j) for k, v in vals.items()}
    modes = np.asarray(sorted(vals), dtype=int)
    coeff = np.asarray([vals[int(k)] for k in modes], dtype=np.complex128)
    keep = np.abs(coeff) > 1e-15
    return modes[keep], coeff[keep]


def rf_accel(x, y, tau, alpha, modes, coeff):
    z = complex(x, y)
    carrier = np.exp(-1j * tau)
    out = np.zeros(3)
    for k, c in zip(modes, coeff):
        k = int(k); n = abs(k); base = z if k > 0 else z.conjugate()
        d = n * c * base ** (n - 1) * carrier
        out[0] -= alpha * d.real
        out[1] += (-1.0 if k > 0 else 1.0) * alpha * (1j*d).real
    return out


def rpy_cross_perp_parallel(rvec, a1, a2, mu):
    """Generalized RPY translational mobility for unequal spheres.

    Piecewise expression follows Zuk et al., J. Fluid Mech. 741 (2014),
    equations (2.9), (3.1), DOI 10.1017/jfm.2013.668.
    """
    r = float(np.linalg.norm(rvec))
    if r <= 1e-18:
        # Limit for complete nesting: mobility equals that of the outer sphere.
        h = 1.0 / (6.0 * math.pi * mu * max(a1, a2))
        return h, h
    if r > a1 + a2:
        s = a1*a1 + a2*a2
        pref = 1.0 / (8.0 * math.pi * mu * r)
        hp = pref * (1.0 + s/(3.0*r*r))
        hl = pref * (2.0 - 2.0*s/(3.0*r*r))
        return hp, hl
    am, an = max(a1, a2), min(a1, a2)
    if r <= am - an:
        h = 1.0 / (6.0 * math.pi * mu * am)
        return h, h
    d2 = (a1-a2)**2
    c_iso = (16.0*r**3*(a1+a2) - (d2+3.0*r*r)**2) / (32.0*r**3)
    c_rr = 3.0*(d2-r*r)**2 / (32.0*r**3)
    pref = 1.0 / (6.0 * math.pi * mu * a1*a2)
    hp = pref * c_iso
    hl = pref * (c_iso + c_rr)
    return hp, hl


def pair_resistance_drag(v1, v2, rvec, mass, radius, omega, mu, model):
    """Compute -R_hyd v using Oseen or RPY mobility then invert analytically.

    The Oseen/RPY two-sphere mobility is diagonal in directions parallel and
    perpendicular to the line of centers. Inverting each 2x2 block avoids a
    generic 6x6 matrix inverse at every ODE evaluation.
    """
    r = float(np.linalg.norm(rvec))
    if r <= 1e-18:
        raise FloatingPointError("coincident particle centers")
    n = rvec/r
    if model == "oseen_pair":
        hperp = 1.0/(8.0*math.pi*mu*r)
        hpar = 2.0*hperp
    else:
        hperp,hpar = rpy_cross_perp_parallel(rvec,radius[0],radius[1],mu)
    zeta = 6.0*math.pi*mu*np.asarray(radius)
    # Mobility block eigenvalues in parallel/perpendicular directions.
    detp = 1.0/(zeta[0]*zeta[1]) - hperp*hperp
    detl = 1.0/(zeta[0]*zeta[1]) - hpar*hpar
    if detp <= 0.0 or detl <= 0.0:
        raise FloatingPointError("non-positive two-sphere mobility determinant")
    P = np.outer(n,n)
    Q = np.eye(3)-P
    def blocks(h, det):
        r11 = (1.0/zeta[1]/det)*Q + (1.0/zeta[1]/detl)*P
        r22 = (1.0/zeta[0]/det)*Q + (1.0/zeta[0]/detl)*P
        r12 = (-hperp/detp)*Q + (-hpar/detl)*P
        return r11,r12,r22
    r11,r12,r22=blocks(hperp,detp)
    # Dimensionless v' to acceleration a': divide physical resistance by m_i*Omega.
    drag1=-(r11@v1+r12@v2)/(mass[0]*omega)
    drag2=-(r12@v1+r22@v2)/(mass[1]*omega)
    return drag1,drag2, (detp,detl)


def initial_state(row, modes, coeff):
    y=np.zeros((N,STRIDE))
    al=np.asarray([row.alpha1,row.alpha2],float)
    la=np.asarray([row.lambda1,row.lambda2],float)
    for i,(r,th,om,vr,z,vz) in enumerate([
        (row.r1_0,row.theta1_0,row.omega1_0,row.vr1_0,row.z1_0,row.vz1_0),
        (row.r2_0,row.theta2_0,row.omega2_0,row.vr2_0,row.z2_0,row.vz2_0)]):
        c,s=math.cos(th),math.sin(th)
        y[i,:6]=[r*c,r*s,z,vr*c-om*r*s,vr*s+om*r*c,vz]
        # Same Stokes-based analytic RF micromotion preparation as checkpoint.
        u=complex(y[i,0],y[i,1]); phase=np.exp(-1j*float(row.phase0))
        fx=0j; fy=0j
        for k,a in zip(modes,coeff):
            k=int(k); n=abs(k); base=u if k>0 else u.conjugate()
            d=-n*a*base**(n-1)*phase
            fx += al[i]*d
            fy += al[i]*((-1j*d) if k>0 else (1j*d))
        dx=fx/(-1.0-1j*la[i]); dy=fy/(-1.0-1j*la[i])
        y[i,0]+=dx.real; y[i,1]+=dy.real
        y[i,3]+=(-1j*dx).real; y[i,4]+=(-1j*dy).real
    return y.reshape(-1)


def solve_case(row, model, modes, coeff, mapping, periods=None):
    alpha=np.asarray([row.alpha1,row.alpha2],float)
    beta=np.asarray([row.beta1,row.beta2],float)
    # This physical map is constructed to reproduce the selected Stokes lambdas.
    lam=np.asarray(mapping["lambda_physical"],float)
    mass=np.asarray([1.0,float(row.mass2_over_mass1)])
    charge=np.asarray([1.0,float(row.charge2_over_charge1)])
    chi=float(row.chi)
    phase0=float(row.phase0)
    rel=periods or int(float(row.periods))
    spp=8
    t_eval=np.linspace(0,rel*2*math.pi,rel*spp+1)
    y0=initial_state(row,modes,coeff)
    hydro_checks={"min_det_perp":float("inf"),"min_det_parallel":float("inf"),
                  "max_drag_positive_power":0.0}
    def f(t,y):
        state=y.reshape(N,STRIDE); out=np.zeros_like(state)
        ar=np.zeros((N,3)); adc=np.zeros((N,3)); ac=np.zeros((N,3))
        ad=np.zeros((N,3)); vel=state[:,3:6]
        for i in range(N):
            ar[i]=rf_accel(state[i,0],state[i,1],t+phase0,alpha[i],modes,coeff)
            adc[i]=beta[i]*np.array([state[i,0],state[i,1],-2*state[i,2]])
        dr=state[0,:3]-state[1,:3]; d2=float(np.dot(dr,dr))
        if d2<1e-14: raise FloatingPointError("Coulomb point singularity floor")
        ac[0]=chi*charge[0]*charge[1]*dr/(mass[0]*d2**1.5)
        ac[1]=-chi*charge[0]*charge[1]*dr/(mass[1]*d2**1.5)
        if model in ("stokes","oseen_self"):
            if model=="stokes": fac=np.ones(N)
            else:
                Re=np.asarray(mapping["re_prefactor_per_dimensionless_speed"])*np.linalg.norm(vel,axis=1)
                fac=1.0+3.0*Re/16.0
            ad=-lam[:,None]*vel*fac[:,None]
        else:
            ad[0],ad[1],dets=pair_resistance_drag(vel[0],vel[1],dr,
                mapping["mass_kg"],mapping["radii_m"],mapping["omega_rf_rad_s"],
                mapping["fluid_mu_Pa_s"],model)
            hydro_checks["min_det_perp"]=min(hydro_checks["min_det_perp"],dets[0])
            hydro_checks["min_det_parallel"]=min(hydro_checks["min_det_parallel"],dets[1])
        drag_power=float(np.sum(mass*np.sum(vel*ad,axis=1)))
        hydro_checks["max_drag_positive_power"]=max(hydro_checks["max_drag_positive_power"],drag_power)
        for i in range(N):
            p=state[i,:3]; v=vel[i]
            out[i,:3]=v; out[i,3:6]=ar[i]+adc[i]+ac[i]+ad[i]
            out[i,6]=np.dot(v,ar[i]); out[i,7]=np.dot(v,adc[i])
            out[i,8]=np.dot(v,ac[i]); out[i,9]=np.dot(v,ad[i])
            out[i,10]=p[0]*ar[i,1]-p[1]*ar[i,0]
            out[i,11]=p[0]*ac[i,1]-p[1]*ac[i,0]
            out[i,12]=p[0]*ad[i,1]-p[1]*ad[i,0]
        return out.ravel()
    res=solve_ivp(f,(0,float(t_eval[-1])),y0,method="DOP853",t_eval=t_eval,
        rtol=3e-9,atol=3e-11,max_step=2*math.pi/64)
    if not res.success: raise RuntimeError(res.message)
    dat=res.y.T.reshape(-1,N,STRIDE)
    pos=dat[:,:,:3]; vel=dat[:,:,3:6]
    radius=np.linalg.norm(pos[:,:,:2],axis=2)
    theta=np.unwrap(np.arctan2(pos[:,:,1],pos[:,:,0]),axis=0)
    turns=(theta[-1]-theta[0])/(2*math.pi)
    rsep=np.linalg.norm(pos[:,0]-pos[:,1],axis=1)
    escaped=bool(np.max(radius)>=0.995 or np.max(np.abs(pos[:,:,2]))>=1.0)
    hard_collision=bool(np.min(rsep)<sum(mapping["radii_m"])/mapping["trap_radius_m"])
    start_relax_periods=max(0.0,5.0/(2*math.pi*float(np.min(lam))))
    relax_idx=min(len(res.t)-1,int(np.searchsorted(res.t,start_relax_periods*2*math.pi)))
    after_turns=(theta[-1]-theta[relax_idx])/(2*math.pi)
    windows=[]
    for lo,hi in ((.65,.78),(.78,.89),(.89,1.0)):
        i0=max(relax_idx,int(lo*(len(res.t)-1))); i1=max(i0+1,int(hi*(len(res.t)-1)))
        i1=min(i1,len(res.t)-1)
        windows.append(((theta[i1]-theta[i0])/(2*math.pi)).tolist())
    signs=np.sign(np.asarray(windows))
    late_signs_opposite=bool(not escaped and not hard_collision and
        ((np.all(signs[:,0]>0) and np.all(signs[:,1]<0)) or
         (np.all(signs[:,0]<0) and np.all(signs[:,1]>0))))
    # Require sustained motion in each late interval; no near-stationary sign is accepted.
    window_activity=bool(np.all(np.abs(np.asarray(windows))>=0.1))
    late_counter=bool(late_signs_opposite and window_activity)
    ten_turn=bool(late_counter and window_activity and np.all(np.abs(after_turns)>=10.0))
    # Total scaled mechanical energy; masses are relative and chi is matching normalized.
    def energy(k):
        e=0.0
        for i in range(N):
            x,y,z=pos[k,i]; v=vel[k,i]
            e += mass[i]*(0.5*np.dot(v,v)+beta[i]*(z*z-0.5*(x*x+y*y)))
        e += chi*charge[0]*charge[1]/rsep[k]
        return float(e)
    workrf=float(sum(mass[i]*dat[-1,i,6] for i in range(N)))
    workdc=float(sum(mass[i]*dat[-1,i,7] for i in range(N)))
    workc=float(sum(mass[i]*dat[-1,i,8] for i in range(N)))
    workdrag=float(sum(mass[i]*dat[-1,i,9] for i in range(N)))
    # The endpoint energy already contains DC and Coulomb storage terms;
    # therefore only RF and hydrodynamic drag work remain on the RHS.
    eres=energy(-1)-energy(0)-workrf-workdrag
    # Detailed forces/Re/Kn and pair far-field validity on the actual trajectory.
    radii_m=np.asarray(mapping["radii_m"]); R=mapping["trap_radius_m"]
    Re_scale=np.asarray(mapping["re_prefactor_per_dimensionless_speed"])
    Re_series=np.linalg.norm(vel,axis=2)*Re_scale[None,:]
    ratios=rsep*R/float(np.max(radii_m))
    physical_collision_distance=(radii_m.sum()/R)
    lz=np.stack([
        mass[i]*(pos[:,i,0]*vel[:,i,1]-pos[:,i,1]*vel[:,i,0]) for i in range(N)],axis=1)
    # Per particle force and torque work channels remain separately recoverable in NPZ.
    rowout={"checkpoint":str(row.run_id),"model":model,"periods":rel,
        "retained":not escaped,"escaped":escaped,"hard_sphere_overlap":hard_collision,
        "late_signs_opposite":late_signs_opposite,
        "late_window_activity_ge_0p1_turns":window_activity,
        "counter_rotating_late":late_counter,"counter_rotating_10turn_after_relax":ten_turn,
        "turns1":float(turns[0]),"turns2":float(turns[1]),
        "turns1_after_relax":float(after_turns[0]),"turns2_after_relax":float(after_turns[1]),
        "late_window_turns":json.dumps(windows),
        "r1_mean":float(np.mean(radius[:,0])),"r2_mean":float(np.mean(radius[:,1])),
        "r1_std":float(np.std(radius[:,0])),"r2_std":float(np.std(radius[:,1])),
        "r1_max":float(np.max(radius[:,0])),"r2_max":float(np.max(radius[:,1])),
        "minimum_distance_over_R":float(np.min(rsep)),"mean_distance_over_R":float(np.mean(rsep)),
        "minimum_distance_over_sum_radii":float(np.min(rsep)*R/sum(radii_m)),
        "minimum_distance_over_max_radius":float(np.min(rsep)*R/max(radii_m)),
        "Re1_max":float(np.max(Re_series[:,0])),"Re2_max":float(np.max(Re_series[:,1])),
        "Re1_mean":float(np.mean(Re_series[:,0])),"Re2_mean":float(np.mean(Re_series[:,1])),
        "Kn1":float(mapping["knudsen"][0]),"Kn2":float(mapping["knudsen"][1]),
        "pair_Oseen_farfield_min_r_over_max_a":float(np.min(ratios)),
        "physical_Oseen_Re_lt_0p1_all":bool(np.max(Re_series)<0.1),
        "continuum_Kn_lt_0p01_all":bool(max(mapping["knudsen"])<0.01),
        "pair_Oseen_farfield_r_over_max_a_gt_5":bool(np.min(ratios)>5.0),
        "mass_density2_kg_m3":float(mapping["density2_kg_m3"]),
        "lambda1_used":float(lam[0]),"lambda2_used":float(lam[1]),
        "lambda2_checkpoint":float(row.lambda2),"lambda2_relative_mapping_error":float(lam[1]/row.lambda2-1),
        "U0_V":float(mapping["U0_V"]),"Vdc_V":float(mapping["Vdc_V"]),
        "q1_C":float(mapping["charge_C"][0]),"q2_C":float(mapping["charge_C"][1]),
        "q1_e":float(mapping["charge_C"][0]/E_CHARGE),"q2_e":float(mapping["charge_C"][1]/E_CHARGE),
        "RF_frequency_Hz":float(mapping["omega_rf_rad_s"]/(2*math.pi)),
        "RF_work":workrf,"DC_work":workdc,"Coulomb_work":workc,"drag_work":workdrag,
        "energy_balance_residual":float(eres),
        "coulomb_torque_impulse1":float(mass[0]*dat[-1,0,11]),
        "coulomb_torque_impulse2":float(mass[1]*dat[-1,1,11]),
        "hydrodynamic_torque_impulse1":float(mass[0]*dat[-1,0,12]),
        "hydrodynamic_torque_impulse2":float(mass[1]*dat[-1,1,12]),
        "Lz1_start":float(lz[0,0]),"Lz1_end":float(lz[-1,0]),
        "Lz2_start":float(lz[0,1]),"Lz2_end":float(lz[-1,1]),
        "minimum_pair_mobility_det_perp":float(hydro_checks["min_det_perp"]) if np.isfinite(hydro_checks["min_det_perp"]) else None,
        "minimum_pair_mobility_det_parallel":float(hydro_checks["min_det_parallel"]) if np.isfinite(hydro_checks["min_det_parallel"]) else None,
        "max_total_drag_power":float(hydro_checks["max_drag_positive_power"])}
    return rowout, {"t":res.t,"state":dat,"pos":pos,"velocity":vel,"theta":theta,
                    "distance":rsep,"parameters":np.asarray([row.alpha1,row.alpha2,row.beta1,row.beta2,
                        lam[0],lam[1],row.mass2_over_mass1,row.charge2_over_charge1,row.chi,row.phase0])}


def physical_mapping(row):
    R=PHYS["trap_radius_m"]; Omega=PHYS["omega_rf_rad_s"]
    mu=PHYS["dynamic_viscosity_Pa_s"]; rhop1=PHYS["particle1_material_density_kg_m3"]
    # Solve Stokes lambda1=6*pi*mu*a1/(m1*Omega) with spherical m1=rho*4*pi*a^3/3.
    a1=math.sqrt(9.0*mu/(2.0*rhop1*Omega*float(row.lambda1)))
    m1=4.0/3.0*math.pi*a1**3*rhop1
    mr=float(row.mass2_over_mass1); m2=mr*m1
    a2=float(row.lambda2)*m2*Omega/(6.0*math.pi*mu)
    rho2=m2/(4.0/3.0*math.pi*a2**3)
    radii=np.array([a1,a2]); masses=np.array([m1,m2])
    q1=math.sqrt(float(row.chi)*4.0*math.pi*EPS0*m1*R**3*Omega**2) if float(row.chi)>0 else 0.0
    U0=(float(row.alpha1)*m1*R**2*Omega**2/q1) if q1>0 else float("nan")
    q2=float(row.charge2_over_charge1)*q1
    vdc_over=float(row.beta1)/float(row.alpha1)
    qom_ratio=(q2/m2)/(q1/m1) if q1>0 else float("nan")
    alpha2_pred=float(row.alpha1)*qom_ratio
    re_pref=2.0*radii*PHYS["fluid_density_kg_m3"]*R*Omega/mu
    Kn=PHYS["mean_free_path_m"]/radii
    nu=mu/PHYS["fluid_density_kg_m3"]
    Wo=radii*np.sqrt(Omega/nu)
    alpha1_back=q1*U0/(m1*R**2*Omega**2) if q1>0 else float("nan")
    chi_back=q1**2/(4.0*math.pi*EPS0*m1*R**3*Omega**2) if q1>0 else 0.0
    beta2_back=alpha2_pred*vdc_over
    return {"trap_radius_m":R,"omega_rf_rad_s":Omega,"fluid_mu_Pa_s":mu,
        "fluid_density_kg_m3":PHYS["fluid_density_kg_m3"],"particle_density1_kg_m3":rhop1,
        "density2_kg_m3":rho2,"radii_m":radii.tolist(),"mass_kg":masses.tolist(),
        "lambda_physical":[6*math.pi*mu*a1/(m1*Omega),6*math.pi*mu*a2/(m2*Omega)],
        "charge_C":[q1,q2],"U0_V":U0,"Vdc_V":vdc_over*U0,
        "qom2_over_qom1_physical":qom_ratio,"alpha2_predicted":alpha2_pred,
        "alpha2_checkpoint":float(row.alpha2),"alpha2_relative_error":float(alpha2_pred/row.alpha2-1),
        "alpha1_recovered":alpha1_back,"alpha1_checkpoint":float(row.alpha1),
        "chi_recovered":chi_back,"chi_checkpoint":float(row.chi),
        "beta2_recovered":beta2_back,"beta2_checkpoint":float(row.beta2),
        "charge2_ratio_checkpoint":float(row.charge2_over_charge1),
        "re_prefactor_per_dimensionless_speed":re_pref.tolist(),"knudsen":Kn.tolist(),
        "kinematic_viscosity_m2_s":nu,"womersley_a_sqrt_Omega_over_nu":Wo.tolist(),
        "womersley_quasi_steady_strict_lt_0p1":bool(np.max(Wo)<0.1),
        "womersley_borderline_0p1_to_0p3":bool(0.1<=np.max(Wo)<0.3),
        "womersley_unsteady_corrections_required_ge_0p3":bool(np.max(Wo)>=0.3),
        "sphere_relation_lambda": "lambda=9*mu/(2*rho_particle*a^2*Omega)",
        "derived_sphere_relation_Wo_squared": "Wo^2=9*rho_fluid/(2*rho_particle*lambda)",
        "mean_free_path_m":PHYS["mean_free_path_m"]}


def rpy_kernel_checks():
    mu=PHYS["dynamic_viscosity_Pa_s"]
    records=[]
    for a1,a2 in ((10e-6,10e-6),(10e-6,15e-6)):
        radii=np.array([a1,a2]); zeta=6*np.pi*mu*radii
        samples=sorted(set([0.0,1e-6*max(a1,a2),abs(a1-a2)*0.5,
            max(abs(a1-a2)*1.01,1e-9),a1+a2-1e-9*(a1+a2),
            a1+a2,a1+a2+1e-9*(a1+a2),3*(a1+a2)]))
        by_distance=[]
        for d in samples:
            hp,hl=rpy_cross_perp_parallel(np.array([d,0.,0.]),a1,a2,mu)
            H=np.diag([hl,hp,hp]); I=np.eye(3)
            M=np.block([[I/zeta[0],H],[H,I/zeta[1]]])
            eig=np.linalg.eigvalsh(M)
            by_distance.append({"r_m":d,"r_over_sum_radii":d/(a1+a2),
                "mu_perp":hp,"mu_parallel":hl,"mobility_min_eigenvalue":float(eig[0]),
                "mobility_min_eigenvalue_relative_to_max":float(eig[0]/eig[-1])})
        contact=a1+a2
        hleft=np.asarray(rpy_cross_perp_parallel(np.array([contact*(1-1e-10),0,0]),a1,a2,mu))
        hright=np.asarray(rpy_cross_perp_parallel(np.array([contact*(1+1e-10),0,0]),a1,a2,mu))
        records.append({"radii_m":[a1,a2],"distances":by_distance,
            "relative_mobility_jump_at_contact":float(np.max(np.abs(hleft-hright)/np.maximum(np.abs(hright),1e-300)))})
    return {"model":"generalized translational RPY, unequal radii, regularized overlap branches",
        "source":"Zuk et al., J. Fluid Mech. 741 (2014), Eq. 3.1, DOI 10.1017/jfm.2013.668",
        "cases":records}


def save_checkpoint(rows):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"matched_hydrodynamics.csv"
    pd.DataFrame(rows).to_csv(path,index=False)


def main():
    modes,coeff=load_spectrum()
    allrows=pd.read_csv(BASE)
    selected=[allrows.loc[allrows.run_id==rid].iloc[0] for rid in CASE_IDS]
    param_rows=[]; result_rows=[]; trajectories={}
    mapping_doc={"physical_assumptions":PHYS,"cases":{}}
    (OUT/"rpy_kernel_checks.json").write_text(json.dumps(rpy_kernel_checks(),indent=2),encoding="utf-8")
    previous=OUT/"matched_hydrodynamics.csv"
    done=set()
    if previous.exists():
        result_rows=pd.read_csv(previous).to_dict("records")
        done={(x["checkpoint"],x["model"]) for x in result_rows}
    for row in selected:
        mapping=physical_mapping(row)
        mapping_doc["cases"][row.run_id]=mapping
        # Check recovered physical RF/DC/coulomb parameters against checkpoint.
        if abs(mapping["alpha2_relative_error"])>1e-8:
            raise ValueError(f"charge/mass mismatch in mapping for {row.run_id}")
        for model in MODELS:
            if (row.run_id,model) in done:
                continue
            res,traj=solve_case(row,model,modes,coeff,mapping)
            result_rows.append(res)
            trajectories[f"{row.run_id}__{model}"]=traj
            np.savez_compressed(OUT/f"{row.run_id}__{model}.npz",**traj)
            print(row.run_id,model,"retained",res["retained"],"turns",round(res["turns1"],4),round(res["turns2"],4),
                  "Re_max",round(max(res["Re1_max"],res["Re2_max"]),4),"sep/a",round(res["minimum_distance_over_max_radius"],2),flush=True)
            save_checkpoint(result_rows)
        (OUT/"physical_mappings.json").write_text(json.dumps(mapping_doc,indent=2),encoding="utf-8")
    summary={"checkpoint_source":str(BASE.relative_to(V4)),"matched_cases":CASE_IDS,
        "models":list(MODELS),"same_initial_states":True,"same_rf_dc_coulomb":True,
        "periods_per_case":int(selected[0].periods),"steps_in_max_step_per_rf_period":64,
        "physical_mapping":PHYS,
        "case_rows":result_rows,
        "retained_fraction_by_model":{},"late_counter_fraction_by_model":{},
        "ten_turn_counter_fraction_by_model":{}}
    for model in MODELS:
        rows=[x for x in result_rows if x["model"]==model]
        summary["retained_fraction_by_model"][model]=sum(x["retained"] for x in rows)/len(rows)
        summary["late_counter_fraction_by_model"][model]=sum(x["counter_rotating_late"] for x in rows)/len(rows)
        summary["ten_turn_counter_fraction_by_model"][model]=sum(x["counter_rotating_10turn_after_relax"] for x in rows)/len(rows)
    (OUT/"matched_hydrodynamics_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"retained_fraction_by_model":summary["retained_fraction_by_model"],
        "late_counter_fraction_by_model":summary["late_counter_fraction_by_model"],
        "ten_turn_counter_fraction_by_model":summary["ten_turn_counter_fraction_by_model"]},indent=2),flush=True)


if __name__=="__main__": main()
