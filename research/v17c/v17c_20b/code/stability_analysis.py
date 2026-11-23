from pathlib import Path
import json,math
import numpy as np
import attractor_search as a
from variational_qr import validate_jacobian,lyapunov_qr,advance_augmented

OUT=a.OUT

def trace(case,state,phase,periods,steps=64,stride=100):
    field=a.load_field();p,q=a.arrays(case,field)
    return a.PAIR._integrate_trace(np.asarray(state,float),np.asarray(phase,float),p,field['family'],field['k'],field['coeff'],q,
      case['a1_over_R'],case['a2_over_R'],case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'],
      periods,steps,0,steps*stride)

def classify(raw,case,periods,skip):
    turns=raw[4];windows=raw[5];status=int(raw[18])
    if status==1: return 'PARTICLE_CONTACT'
    if status==2: return 'WALL_CONTACT'
    if status==3: return 'UNRESOLVED'
    signs=np.sign(np.asarray(windows))
    if min(abs(np.asarray(turns)))>=10 and np.all(signs[:,0]*signs[:,1]<0):return 'STRICT_CR'
    if np.all(signs[:,0]*signs[:,1]<0):return 'PROVISIONAL_CR'
    if np.all(signs[:,0]*signs[:,1]>0):return 'PROVISIONAL_CO'
    return 'UNRESOLVED'

def main():
    cfg=a.read_json(OUT/'candidate_parameters.json');cases=cfg['candidates'];field=a.load_field();fresults=[];qresults=[];phase_results=[]
    for case in cases:
        cid=case['id']
        tag=('refine12_240k_corrected_drag' if case['legacy_candidate']=='refine12_sobol17' else f"{case['legacy_candidate']}_50000_common_corrected")
        z=np.load(OUT/f'{tag}_trajectory.npz');state=z['states'][-1].copy();periods=int(z['periods']);phase0=z['phase'];
        p,q=a.arrays(case,field);phase=(phase0+np.array([1.0,float(p[2])])*2*math.pi*periods)%(2*math.pi)
        # Audit the analytic Jacobian and RK4 augmented base flow against an independent call to the audited RHS.
        fresults.append({'candidate':cid,**validate_jacobian(case,seed=20261123)})
        dt=2*math.pi/32
        yaug,M,tloc=advance_augmented(state,np.eye(12),0.0,dt,20*32,p,phase,field['family'],field['k'],field['coeff'],q,
          case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'],case['a1_over_R'],case['a2_over_R'])
        raw20=a.SIM.integrate_one(state,phase,p,field['family'],field['k'],field['coeff'],20,32,0,q,case['re1_per_speed'],0,
          case['a1_over_R'],case['a2_over_R'],case['re2_per_speed'],case['gravity_z'],2,1.0)
        fresults[-1]['augmented_base_vs_audited_RK4_20p_max_abs_error']=float(np.max(abs(yaug-raw20[0])))
        # Conditional mechanical QR spectrum at fixed externally prescribed RF phases.
        qrun=lyapunov_qr(case,state,phase,periods=2000,steps=32,qr_periods=20)
        a.write_json(OUT/f'lyapunov_{case["legacy_candidate"]}_2000p.json',qrun)
        fresults[-1]['lyapunov_qr_file']=f'lyapunov_{case["legacy_candidate"]}_2000p.json'
        # Reference continuation at the exact late forcing phase.
        ref=trace(case,state,phase,5000,64,100);refstates=ref[1];refidx=ref[2]/64
        perturbations=[]
        rbar=np.asarray(a.read_json(OUT/f'{tag}_summary.json')['late_mean_radius_R'],float)
        fr=np.asarray(a.read_json(OUT/f'{tag}_summary.json')['late_frequency_Hz'],float)
        vsecs=rbar*np.abs(fr)/case['f1_Hz']
        for particle,j in ((0,0),(1,6)):
            rad=math.hypot(state[j],state[j+1]);ur=state[j:j+2]/max(rad,1e-30);ut=np.array([-ur[1],ur[0]])
            vv=state[j+3:j+6];vsec=float(vsecs[particle])
            specs=[('radial_position_1e-6_orbit_radius','pos',1e-6*rad,ur),
              ('tangential_position_1e-6_orbit_radius','pos',1e-6*rad,ut),
              ('radial_velocity_1e-3_secular_speed','vel',1e-3*vsec,ur)]
            for label,kind,amp,direction in specs:
                y0=state.copy()
                if kind=='pos': y0[j:j+2]+=amp*direction
                else:y0[j+3:j+5]+=amp*direction
                d0=float(np.linalg.norm(y0-state))
                pert=trace(case,y0,phase,5000,64,100)
                sstates=pert[1];sidx=pert[2]/64
                n=min(len(refstates),len(sstates));sep=np.linalg.norm(refstates[:n]-sstates[:n],axis=1)
                times=refidx[:n]
                points=[]
                for target in (100,500,1000,2500,5000):
                    ix=int(np.argmin(abs(times-target)));d=max(float(sep[ix]),1e-300)
                    points.append({'periods':float(times[ix]),'distance':d,'distance_over_initial':d/max(d0,1e-300),
                      'finite_time_exponent_per_tau':float(math.log(d/max(d0,1e-300))/(2*math.pi*max(times[ix],1)))})
                perturbations.append({'particle':particle+1,'direction':label,'initial_separation':d0,
                  'status':int(pert[18]),'completed_periods':int(pert[17]),'classification':classify(pert,case,5000,0),
                  'minimum_pair_distance_R':float(pert[10]),'separation_history':sep[:n].tolist(),'period_index':times[:n].tolist(),
                  'checkpoints':points})
        a.write_json(OUT/f'finite_perturbations_{case["legacy_candidate"]}.json',{'candidate':cid,'reference_periods':periods,
          'phase_preserved_at_reference_end':phase.tolist(),'perturbation_scale_definitions':{'position':'1e-6 of current orbital radius','velocity':'1e-3 of late secular tangential speed'},
          'rows':perturbations,'interpretation':'Same fixed forcing phases; finite-time local sensitivity panel, not a global basin estimate.'})
        # Independent forcing-phase perturbations are classified separately from mechanical tangent growth.
        for kphase in (0,1):
            ph=phase.copy();ph[kphase]+=math.radians(1.0)
            rawp=trace(case,state,ph,15000,64,1000)
            phase_results.append({'candidate':cid,'phase_channel':kphase+1,'delta_degrees':1.0,
              'classification':classify(rawp,case,15000,1000),'retained':bool(rawp[18]==0 and rawp[17]>=15000),
              'completed_periods':int(rawp[17]),'turns':np.asarray(rawp[4]).tolist(),'window_turns':np.asarray(rawp[5]).tolist(),
              'min_pair_distance_R':float(rawp[10]),'mean_radius_R':np.asarray(rawp[11]).tolist()})
        print(f'{cid}: Jacobian err={fresults[-1]["max_relative_jacobian_error"]:.2e}; QR top={qrun["rows"][-1]["top_per_tau"]:.4g}/tau; perturb tests={len(perturbations)}',flush=True)
    a.write_json(OUT/'variational_audit.json',{'rows':fresults,'phase_perturbation_rows':phase_results,
      'phase_note':'RF phase shifts change the forcing point on its phase torus and are reported as basin sensitivity, not mechanical Lyapunov directions.'})

if __name__=='__main__':main()
