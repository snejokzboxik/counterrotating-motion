from pathlib import Path
import json,hashlib,collections
import numpy as np
import attractor_search as a
OUT=a.OUT; ROOT=OUT.parent

def loadj(name): return a.read_json(OUT/name)
def sha(path): return a.sha256(path)

def main():
    cfg=loadj('candidate_parameters.json');base=loadj('baseline_reproduction.json')['rows']
    grid=[json.loads(x) for x in (OUT/'parameter_grid_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    uni=[json.loads(x) for x in (OUT/'univariate_scan_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    sob=[json.loads(x) for x in (OUT/'multivariate_sobol_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    crows=[json.loads(x) for x in (OUT/'coulomb_sensitivity_checkpoint.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    qra=loadj('variational_audit.json');qmap={x['candidate']:x for x in qra['rows']}
    phase=collections.defaultdict(list)
    for row in qra['phase_perturbation_rows']: phase[row['candidate']].append(row)
    dop=loadj('independent_dop853_validation.json');dopmap={x['candidate']:x for x in dop['rows']}
    step=loadj('step_refinement_3000p.json');stepmap={x['candidate']:x for x in step['rows']}
    basin=loadj('basin_ensemble_summary.json')
    long=[]
    for c in cfg['candidates']:
        if c['legacy_candidate']=='refine12_sobol17':
            long.append((c,loadj('refine12_240k_corrected_drag_summary.json'),'refine12_240k_corrected_drag_trajectory.npz'))
        else:
            tag=f"{c['legacy_candidate']}_50000_common_corrected"
            long.append((c,loadj(tag+'_summary.json'),tag+'_trajectory.npz'))
    candidates=[]
    for c,s,npz in long:
        name=c['id'];summary_base=next(x for x in base if x['candidate']==name)
        qr=loadj(f"lyapunov_{c['legacy_candidate']}_2000p.json")
        pert=loadj(f"finite_perturbations_{c['legacy_candidate']}.json")
        sens=sorted((x for x in crows if x['candidate']==name),key=lambda x:x['sC'])
        candidates.append({
          'id':name,'legacy_name':c['legacy_candidate'],'evidence_status':'LONG_LIVED_COUNTERROTATION; ATTRACTOR_NOT_ESTABLISHED',
          'field_parameters':{'A':c['p_unadjusted'][0],'B':c['p_unadjusted'][1],'nu':c['p_unadjusted'][2],'beta':c['p_unadjusted'][3],'RF_frequencies_Hz':[c['f1_Hz'],c['f2_Hz']],
            'MFS_modes':cfg['model_source']['harmonics_k'],'MFS_families':cfg['model_source']['harmonics_family'],'MFS_complex_coefficients':cfg['model_source']['harmonics_coeff']},
          'particle_parameters':{'alpha1':c['alpha1'],'alpha2':c['alpha2'],'lambda1_corrected':c['lambda_corrected_divide_Cc'][0],'lambda2_corrected':c['lambda_corrected_divide_Cc'][1],
            'mu_m2_over_m1':c['mu'],'Q_q2_over_q1':c['qratio'],'C12':c['p_corrected'][5],'particle_radius_over_R':[c['a1_over_R'],c['a2_over_R']],
            'diameters_um':c['diameters_um'],'charges_e':c['charges_e'],'gravity_z':c['gravity_z'],'Re_per_unit_dimensionless_speed': [c['re1_per_speed'],c['re2_per_speed']],
            'polarization':'disabled','interparticle_hydrodynamic_coupling':'disabled'},
          'initial_condition':{'selection':'shared frozen rest start, V12B start index 7, seed 20261850','state':cfg['matched_common_start']['initial'],'RF_phases_rad':cfg['matched_common_start']['phase'],
            'long_replay_start':cfg['refine12_240k_seed']['initial'] if c['legacy_candidate']=='refine12_sobol17' else None},
          'integration':{'method':'Numba RK4 FP64, full 8-complex MFS, finite-sphere event/contact checks','steps_per_RF_period':64,'periods':s['completed_periods'],
            'transient_periods':s.get('transient_periods',5000),'trajectory_file':npz,'baseline_15k_file':f"baseline_{name}_15000p_rk464.npz",
            'trajectory_initial_condition':'refine12_240k_seed (V12B long rest-start index 54)' if c['legacy_candidate']=='refine12_sobol17' else 'matched_common_start (rest index 7)',
            'source_integrator_sha256':cfg['model_source']['integrator_sha256'],'pair_integrator_sha256':cfg['model_source']['verified_pair_integrator_sha256']},
          'late_observables':{'turns':s['late_turns'],'frequency_Hz':s['late_frequency_Hz'],'mean_radius_over_R':s['late_mean_radius_R'],
            'minimum_pair_distance_over_R':s['minimum_pair_distance_R'],'retained':s['retained'],'consistent_counterrotation_all_chunks':s['all_late_chunks_counterrotating']},
          'independent_DOP853':dopmap[name],
          'step_refinement_RK4_128':stepmap[name],
          'basin_subsample_N8_per_ensemble':{x['ensemble']:{'n':x['n'],'strict_counterrotating':x['counts']['STRICT_CR'],'wall_contact':x['counts']['WALL_CONTACT'],
            'strict_fraction_among_all_starts':x['strict_cr_fraction'],'retained_fraction':x['retained_fraction']}
            for x in basin['candidate_ensemble_results'] if x['candidate']==name},
          'attraction_tests':{'conditional_mechanical_QR_periods':qr['periods'],'conditional_top_exponent_per_tau':qr['rows'][-1]['top_per_tau'],
            'mechanical_perturbations':[{k:r[k] for k in ('particle','direction','initial_separation','classification','minimum_pair_distance_R','checkpoints')}
              for r in pert['rows']],
            'one_degree_RF_phase_trials':[{k:r[k] for k in ('phase_channel','delta_degrees','classification','turns','window_turns','min_pair_distance_R')}
              for r in phase[name]],'interpretation':'Finite-time evidence only; no global basin fraction and no asymptotic attraction claim.'},
          'Coulomb_scale_diagnostic':[{k:x[k] for k in ('sC','frequency_Hz','pair_min_R','screen_class','ten_turn_requirement_met')} for x in sens],
          'source_note':'The dimensional map is a legacy candidate mapping, not a jointly identified physical particle pair.'})
    a.write_json(ROOT/'ATTRACTOR_CANDIDATES.json',{'schema_version':'1.0','study':'V17C.20B','source_base_sha':cfg['base_sha'],
      'code_sha256':{p.name:sha(p) for p in (ROOT/'code').glob('*.py')},'candidates':candidates,
      'global_interpretation':'These are reproducible finite-horizon counterrotating trajectories. They are not labeled proven attractors.'})
    pairs=[('alpha1','alpha2'),('alpha1','lambda1'),('alpha2','lambda2'),('lambda1','lambda2'),('mu','C12'),('alpha1','C12'),('alpha2','C12')]
    cov={'study':'V17C.20B','source_base_sha':cfg['base_sha'],'mathematical_scan':{'nominal_multipliers':[0.6,0.8,1.0,1.2,1.4],
      'central_point':'excluded from the off-centre grid; separately covered by the three nominal baselines',
      'runs':len(grid),'unique_ids':len({r['run_id'] for r in grid}),'per_candidate':{c['id']:sum(r['candidate']==c['id'] for r in grid) for c in cfg['candidates']},
      'horizon_periods':3000,'RK4_steps_per_period':32,'frozen_starts':'single shared rest start index 7; classifications are short-screen signs only',
      'all_retained':all(r['retained'] for r in grid),'strict_ten_turn_cases':sum(r['ten_turn_requirement_met'] for r in grid),
      'covered_pairwise_slices':[{'x':x,'y':y,'coverage':'5x5 multiplier grid excluding central nominal point'} for x,y in pairs],
      'outcome_counts':{c['id']:{sl:dict(collections.Counter(r['screen_class'] for r in grid if r['candidate']==c['id'] and r['slice']==sl))
        for sl in sorted({r['slice'] for r in grid if r['candidate']==c['id']})} for c in cfg['candidates']}},
      'univariate_scan':{'runs':len(uni),'unique_ids':len({r['run_id'] for r in uni}),'nominal_multipliers':[0.6,0.8,1.0,1.2,1.4],
        'off_nominal_runs_per_candidate':24,'off_nominal_parameters':['alpha1','alpha2','lambda1','lambda2','mu','C12'],
        'horizon_periods':3000,'RK4_steps_per_period':32,'frozen_start_index':7,
        'Q_handling':'Derived algebraically as mu*alpha2/alpha1; a separate Q-only axis would violate the selected independent-acceleration/C12 parameterization.',
        'per_candidate_counts':{c['id']:sum(r['candidate']==c['id'] for r in uni) for c in cfg['candidates']},
        'outcome_counts':{c['id']:{par:dict(collections.Counter(r['screen_class'] for r in uni if r['candidate']==c['id'] and r['parameter']==par))
          for par in ['alpha1','alpha2','lambda1','lambda2','mu','C12']} for c in cfg['candidates']}},
      'multivariate_sobol':{'runs':len(sob),'unique_ids':len({r['run_id'] for r in sob}),'points_per_candidate':32,'seed_base':20261123,
        'parameters':['alpha1','alpha2','lambda1','lambda2','mu','C12'],'multiplier_range':[0.6,1.4],
        'horizon_periods':3000,'RK4_steps_per_period':32,'mathematical_only':True,
        'per_candidate_outcomes':{c['id']:{**dict(collections.Counter(r['screen_class'] for r in sob if r['candidate']==c['id'])),
            'retained':sum(r['retained'] for r in sob if r['candidate']==c['id']),
            'ten_turn':sum(r['ten_turn_requirement_met'] for r in sob if r['candidate']==c['id'])} for c in cfg['candidates']}},
      'coulomb_sensitivity':{'multipliers':[x['sC'] for x in sorted(crows,key=lambda x:x['sC']) if x['candidate']==cfg['candidates'][0]['id']],
        'runs':len(crows),'runs_per_candidate':{c['id']:sum(r['candidate']==c['id'] for r in crows) for c in cfg['candidates']},
        'horizon_periods':5000,'RK4_steps_per_period':32,'strict_ten_turn_cases':sum(r['ten_turn_requirement_met'] for r in crows),
        'all_screen_counterrotating':all(r['screen_class']=='SCREEN_CR' for r in crows),
        'mathematical_only':True,'note':'C varied independently; not a physical pair scan.'},
      'physical_linked_scan':{'status':'NOT_PERFORMED','reason':'The source candidate dimensional mappings are legacy assumptions; no joint material/charge/size distribution was specified or independently validated.'},
      'uncovered_parameter_pairs':['alpha1-lambda2','alpha2-lambda1','alpha1-mu','alpha2-mu','lambda1-mu','lambda2-mu','Q with lambda/mu/C separately','size/drag/C linked through material density, charge and common gas'],
      'basin_subsample':{'runs':sum(x['n'] for x in basin['candidate_ensemble_results']),'N_per_candidate_ensemble':8,
        'systematic_indices':basin['indices_per_ensemble'],'periods':basin['periods'],'steps_per_RF_period':basin['steps_per_rf'],'source_starts_sha256':basin['source_frozen_starts_sha256'],
        'sampling_note':'deterministic subset, descriptive only; not IID probability estimate',
        'candidate_ensemble_results':basin['candidate_ensemble_results']},
      'global_limiter':'No full 12D basin map; the surface scans use one shared rest start. The N=8 frozen-start panel is small and deterministic.'}
    a.write_json(ROOT/'PARAMETER_COVERAGE.json',cov)
if __name__=='__main__':main()
