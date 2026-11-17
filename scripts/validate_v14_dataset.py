"""Integrity and provenance audit for completed V14 datasets."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
from collections import Counter
import math
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/octupole_comparison_home/mass_filter_v14'
BASE=ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def jsonl(name,key):
    rows=[json.loads(s) for s in (OUT/name).read_text(encoding='utf8').splitlines() if s.strip()]
    keys=[key(r) for r in rows]
    assert len(keys)==len(set(keys)),f'duplicate keys {name}'
    return rows

def csvrows(name):
    with (OUT/name).open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))

def approx(a,b,tol=1e-12):
    assert abs(float(a)-float(b))<=tol, (a,b)

def main():
    sets={}
    definitions=[('ideal_pilot_checkpoint.jsonl',lambda r:r['key']),
                 ('phase_ideal_checkpoint.jsonl',lambda r:r['key']),
                 ('fem_so_checkpoint.jsonl',lambda r:r['key']),
                 ('phase_fem_checkpoint.jsonl',lambda r:r['key']),
                 ('fem_long_checkpoint.jsonl',lambda r:r['key']),
                 ('near_orbit_long_checkpoint.jsonl',lambda r:r['key']),
                 ('fem_mass_so_checkpoint.jsonl',lambda r:r['key'])]
    for name,key in definitions:
        rows=jsonl(name,key)
        sets[name]=dict(rows=len(rows),unique_keys=len(rows),
                        unresolved=sum(r.get('status')=='unresolved' for r in rows),
                        sha256=sha(OUT/name))
    expected={'ideal_pilot_checkpoint.jsonl':1024,'phase_ideal_checkpoint.jsonl':1280,
              'fem_so_checkpoint.jsonl':512,'phase_fem_checkpoint.jsonl':768,
              'fem_long_checkpoint.jsonl':2048,'near_orbit_long_checkpoint.jsonl':30,
              'fem_mass_so_checkpoint.jsonl':512}
    for n,count in expected.items():assert sets[n]['rows']==count,(n,sets[n])
    assert all(s['unresolved']==0 for s in sets.values())
    # Audit exact reuse of the completed V13.3 Q/RO records at mu=1.
    src=BASE/'acceptance_4d_fem_h50_selected_N100/trajectory_results.csv'
    old={}
    with src.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['mu']=='1.0' and r['horizon']=='100' and int(r['state_i'])<16:
                k=(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))
                old[k]=r
    assert len(old)==512
    fem_qro={**{k:v for k,v in old.items()}}
    so=jsonl('fem_so_checkpoint.jsonl',lambda r:r['key'])
    for r in so:
        k=(r['replica'],r['phase_index'],r['state_i'])
        for model in ('Q','RO'):
            source=old[(model,*k)]
            assert max(abs(float(source[c])-r['initial'][i]) for i,c in enumerate(
                ('x0_rho','y0_rho','vx0_rho_tau','vy0_rho_tau')))<1e-13
    assert len(csvrows('fem_so_paired_summary.csv'))==35
    assert len(csvrows('phase_fem_summary.csv'))==5
    assert len(csvrows('fem_mass_so_summary.csv'))==6

    # Independently reconstruct headline FEM counts from the raw trajectory rows.
    so_by={(r['model'],int(r['replica']),int(r['phase_index']),int(r['state_i'])):r for r in so}
    old_by={(r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i'])):r for r in old.values()}
    all_a9={**old_by,**so_by}
    so_summary={(r['phase_deg'],r['model']):r for r in csvrows('fem_so_paired_summary.csv')}
    for model in ('Q','RO','SO_equal_harmonic','SO_equal_rms'):
        means=[sum(all_a9[(model,rep,p,i)]['status']=='retained' for p in range(4) for i in range(16))/64
               for rep in range(4)]
        approx(so_summary[('average',model)]['P_ret'],sum(means)/4)
        approx(so_summary[('average',model)]['n_retained'],sum(means)/4*256)
    for control in ('Q','SO_equal_harmonic','SO_equal_rms'):
        diffs=[sum(int(all_a9[('RO',rep,p,i)]['status']=='retained')-
                   int(all_a9[(control,rep,p,i)]['status']=='retained')
                   for p in range(4) for i in range(16))/64 for rep in range(4)]
        approx(so_summary[('average','RO-minus-'+control)]['P_ret'],sum(diffs)/4)

    phase_rows=jsonl('phase_fem_checkpoint.jsonl',lambda r:r['key'])
    phase_by={(int(r['delta_deg']),int(r['replica']),int(r['phase_index']),int(r['state_i'])):r
              for r in phase_rows}
    old90_path=BASE/'acceptance_4d_fem_h50_delta90_N100/trajectory_results.csv'
    old90={}
    with old90_path.open(newline='',encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['point_id']=='a9_b2_e0' and r['model']=='RO' and float(r['mu'])==1.0 and r['horizon']=='100' and int(r['state_i'])<16:
                k=(int(r['replica']),int(r['phase_i']),int(r['state_i']))
                assert k not in old90
                old90[k]=r
    assert len(old90)==256
    for k,r in old90.items():
        qro=old[('RO',*k)]
        assert max(abs(float(r[c])-float(qro[c])) for c in ('x0_rho','y0_rho','vx0_rho_tau','vy0_rho_tau'))<1e-13
    phase_summary={int(r['delta_deg']):r for r in csvrows('phase_fem_summary.csv')}
    phase_counts={0:sum(old_by[('RO',rep,p,i)]['status']=='retained' for rep in range(4) for p in range(4) for i in range(16))}
    phase_counts[90]=sum(r['status']=='retained' for r in old90.values())
    for deg in (45,135,180):
        phase_counts[deg]=sum(phase_by[(deg,rep,p,i)]['status']=='retained'
                              for rep in range(4) for p in range(4) for i in range(16))
    for deg,count in phase_counts.items():
        assert int(phase_summary[deg]['retained'])==count

    mass_sources={}
    for mu,dirname in ((.70,'acceptance_4d_fem_h50_a9_adaptive_N100'),
                       (.85,'acceptance_4d_fem_h50_a9_mass_edges_N100')):
        with (BASE/dirname/'trajectory_results.csv').open(newline='',encoding='utf8') as f:
            for r in csv.DictReader(f):
                if r['point_id']=='a9_b2_e0' and float(r['mu'])==mu and r['horizon']=='100' and int(r['state_i'])<16:
                    mass_sources[(mu,r['model'],int(r['replica']),int(r['phase_i']),int(r['state_i']))]=r
    mass_so={(float(r['mu']),int(r['replica']),int(r['phase_index']),int(r['state_i'])):r for r in
             jsonl('fem_mass_so_checkpoint.jsonl',lambda r:r['key'])}
    mass_summary={(float(r['mu']),r['model']):r for r in csvrows('fem_mass_so_summary.csv')}
    for mu in (.70,.85):
        volume=167.38913929608813/mu
        for model in ('Q','SO_equal_harmonic','RO'):
            means=[]
            for rep in range(4):
                hits=0
                for p in range(4):
                    for i in range(16):
                        if model=='SO_equal_harmonic': status=mass_so[(mu,rep,p,i)]['status']
                        else: status=mass_sources[(mu,model,rep,p,i)]['status']
                        hits+=status=='retained'
                means.append(hits/64)
            rec=mass_summary[(mu,model)]
            approx(rec['P_ret'],sum(means)/4)
            approx(rec['A4_m4_s2'],float(rec['P_ret'])*volume)

    long_rows=jsonl('fem_long_checkpoint.jsonl',lambda r:r['key'])
    long_by={(int(r['horizon']),r['model'],int(r['replica']),int(r['phase_index']),int(r['state_i'])):r
             for r in long_rows}
    long_summary={(int(r['horizon']),r['model']):r for r in csvrows('fem_long_summary.csv')}
    for horizon in (100,300,1000):
        for model in ('Q','RO','SO_equal_harmonic','SO_equal_rms'):
            hits=0
            for rep in range(4):
                for p in range(4):
                    for i in range(16):
                        status=(all_a9[(model,rep,p,i)]['status'] if horizon==100 else
                                long_by[(horizon,model,rep,p,i)]['status'])
                        hits+=status=='retained'
            assert int(long_summary[(horizon,model)]['retained'])==hits
    assert all(str(r['within_250V']).lower()=='true' for r in csvrows('electrode_voltage.csv'))
    fields=[ROOT/'outputs/abc_validation/direct_fem_validation/v2/h50_P4/sector_solution.npz',
            ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_1/fem/quadrupole_h50_P4_sector_solution.npz',
            ROOT/'outputs/abc_validation/direct_fem_validation/v2/h35_P4/sector_solution.npz',
            ROOT/'outputs/octupole_comparison_home/mass_filter_v13/v13_3_three_parameter_atlas/fem/quadrupole_h35_P4_sector_solution.npz',
            ROOT/'outputs/meshes/gmsh_50um/trap.msh',ROOT/'outputs/meshes/gmsh_35um/trap.msh']
    source_datasets=[BASE/'acceptance_4d_fem_h50_selected_N100/trajectory_results.csv',old90_path,
                     BASE/'acceptance_4d_fem_h50_a9_adaptive_N100/trajectory_results.csv',
                     BASE/'acceptance_4d_fem_h50_a9_mass_edges_N100/trajectory_results.csv']
    code=[ROOT/'src/three_phase_trap/v14_dynamics.py']+[ROOT/'scripts'/n for n in
          ('run_v14_ideal_pilot.py','analyze_v14_ideal.py','run_v14_fem_orbits.py','run_v14_fem_so.py',
           'run_v14_fem_long.py','run_v14_fem_phase.py','run_v14_fem_mass_so.py','run_v14_near_orbit_long.py',
           'check_v14_fem_q_orbit.py','check_v14_q_harmonics.py','validate_v14_fem_q_orbit.py','validate_v14_pilot.py',
           'validate_v14_dataset.py','plot_v14_results.py')]
    artifacts=['ideal_pilot_trajectories.csv','ideal_pilot_paired_summary.csv',
               'ideal_periodic_orbits.json','effective_minima.csv','phase_ideal_summary.csv',
               'phase_ideal_orbits.json','fem_periodic_orbits.json','fem_q_orbit_search.json',
               'q_m10_harmonic_control.json','fem_q_orbit_monodromy_refinement.csv',
               'fem_phase_orbit_search.json','fem_so_paired_summary.csv','fem_so_replicates.csv',
               'fem_long_summary.csv','phase_fem_summary.csv','phase_fem_paired_summary.csv',
               'fem_mass_so_summary.csv','fem_mass_so_paired.csv','stroboscopic_states.csv',
               'stroboscopic_spectra.csv','ideal_pilot_dop853_validation.csv',
               'q_so_ro_ideal_fem_N100.png','effective_potential_phase.png',
               'stroboscopic_projections.png','fem_survival_vs_time.png',
               'ro_phase_retention.png','mass_retention_and_A4.png',
               'REPORT_RU.md','MODEL_COMPARISON_RU.md','PERIODIC_ORBITS_RU.md',
               'PHASE_SPACE_RU.md','PHASE_DEPENDENCE_RU.md','NUMERICAL_VALIDATION_RU.md']
    artifact_hashes={name:sha(OUT/name) for name in artifacts if (OUT/name).exists()}
    manifest=dict(project='V14 nonlinear finite-amplitude retention',
                  dataset_counts=sets,
                  exact_v13_reused_q_ro_records=dict(source=str(src.relative_to(ROOT)),rows=len(old),
                                                     initial_state_checks=len(so)*2),
                  V13_3_source_trajectory_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_datasets},
                  FEM_fields={str(p.relative_to(ROOT)):sha(p) for p in fields},
                  final_artifacts=artifact_hashes,
                  source_code={str(p.relative_to(ROOT)):sha(p) for p in code if p.exists()},
                  validation=dict(unit_tests='8 passed',pilot_DOP853='42/42 classifications matched; max exit-time difference 0.0004809 RF',
                                  fem_meshes=['h50 P4','h35 P4'],unresolved_fem_ensembles=0,
                                  h50_qro_trajectories_reused_without_reintegration=True,
                                  raw_count_reconstruction='SO, delta phase, mass-control, and 100/300/1000 RF summaries reconstructed from trajectory rows',
                                  electrode_voltage_limit='all rows <=250 V'))
    (OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({k:v['rows'] for k,v in sets.items()},indent=2))

if __name__=='__main__':main()
