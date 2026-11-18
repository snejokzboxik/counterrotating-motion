"""Build a reproducible compact ZIP and SHA-256 manifest for V17C.2."""
import hashlib,json,zipfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
RESULT_FILES='''
averaged_torque_map.csv averaging_micro_check.json averaged_orbit_equilibria_summary.json
single_particle_maps.csv single_particle_maps.json single_particle_maps.png
negative_damping_scan.csv negative_damping_long.json outer_phase_search.csv
synchronous_resonance_scan.csv search_summary.json
G2_positive_50000RF.json G2_positive_50000RF.npz
G1_inner_positive_8000RF.npz G1_outer_negative_8000RF.npz
G2_radius_plus_1pct_8000RF.npz G2_alpha_plus_0p5pct_8000RF.npz
positive_candidate_robustness.json positive_perturbation_tracks.json
negative_long_0_6_ABC_moderate_lambda0.0001.npz negative_long_1_6_ABC_moderate_lambda0.0005.npz
negative_long_2_6_ABC_moderate_lambda0.002.npz negative_long_3_6_ABC_thin_lambda0.0001.npz
negative_long_4_6_ABC_thin_lambda0.0005.npz negative_long_5_6_ABC_thin_lambda0.002.npz
long_candidate_validation.json single_long_validation.json
long_energy_comparison.json full_rf_balance_summary.json dop853_validation.json
dop853_6_ABC_moderate_neg_800RF.npz dop853_6_ABC_thin_pos_1200RF.npz
rk4_step_convergence.json fem_coefficient_convergence.png averaged_torque_sign_map.png
rejected_counterrotation_candidates.png G2_positive_robustness.png
'''.split()
REPORT_FILES='''REPORT_RU.md GEOMETRY_OPTIMIZATION_RU.md HARMONIC_ANALYSIS_RU.md
AVERAGED_TORQUE_THEORY_RU.md SINGLE_PARTICLE_DIRECTION_MAPS_RU.md
TWO_PARTICLE_COUNTER_ROTATION_RU.md COULOMB_COUPLING_RU.md FEM_VALIDATION_RU.md
ELECTRICAL_BREAKDOWN_RU.md PHYSICAL_REALIZABILITY_RU.md NUMERICAL_VALIDATION_RU.md
REPRODUCING_RU.md'''.split()

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    files=[HERE/x for x in REPORT_FILES]
    files += sorted(HERE.glob('*.py'))
    files += sorted(p for p in (HERE/'input').iterdir() if p.is_file() and p.name!='V17C_geometry_scan.zip')
    files += [RESULTS/x for x in RESULT_FILES]
    missing=[str(p.relative_to(HERE)) for p in files if not p.exists()]
    if missing:raise FileNotFoundError('Missing release files: '+', '.join(missing))
    # Do not include transient caches, checkpoints, raw duplicate JSON scans or nested archives.
    archive_source=HERE.parents[2]/'V17C_geometry_scan.zip'
    manifest={
      'study':'V17C.2',
      'status':'interim negative: no robust, externally supported two-particle counter-rotation confirmed; one fragile long-lived positive single-particle orbit found',
      'branch':'research/v17c',
      'release_zip':'V17C2_results.zip',
      'model_scope':'2D transverse electrostatics for infinitely long cylindrical electrodes; full RF trajectories for one point particle; no Coulomb pair or 3D finite-end FEM',
      'source_geometry_archive':{'path':'V17C_geometry_scan.zip','sha256':sha(archive_source),'included_inside_release_zip':False,'extracted_sources_included':True},
      'dynamics_summary':{'negative_damping_scan_runs':10240,'outer_phase_scan_runs':5488,'synchronous_resonance_runs':2592,'G2_nominal_positive_horizon_RF':50000,'G2_nominal_positive_turns':9086.5117,'two_particle_runs':0},
      'hash_scope':'SHA-256 for every file included in the ZIP except this manifest itself and the ZIP archive (the latter has a companion .sha256 file).',
      'files':[{'path':p.relative_to(HERE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    }
    manifest_path=HERE/'MANIFEST.json';manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    zip_path=HERE/'V17C2_results.zip'
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,p.relative_to(HERE).as_posix())
        z.write(manifest_path,'MANIFEST.json')
    digest=sha(zip_path)
    (HERE/'V17C2_results.zip.sha256').write_text(f'{digest}  V17C2_results.zip\n',encoding='ascii')
    print(json.dumps({'files':len(files)+1,'zip_bytes':zip_path.stat().st_size,'zip_sha256':digest},indent=2))

if __name__=='__main__':main()
