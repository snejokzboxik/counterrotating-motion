from pathlib import Path
import json, hashlib, math, argparse
import numpy as np
import attractor_search as a

OUT=a.OUT

def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate',choices=['candidate52','local15','refine12_sobol17'],default='refine12_sobol17')
    ap.add_argument('--periods',type=int,default=240000);ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--chunk-periods',type=int,default=5000);ap.add_argument('--transient',type=int,default=20000)
    ap.add_argument('--start',choices=['v12b-240k','common'],default='v12b-240k')
    ap.add_argument('--tag',default='')
    args=ap.parse_args();cfg=a.read_json(OUT/"candidate_parameters.json") if (OUT/"candidate_parameters.json").exists() else a.build_config()
    start=cfg['refine12_240k_seed'] if args.start=='v12b-240k' else cfg['matched_common_start']
    case=next(x for x in cfg['candidates'] if x['legacy_candidate']==args.candidate)
    field=a.load_field();p,qratio=a.arrays(case,field);total=args.periods;steps=args.steps;chunk=args.chunk_periods;transient=args.transient
    tag=args.tag or f'{args.candidate}_{total}p_corrected_drag';directory=OUT/'checkpoints'/tag;directory.mkdir(parents=True,exist_ok=True)
    meta_path=directory/'checkpoint.json';state=np.asarray(start['initial'],float).copy();phase0=np.asarray(start['phase'],float)
    done=0;work=np.zeros(21);files=[];rows=[];status=0
    start_sha=hashlib.sha256(np.asarray(start['initial'],dtype=np.float64).tobytes()).hexdigest()
    if meta_path.exists():
        m=a.read_json(meta_path)
        if m.get('start_sha')!=start_sha or m.get('total_periods')!=total or m.get('steps')!=steps:raise RuntimeError('checkpoint mismatch')
        done=int(m['completed_periods']);state=np.asarray(m['end_state'],float);work=np.asarray(m['work_total'],float)
        files=list(m['chunk_files']);rows=list(m['chunk_rows']);status=int(m.get('status',0))
    while done<total and not status:
        n=min(chunk,total-done);phase=(phase0+np.array([1.0,float(p[2])])*2*math.pi*done)%(2*math.pi)
        raw=a.PAIR._integrate_trace(state,phase,p,field['family'],field['k'],field['coeff'],qratio,
          case['a1_over_R'],case['a2_over_R'],case['re1_per_speed'],case['re2_per_speed'],case['gravity_z'],
          n,steps,0,steps*100)
        end,states,idx,wh,turns,windows,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,winrm,windm,wrk,n_done,status=raw
        if n_done<=0:raise RuntimeError('integrator made no progress')
        path=directory/f'chunk_{len(files):04d}.npz';cwh=wh+work[None,:]
        np.savez_compressed(path,states=states,step_index=idx+done*steps,work_history=cwh,
          chunk_start_period=done,chunk_periods=n_done,phase=phase,status=status)
        files.append(str(path.relative_to(OUT)).replace('\\','/'))
        rows.append({'start':done,'periods':int(n_done),'turns':np.asarray(turns).tolist(),
          'window_turns':np.asarray(windows).tolist(),'dmin_R':float(dmin),'status':int(status),
          'rmin_R':np.asarray(rmin).tolist(),'rmax_R':np.asarray(rmax).tolist(),'zmax_R':np.asarray(zmax).tolist()})
        done+=int(n_done);state=np.asarray(end,float);work+=np.asarray(wrk,float)
        save_json(meta_path,{'candidate':case['id'],'tag':tag,'total_periods':total,'steps':steps,'completed_periods':done,
          'end_state':state.tolist(),'work_total':work.tolist(),'chunk_files':files,'chunk_rows':rows,'status':int(status),
          'start_sha':start_sha,'phase_start':phase0.tolist(),'phase_continuity':'phase(t0)=phase0+[1,nu]*2pi*N; carrier not reset'})
        print(f'{tag}: checkpoint {done}/{total}, status={status}',flush=True)
    states_all=[];idx_all=[];wh_all=[]
    for ci,rel in enumerate(files):
        z=np.load(OUT/rel);st=z['states'];ix=z['step_index'];ww=z['work_history']
        if ci and ix[0]==idx_all[-1][-1]:st=st[1:];ix=ix[1:];ww=ww[1:]
        states_all.append(st);idx_all.append(ix);wh_all.append(ww)
    states_all=np.vstack(states_all);idx_all=np.concatenate(idx_all);wh_all=np.vstack(wh_all)
    np.savez_compressed(OUT/f'{tag}_trajectory.npz',states=states_all,step_index=idx_all,work_history=wh_all,
      initial=start['initial'],phase=start['phase'],p=p,qratio=qratio,periods=done,steps=steps,transient=transient)
    period_index=idx_all/steps;mask=period_index>=transient;late=states_all[mask];lt=period_index[mask]
    turns_late=[];freq=[];mean_r=[]
    for j in (0,6):
        th=np.unwrap(np.arctan2(late[:,j+1],late[:,j]));tr=float((th[-1]-th[0])/(2*math.pi))
        turns_late.append(tr);freq.append(tr*case['f1_Hz']/(lt[-1]-lt[0]));mean_r.append(float(np.mean(np.hypot(late[:,j],late[:,j+1]))))
    late_rows=[r for r in rows if r['start']>=transient]
    chunk_signs=[]
    for r in late_rows:
        for v in r['window_turns']:
            chunk_signs.append([int(np.sign(v[0])) if abs(v[0])>.05 else 0,int(np.sign(v[1])) if abs(v[1])>.05 else 0])
    minpair=min((r['dmin_R'] for r in rows),default=float('nan'))
    a.PAIR._energy_gz=float(case['gravity_z']);fam=field['family'];modes=field['k'];coeff=field['coeff']
    e0=a.PAIR._energy(np.asarray(start['initial'],float),p,phase0,fam,modes,coeff,qratio,0.0)
    e1=a.PAIR._energy(state,p,phase0,fam,modes,coeff,qratio,2*math.pi*done)
    L0=np.cross(start['initial'][:3],start['initial'][3:6])[2]+np.cross(start['initial'][6:9],start['initial'][9:12])[2]/p[4]
    L1=np.cross(state[:3],state[3:6])[2]+np.cross(state[6:9],state[9:12])[2]/p[4]
    torque=work[10:20].reshape(2,5);mech=work[:10];wdrag=float(mech[3]+mech[8]);rfexp=float(work[20])
    ledger={'delta_total_energy':float(e1[0]-e0[0]),'explicit_RF_work':rfexp,'drag_work':wdrag,
      'energy_residual':float(e1[0]-e0[0]-rfexp-wdrag),
      'relative_energy_residual':float(abs(e1[0]-e0[0]-rfexp-wdrag)/max(1,abs(rfexp),abs(wdrag))),
      'delta_total_Lz':float(L1-L0),'integrated_total_torque':float(torque.sum()),
      'angular_momentum_residual':float(L1-L0-torque.sum()),'Coulomb_torque_particles':torque[:,2].tolist(),
      'internal_Coulomb_torque_sum':float(torque[:,2].sum()),'RF_torque_particles':torque[:,0].tolist(),
      'drag_torque_particles':torque[:,3].tolist(),'Coulomb_work_particles':mech[[2,7]].tolist(),
      'DC_work_particles':mech[[1,6]].tolist(),'gravity_work_particles':mech[[4,9]].tolist(),
      'Coulomb_potential_start_end':[float(e0[4]),float(e1[4])]}
    result={'candidate':case['id'],'source_start':start,'requested_periods':total,'completed_periods':done,'steps_per_period':steps,
      'transient_periods':transient,'status':int(status),'retained':bool(status==0 and done==total),
      'late_turns':turns_late,'late_frequency_Hz':freq,'late_mean_radius_R':mean_r,'late_consistent_sign_each_chunk':chunk_signs,
      'all_late_chunks_counterrotating':bool(chunk_signs and all(x[0]*x[1]<0 for x in chunk_signs)),
      'minimum_pair_distance_R':float(minpair),'number_of_checkpoints':len(files),'energy_angular_momentum':ledger,
      'scope_note':'The inherited 240k archive stores initial conditions and summaries only. This is a corrected-drag replay that generates the missing state checkpoint; finite time does not prove an attractor.'}
    save_json(OUT/f'{tag}_summary.json',result)
    print(json.dumps({k:result[k] for k in ('completed_periods','retained','late_turns','late_frequency_Hz','late_mean_radius_R','minimum_pair_distance_R','all_late_chunks_counterrotating')},indent=2))

if __name__=='__main__':main()
