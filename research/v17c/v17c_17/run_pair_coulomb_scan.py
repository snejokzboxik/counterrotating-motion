from __future__ import annotations
import argparse, importlib.util, json, math, hashlib, csv, sys, time
from pathlib import Path
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[3]
V12 = ROOT / "research" / "v17c" / "v17c_12b"
V15 = ROOT / "research" / "v17c" / "v17c_15"
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
(TARGET := OUT / "trajectories").mkdir(parents=True, exist_ok=True)
SIM_PATH = V12 / "results" / "rescaled_legacy_candidate_validation" / "campaign_integrator_source.py"
spec = importlib.util.spec_from_file_location("v12b_audited_integrator", SIM_PATH)
sim = importlib.util.module_from_spec(spec); sys.modules[spec.name] = sim; spec.loader.exec_module(sim)
TWOPI = 2.0 * math.pi

@njit(cache=True)
def _rf_values(t, pos, p, phase, family, modes, coeff):
    A, B, nu = p[0], p[1], p[2]
    carrier1 = A * complex(math.cos(t+phase[0]), -math.sin(t+phase[0]))
    carrier2 = B * complex(math.cos(nu*t+phase[1]), -math.sin(nu*t+phase[1]))
    u = complex(pos[0], pos[1]); ex = 0.0; ey = 0.0; phi = 0.0; ptime = 0.0
    for h in range(modes.size):
        m = modes[h]; degree = abs(m)
        base = u if m > 0 else u.conjugate()
        carrier = carrier1 if family[h] == 0 else carrier2
        omega = 1.0 if family[h] == 0 else nu
        grad = carrier * coeff[h] * degree * base**(degree-1)
        ex -= grad.real
        ey += grad.imag if m > 0 else -grad.imag
        term = carrier * coeff[h] * base**degree
        phi += term.real
        ptime += (complex(0.0, -omega) * term).real
    return ex, ey, phi, ptime

@njit(cache=True)
def _rates(t, y, p, phase, family, modes, coeff, qratio, re1, re2, gravity_z):
    # 21 entries: work[particle, RF/DC/C/drag/g], torque[particle, same], explicit RF potential power.
    out = np.zeros(21, dtype=np.float64)
    s, C = p[4], p[5]
    mratio = (1.0, 1.0/s); qfac = (1.0, qratio); efac = (1.0, s*qratio)
    dx, dy, dz = y[0]-y[6], y[1]-y[7], y[2]-y[8]
    d2 = dx*dx+dy*dy+dz*dz
    invd3 = 0.0 if d2 <= 1e-28 or C == 0.0 else 1.0/(d2*math.sqrt(d2))
    for i in range(2):
        j = 6*i; x, yy, z, vx, vy, vz = y[j:j+6]
        ex, ey, phi, ptime = _rf_values(t, y[j:j+3], p, phase, family, modes, coeff)
        acc = np.zeros((5,3), dtype=np.float64)
        acc[0,0] = efac[i]*ex; acc[0,1] = efac[i]*ey
        acc[1,0] = efac[i]*p[3]*x; acc[1,1] = efac[i]*p[3]*yy; acc[1,2] = efac[i]*(-2.0*p[3]*z)
        sg = 1.0 if i == 0 else -1.0
        invmass = 1.0 if i == 0 else s
        dd = (dx,dy,dz)
        for k in range(3): acc[2,k] = invmass*sg*C*invd3*dd[k]
        lam = p[6] if i == 0 else p[7]
        re_per_speed = re1 if i == 0 else re2
        speed = math.sqrt(vx*vx + vy*vy + vz*vz)
        drag_factor = 1.0 + 0.15*(re_per_speed*speed)**0.687
        acc[3,0] = -lam*drag_factor*vx; acc[3,1] = -lam*drag_factor*vy; acc[3,2] = -lam*drag_factor*vz
        acc[4,2] = gravity_z
        vv = (vx,vy,vz)
        for c in range(5):
            out[5*i+c] = mratio[i]*(vv[0]*acc[c,0]+vv[1]*acc[c,1]+vv[2]*acc[c,2])
            out[10+5*i+c] = mratio[i]*(x*acc[c,1]-yy*acc[c,0])
        out[20] += qfac[i]*ptime
    return out

@njit(cache=True)
def _integrate_trace(initial, phase, p, family, modes, coeff, qratio, a1, a2, re1, re2, gravity_z,
                     periods, steps, skip, save_stride):
    dt = TWOPI/steps; total_steps = periods*steps; y = initial.copy()
    nsave = total_steps//save_stride + 1
    states = np.empty((nsave,12), dtype=np.float64); step_idx = np.empty(nsave, dtype=np.int64)
    work_history = np.empty((nsave,21), dtype=np.float64)
    states[0,:] = y; step_idx[0] = 0; work_history[0,:] = 0.0; nr = 1
    k1=np.empty(12); k2=np.empty(12); k3=np.empty(12); k4=np.empty(12); tmp=np.empty(12); yn=np.empty(12)
    wrk = np.zeros(21, dtype=np.float64); rsum=np.zeros(2); dsum=0.0; ucsum=0.0; nmean=0
    rwin=np.zeros((3,2)); dwin=np.zeros(3); cwin=np.zeros(3,dtype=np.int64)
    rmin=np.array([math.hypot(y[0],y[1]),math.hypot(y[6],y[7])]); rmax=rmin.copy()
    zmax=np.array([abs(y[2]),abs(y[8])]); vmax=np.zeros(2); turns=np.zeros(2); win_turns=np.zeros((3,2))
    d0x=y[0]-y[6]; d0y=y[1]-y[7]; d0z=y[2]-y[8]
    dmin=math.sqrt(d0x*d0x+d0y*d0y+d0z*d0z); status=0; done=0
    edges0=skip; edges1=skip+(periods-skip)//3; edges2=skip+2*(periods-skip)//3
    if a1+a2>0.0 and dmin <= a1+a2: status=1
    if (math.hypot(y[0],y[1])+a1>=1.0 or math.hypot(y[6],y[7])+a2>=1.0 or abs(y[2])+a1>=2.0 or abs(y[8])+a2>=2.0): status=2
    for step in range(total_steps):
        if status != 0: break
        t=step*dt
        sim._rhs(t,y,p,phase,family,modes,coeff,qratio,k1,re1,0,a1,a2,re2,gravity_z,2,1.0)
        for q in range(12): tmp[q]=y[q]+0.5*dt*k1[q]
        sim._rhs(t+0.5*dt,tmp,p,phase,family,modes,coeff,qratio,k2,re1,0,a1,a2,re2,gravity_z,2,1.0)
        for q in range(12): tmp[q]=y[q]+0.5*dt*k2[q]
        sim._rhs(t+0.5*dt,tmp,p,phase,family,modes,coeff,qratio,k3,re1,0,a1,a2,re2,gravity_z,2,1.0)
        for q in range(12): tmp[q]=y[q]+dt*k3[q]
        sim._rhs(t+dt,tmp,p,phase,family,modes,coeff,qratio,k4,re1,0,a1,a2,re2,gravity_z,2,1.0)
        w1=_rates(t,y,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z)
        for q in range(12): tmp[q]=y[q]+0.5*dt*k1[q]
        w2=_rates(t+0.5*dt,tmp,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z)
        for q in range(12): tmp[q]=y[q]+0.5*dt*k2[q]
        w3=_rates(t+0.5*dt,tmp,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z)
        for q in range(12): tmp[q]=y[q]+dt*k3[q]
        w4=_rates(t+dt,tmp,p,phase,family,modes,coeff,qratio,re1,re2,gravity_z)
        for q in range(12): yn[q]=y[q]+dt*(k1[q]+2*k2[q]+2*k3[q]+k4[q])/6.0
        for q in range(21): wrk[q]+=dt*(w1[q]+2*w2[q]+2*w3[q]+w4[q])/6.0
        cycle=step//steps
        for i,(xx,yy) in enumerate(((0,1),(6,7))):
            ro=math.hypot(y[xx],y[yy]); rn=math.hypot(yn[xx],yn[yy])
            rmin[i]=min(rmin[i],ro,rn); rmax[i]=max(rmax[i],ro,rn); zmax[i]=max(zmax[i],abs(y[xx+2]),abs(yn[xx+2]))
            speed=math.sqrt(yn[xx+3]**2+yn[xx+4]**2+yn[xx+5]**2); vmax[i]=max(vmax[i],speed)
            if ro>0.02 and rn>0.02:
                cross=y[xx]*yn[yy]-y[yy]*yn[xx]; dot=y[xx]*yn[xx]+y[yy]*yn[yy]
                dtheta=math.atan2(cross,dot)/TWOPI
                if cycle>=skip:
                    turns[i]+=dtheta
                    win=0 if cycle<edges1 else (1 if cycle<edges2 else 2)
                    win_turns[win,i]+=dtheta
        ddx=yn[0]-yn[6]; ddy=yn[1]-yn[7]; ddz=yn[2]-yn[8]
        segd,alpha=sim._segment_min_separation(y,yn); dmin=min(dmin,segd)
        if a1+a2>0 and segd<=a1+a2:
            status=1
            for q in range(12): y[q]=y[q]+alpha*(yn[q]-y[q])
            done=step+1; break
        for i in range(2):
            xx=0 if i==0 else 6; rad=math.hypot(yn[xx],yn[xx+1]); ar=a1 if i==0 else a2
            if rad+ar>=1.0 or abs(yn[xx+2])+ar>=2.0: status=2
        if status!=0:
            y[:]=yn; done=step+1; break
        if ddx*ddx+ddy*ddy+ddz*ddz<1e-10: status=3; done=step+1; break
        if not np.all(np.isfinite(yn)): status=3; done=step+1; break
        if cycle>=skip:
            rr=np.array([math.hypot(yn[0],yn[1]),math.hypot(yn[6],yn[7])]); d=math.sqrt(ddx*ddx+ddy*ddy+ddz*ddz)
            rsum+=rr; dsum+=d; ucsum+=p[5]/d; nmean+=1
            win=0 if cycle<edges1 else (1 if cycle<edges2 else 2)
            rwin[win]+=rr; dwin[win]+=d; cwin[win]+=1
        y[:]=yn; done=step+1
        if done%save_stride==0 or done==total_steps:
            if nr<nsave: states[nr,:]=y; step_idx[nr]=done; work_history[nr,:]=wrk; nr+=1
        if done%steps==0 and (done//steps)==periods: break
    completed=done//steps
    # Be certain the terminal state is present even if an event interrupted between strides.
    if nr==0 or step_idx[nr-1]!=done:
        if nr<nsave: states[nr,:]=y; step_idx[nr]=done; work_history[nr,:]=wrk; nr+=1
    avgr=np.zeros(2); winrm=np.zeros((3,2)); windm=np.zeros(3)
    if nmean>0: avgr=rsum/nmean
    for w in range(3):
        if cwin[w]>0: winrm[w]=rwin[w]/cwin[w]; windm[w]=dwin[w]/cwin[w]
    avd=dsum/max(1,nmean); avuc=ucsum/max(1,nmean)
    return (y,states[:nr].copy(),step_idx[:nr].copy(),work_history[:nr].copy(),turns,win_turns,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,
            winrm,windm,wrk,completed,status)

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def json_read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def build_inputs():
    meta=json_read(V12/'results'/'updated_objective_validation'/'campaign_meta.json')
    rows=[json.loads(x) for x in (V12/'results'/'updated_objective_validation'/'trajectories.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    by={(r['configuration'],r['ensemble'],int(r['start_index'])):r for r in rows}
    a='phys0570'; b='legacy_refine12_sobol17'; chosen=[]
    for e in ('rest','weak','wide','edge'):
        common=[]
        ia=[k[2] for k in by if k[0]==a and k[1]==e]
        for i in ia:
            ra=by.get((a,e,i)); rb=by.get((b,e,i))
            if ra and rb and ra['strict_late_category'].startswith('STRICT_CR') and rb['strict_late_category'].startswith('STRICT_CR'):
                common.append((min(float(ra['distance_min_over_R']),float(rb['distance_min_over_R'])),i))
        if common: chosen.append((e,sorted(common,reverse=True)[0][1],len(common)))
    if not chosen: raise RuntimeError('No matched V12B strict-CR start found for phys0570 and refine12_sobol17.')
    # Use a rest/zero-velocity shared success as the common controlled initial state.
    e,i,n=next((x for x in chosen if x[0]=='rest'),chosen[0])
    startrow=by[(a,e,i)]; refrow=by[(b,e,i)]
    if startrow['initial']!=refrow['initial'] or startrow['phase']!=refrow['phase']:
        raise RuntimeError('Matched frozen-start consistency failure.')
    scaled_meta=json_read(V12/'results'/'rescaled_legacy_candidate_validation'/'campaign_meta.json')
    scaled=next(c for c in scaled_meta['candidates'] if c['legacy_candidate']=='refine12_sobol17')
    physical=next(c for c in meta['candidates'] if c['id']=='phys0570')
    cases=[]
    for cid,cfg in (('scaled_refine12_sobol17_f50hz',scaled),('phys0570',physical)):
        p0=np.asarray(cfg['p'],dtype=np.float64).copy(); ph=cfg['physical']
        cc1=float(ph['particle1']['Cunningham']); cc2=float(ph['particle2']['Cunningham'])
        lam_s=(float(p0[6]),float(p0[7])); p0[6]/=cc1; p0[7]/=cc2
        cases.append({'id':cid,'p_legacy':np.asarray(cfg['p'],dtype=float).tolist(),'p_corrected':p0.tolist(),
          'qratio':float(cfg['qratio']),'physical':ph,'lambda_stokes':list(lam_s),
          'lambda_corrected_divide_Cc':[float(p0[6]),float(p0[7])],
          'Re_per_dimensionless_speed':[float(ph['particle1']['Re_per_dimensionless_speed']),float(ph['particle2']['Re_per_dimensionless_speed'])],
          'drag_convention':'Schiller-Naumann empirical finite-Re self-drag factor 1+0.15 Re^0.687 applied to Stokes gamma corrected individually as 6*pi*eta*a/Cc; no cross-hydrodynamic term',
          'reference_C':float(cfg['p'][5])})
    field=json_read(V12/'model'/'field_model.json')
    inp={'study':'V17C.17','base_branch_sha':'9716e9a701b4ebd8fc3ae4b72a8aaeda9437289b',
      'field_source':'V17C.12B full MFS dual-RF field_model.json; all 8 complex spatial harmonics retained',
      'field_family':field['family'],'field_k':field['k'],'field_coeff':field['coeff'],
      'initial_source':'Matched successful V17C.12B updated_objective_validation frozen start; selected deterministically among common strict CR starts.',
      'selected_ensemble':e,'selected_start_index':i,'matched_common_success_count_by_ensemble':{x[0]:x[2] for x in chosen},
      'initial':startrow['initial'],'phase':startrow['phase'],'source_rows':{'phys0570':startrow,'legacy_refine12_sobol17':refrow},
      'prior_model_labels':{'phys0570':startrow['strict_late_category'],'legacy_refine12_sobol17':refrow['strict_late_category']},
      'cases':cases,'protocol':'constant fields; no loading or phase reset; 3D translational dynamics with ideal z-independent RF and harmonic DC plus gravity.'}
    save_json(OUT/'input_configuration.json',inp)
    return inp,field

def load_inputs():
    fpath=OUT/'input_configuration.json'
    if not fpath.exists(): return build_inputs()
    inp=json_read(fpath)
    for case in inp['cases']:
        physical=case['physical']
        case['Re_per_dimensionless_speed']=[float(physical['particle1']['Re_per_dimensionless_speed']),float(physical['particle2']['Re_per_dimensionless_speed'])]
        case['drag_convention']='Schiller-Naumann empirical factor 1+0.15 Re^0.687 on individually Cunningham-corrected Stokes drag; no cross-hydrodynamic term'
    inp['primary_drag_model']='gamma_i=gamma_Stokes,i/Cc_i; dimensionless self drag -lambda_i*(1+0.15*Re_i^0.687)*v_i with Re_i=Re_per_dimensionless_speed_i*|v_i|; hydro_mode=0. This is the existing V12B Schiller-Naumann closure with Cunningham corrected per V15; no Oseen theory or cross-hydrodynamic interaction is added.'
    inp['stokes_only_control']='The earlier linear-Stokes pilot is preserved separately under results/stokes_only_preliminary; it is not used for the primary scan because it did not reproduce the drag model of the published successful starts.'
    save_json(fpath,inp)
    field={'family':inp['field_family'],'k':inp['field_k'],'coeff':inp['field_coeff']}
    return inp,field

def arrays(inp,field,case,scale=1.0):
    p=np.asarray(case['p_corrected'],dtype=np.float64).copy(); p[5]*=scale
    fam=np.asarray(field['family'],dtype=np.int64); modes=np.asarray(field['k'],dtype=np.int64)
    coeff0=np.asarray(field['coeff'],dtype=np.float64); coeff=coeff0[:,0]+1j*coeff0[:,1]
    ph=np.asarray(inp['phase'],dtype=np.float64); initial=np.asarray(inp['initial'],dtype=np.float64)
    physical=case['physical']; a1=float(physical['a1_over_R']); a2=float(physical['a2_over_R'])
    gz=float(physical['gravity_z_dimensionless'])
    re1,re2=map(float,case['Re_per_dimensionless_speed'])
    return initial,ph,p,fam,modes,coeff,float(case['qratio']),a1,a2,re1,re2,gz

def simulate(inp,field,case,scale,periods=15000,steps=64,skip=200,stride=None):
    if stride is None: stride=steps
    args=arrays(inp,field,case,scale)
    return _integrate_trace(args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11],periods,steps,skip,stride)

def old_replay(inp,field,case,oldrow,periods=15000,steps=64,skip=200):
    # Replays the archived V17C.12B self-drag + RPY + Schiller-Naumann/Cc-mean model exactly.
    source=oldrow
    cfg_meta=json_read(V12/'results'/'updated_objective_validation'/'campaign_meta.json')
    c=next(x for x in cfg_meta['candidates'] if x['id']==source['configuration'])
    ph0=np.asarray(inp['phase'],dtype=float); initial=np.asarray(inp['initial'],dtype=float)
    field0=json_read(V12/'model'/'field_model.json')
    fam=np.asarray(field0['family'],dtype=np.int64); modes=np.asarray(field0['k'],dtype=np.int64)
    cc=np.asarray(field0['coeff'],float); coeff=cc[:,0]+1j*cc[:,1]
    p=np.asarray(c['p'],dtype=np.float64); qratio=float(c['qratio']); physical=c['physical']
    raw=sim.integrate_one(initial,ph0,p,fam,modes,coeff,periods,steps,skip,qratio=qratio,
       oseen_Re_per_speed=float(physical['particle1']['Re_per_dimensionless_speed']),hydro_mode=2,
       a1_over_R=float(physical['a1_over_R']),a2_over_R=float(physical['a2_over_R']),
       oseen_Re2_per_speed=float(physical['particle2']['Re_per_dimensionless_speed']),
       gravity_z=float(physical['gravity_z_dimensionless']),drag_mode=2,
       slip_correction=float(physical['Cunningham_mean']))
    end,turns,windows,rmin,rmax,zmax,dtheta,dmin,completed,retained,escaped,singular,category,freq,vmax=raw
    return {'end_state_max_abs_diff':float(np.max(np.abs(end-np.asarray(source['final_state'])))),
      'turns_max_abs_diff':float(np.max(np.abs(turns-np.asarray(source['turns'])))),
      'window_turns_max_abs_diff':float(np.max(np.abs(windows-np.asarray(source['window_turns'])))),
      'min_distance_abs_diff':float(abs(dmin-float(source['distance_min_over_R']))),
      'retained_match':bool(retained==source['retained']),'strict_label_reference':source['strict_late_category'],
      'replay_raw_category':str(category),'completed_periods':int(completed),'frequency_array_legacy_bug':freq.tolist()}

def _field_np(pos,t,p,phase,family,modes,coeff):
    A,B,nu=p[:3]; u=complex(float(pos[0]),float(pos[1])); c1=A*np.exp(-1j*(t+phase[0])); c2=B*np.exp(-1j*(nu*t+phase[1]))
    ex=0.0;ey=0.0;phi=0.0;dphi=0.0
    for fam,m,coef in zip(family,modes,coeff):
        base=u if m>0 else u.conjugate(); n=abs(int(m)); c=c1 if fam==0 else c2; w=1.0 if fam==0 else nu
        g=c*coef*n*base**(n-1); ex-=g.real; ey+=g.imag if m>0 else -g.imag
        term=c*coef*base**n; phi+=term.real; dphi+=(-1j*w*term).real
    return np.array([ex,ey,0.]),float(phi),float(dphi)

def _rf_phi(pos,t,p,phase,family,modes,coeff): return _field_np(pos,t,p,phase,family,modes,coeff)[1]

def force_audit(inp,field,case):
    initial,phase,p,fam,modes,coeff,qratio,a1,a2,re1,re2,gz=arrays(inp,field,case,1.0)
    rng=np.random.default_rng(170017); maxrel=0.; maxgrad=0.; maxpair=0.
    for _ in range(32):
        y=rng.normal(0,0.25,12); y[:3]*=.7; y[6:9]*=.7; t=float(rng.uniform(0,20))
        out=np.empty(12); sim._rhs(t,y,p,phase,fam,modes,coeff,qratio,out,re1,0,a1,a2,re2,gz,2,1.)
        for i in range(2):
            j=6*i; ex,phi,_=_field_np(y[j:j+3],t,p,phase,fam,modes,coeff)
            ef=1. if i==0 else p[4]*qratio
            s=p[4]; dx=y[0]-y[6];dy=y[1]-y[7];dz=y[2]-y[8]; d=np.sqrt(dx*dx+dy*dy+dz*dz)
            ac=np.array([dx,dy,dz])*(p[5]/d**3)*(1. if i==0 else -s)
            lam=p[6] if i==0 else p[7]; dc=np.array([p[3]*y[j],p[3]*y[j+1],-2*p[3]*y[j+2]])*ef
            re_per_speed=re1 if i==0 else re2; speed=np.linalg.norm(y[j+3:j+6]); drag=lam*(1+0.15*(re_per_speed*speed)**0.687)
            acc=ef*ex+dc+ac-drag*y[j+3:j+6]+np.array([0.,0.,gz])
            maxrel=max(maxrel,float(np.max(np.abs(acc-out[j+3:j+6]))))
        d=np.linalg.norm(y[:3]-y[6:9]);
        if d>1e-4:
            F1=p[5]*(y[:3]-y[6:9])/d**3; F2=-F1
            maxpair=max(maxpair,float(np.max(np.abs(F1+F2))))
        pos=y[:3].copy(); h=1e-6
        ex,_,_=_field_np(pos,t,p,phase,fam,modes,coeff)
        grad=[]
        for q in range(2):
            pp=pos.copy();pm=pos.copy();pp[q]+=h;pm[q]-=h
            grad.append((_rf_phi(pp,t,p,phase,fam,modes,coeff)-_rf_phi(pm,t,p,phase,fam,modes,coeff))/(2*h))
        maxgrad=max(maxgrad,float(np.max(np.abs(ex[:2]+np.asarray(grad)))))
    return {'source_rhs_vs_independent_force_decomposition_max_abs_acceleration_error':maxrel,
      'finite_difference_MFS_potential_gradient_max_abs_error':maxgrad,
      'pair_force_action_reaction_max_abs_error':maxpair,
      'DC_azimuthal_torque_analytic':'identically zero for axisymmetric harmonic DC potential',
      'harmonics_count':len(modes),'harmonic_k':modes.tolist(),'family':fam.tolist()}

def _energy(y,p,phase,family,modes,coeff,qratio,t):
    s=p[4]; qfac=(1.,qratio); mfac=(1.,1./s); kinetic=0.; potential=0.; urf=0.; udc=0.; ug=0.
    for i in range(2):
        j=6*i; v=y[j+3:j+6]; kinetic+=.5*mfac[i]*float(v@v)
        _,phi,_=_field_np(y[j:j+3],t,p,phase,family,modes,coeff); urf+=qfac[i]*phi
        x,yy,z=y[j:j+3]; udc+=qfac[i]*p[3]*(z*z-.5*(x*x+yy*yy))
        ug-=mfac[i]*float(_energy_gz)*z
    d=float(np.linalg.norm(y[:3]-y[6:9])); uc=p[5]/d if d>0 else math.inf
    return kinetic+urf+udc+uc+ug,kinetic,urf,udc,uc,ug

# gravity is passed to energy explicitly through this context variable only for the terminal summary.
_energy_gz=0.0

def summarize_case(inp,field,case,scale,result,periods,steps,skip):
    end,states,idx,work_history,turns,windows,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,winrm,windm,wrk,completed,status=result
    p=np.asarray(case['p_corrected'],float).copy();p[5]*=scale
    f1=float(case['physical']['frequency_f1_Hz']); qratio=float(case['qratio']); s=float(p[4]); nlate=max(1,periods-skip)
    f_orbit=np.asarray(turns)*f1/nlate
    f_windows=np.asarray(windows)*f1/np.maximum(1,(periods-skip)//3)
    gz=float(case['physical']['gravity_z_dimensionless'])
    global _energy_gz; _energy_gz=gz
    fam=np.asarray(field['family'],np.int64); modes=np.asarray(field['k'],np.int64); cc=np.asarray(field['coeff'],float); coeff=cc[:,0]+1j*cc[:,1]
    e0=_energy(np.asarray(inp['initial'],float),p,np.asarray(inp['phase'],float),fam,modes,coeff,qratio,0.)
    e1=_energy(np.asarray(end,float),p,np.asarray(inp['phase'],float),fam,modes,coeff,qratio,TWOPI*completed)
    # Work indexes 0..9 are by particle, component [RF,DC,C,drag,g]; torque 10..19.
    work=np.asarray(wrk[:10]); torq=np.asarray(wrk[10:20]).reshape(2,5); pexplicit=float(wrk[20])
    Wdrag=float(work[3]+work[8]); Wmech=float(np.sum(work)); Wc=float(work[2]+work[7]); Mc=float(torq[0,2]+torq[1,2])
    dE=float(e1[0]-e0[0]); dK=float(e1[1]-e0[1]); dL=float((np.cross(end[:3],end[3:6])[2]+np.cross(end[6:9],end[9:12])[2]/s)-
        (np.cross(inp['initial'][:3],inp['initial'][3:6])[2]+np.cross(inp['initial'][6:9],inp['initial'][9:12])[2]/s))
    bal={'delta_total_energy':dE,'explicit_RF_work_plus_drag':pexplicit+Wdrag,'energy_residual':dE-pexplicit-Wdrag,
      'energy_residual_relative_to_RF_work':abs(dE-pexplicit-Wdrag)/max(1.,abs(pexplicit),abs(Wdrag)),
      'delta_kinetic_energy':dK,'sum_mechanical_work_RF_DC_C_drag_gravity':Wmech,'kinetic_work_residual':dK-Wmech,
      'delta_total_Lz':dL,'integrated_total_torque':float(np.sum(torq)),'angular_momentum_residual':dL-float(np.sum(torq)),
      'integrated_Coulomb_work_per_particle':work[[2,7]].tolist(),'integrated_Coulomb_torque_per_particle':torq[:,2].tolist(),
      'internal_Coulomb_torque_sum':Mc,'RF_work_per_particle':work[[0,5]].tolist(),'drag_work_per_particle':work[[3,8]].tolist(),
      'DC_work_per_particle':work[[1,6]].tolist(),'gravity_work_per_particle':work[[4,9]].tolist(),
      'RF_torque_per_particle':torq[:,0].tolist(),'drag_torque_per_particle':torq[:,3].tolist(),'DC_torque_per_particle':torq[:,1].tolist(),
      'Coulomb_potential_initial_final':[e0[4],e1[4]],'Coulomb_work_plus_delta_potential':Wc+(e1[4]-e0[4])}
    signs=np.sign(f_windows)
    cr=bool(completed>=periods and np.all(signs[:,0]*signs[:,1]<0) and np.all(np.abs(windows)>0.05) and np.all(np.abs(turns)>=10))
    status_name={0:'COMPLETE',1:'PARTICLE_CONTACT',2:'WALL_OR_AXIAL_CONTACT',3:'SINGULAR_OR_NONFINITE'}.get(int(status),'INCOMPLETE')
    return {'configuration':case['id'],'coulomb_scale':float(scale),'periods_requested':periods,'periods_completed':int(completed),'steps_per_RF_period':steps,
      'drag_model':case['drag_convention'],'status':status_name,'retained':bool(status==0 and completed>=periods),'counterrotating_screen':cr,
      'turns':np.asarray(turns).tolist(),'late_window_turns':np.asarray(windows).tolist(),'frequency_Hz_corrected_from_cycles':f_orbit.tolist(),
      'late_window_frequency_Hz':f_windows.tolist(),'mean_radius_over_R':np.asarray(avgr).tolist(),'mean_radius_windows_over_R':np.asarray(winrm).tolist(),
      'min_radius_over_R':np.asarray(rmin).tolist(),'max_radius_over_R':np.asarray(rmax).tolist(),'max_abs_z_over_R':np.asarray(zmax).tolist(),
      'mean_separation_over_R':float(avd),'min_separation_over_R':float(dmin),'mean_Coulomb_energy_over_m1R2Omega2':float(avuc),
      'max_speed_over_ROmega':np.asarray(vmax).tolist(),'max_Re_estimate':(np.asarray(vmax)*np.asarray(case['Re_per_dimensionless_speed'])).tolist(),
      'completed_mechanical_work':work.tolist(),'torque_integrals_by_particle_RF_DC_C_drag_g':torq.tolist(),
      'balances':bal,'sample_count':int(states.shape[0]),'sample_stride_steps':int((periods*steps)//max(1,states.shape[0]-1))}

def write_jsonl(path,row):
    with Path(path).open('a',encoding='utf-8') as f: f.write(json.dumps(row,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n')

def archived_audit(inp,field):
    rows=inp['source_rows'];res={}
    for cid,oldid in (('phys0570','phys0570'),('scaled_refine12_sobol17_f50hz','legacy_refine12_sobol17')):
        src=rows[oldid]
        # The frequency-scaled case maps the same dimensionless point; archived source uses the literal candidate here.
        replayrow=src.copy(); replayrow['configuration']='legacy_refine12_sobol17' if 'refine12' in cid else 'phys0570'
        r=old_replay(inp,field,{},replayrow)
        res[cid]=r
    return res

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['audit','scan','validate','balance','long','robust','all'],default='all'); ap.add_argument('--periods',type=int,default=15000); ap.add_argument('--steps',type=int,default=64); ap.add_argument('--skip',type=int,default=200); ap.add_argument('--workers',type=int,default=1); args=ap.parse_args()
    t0=time.perf_counter(); inp,field=load_inputs(); cases=inp['cases']
    if args.stage in ('audit','all'):
        audit={'selected_start':{'ensemble':inp['selected_ensemble'],'index':inp['selected_start_index'],'common_CR_counts':inp['matched_common_success_count_by_ensemble'],
          'initial':inp['initial'],'phase':inp['phase']},'force_field_checks':force_audit(inp,field,cases[0]),
          'archived_exact_replays':archived_audit(inp,field),'source_SHA256':{str(p.relative_to(ROOT)):sha(p) for p in [SIM_PATH,V12/'results'/'updated_objective_validation'/'campaign_runner_source.py',V12/'results'/'updated_objective_validation'/'campaign_meta.json',V12/'results'/'updated_objective_validation'/'trajectories.jsonl',V12/'model'/'field_model.json',V15/'numerical'/'physical_drag.py']}}
        save_json(OUT/'model_and_history_audit.json',audit)
    if args.stage in ('scan','all'):
        scales=[0.,.1,.25,.5,.75,1.,1.5,2.,3.,5.]
        scanpath=OUT/'coulomb_scan_corrected_drag.jsonl'; done=set()
        if scanpath.exists():
            for line in scanpath.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    r=json.loads(line);done.add((r['configuration'],r['coulomb_scale']))
        for case in cases:
            for sc in scales:
                key=(case['id'],sc); stem=f"{case['id']}_C{sc:g}_rk4{args.steps}_{args.periods}p"
                traj=TARGET/(stem+'.npz')
                if key in done and traj.exists(): continue
                tic=time.perf_counter(); trace=simulate(inp,field,case,sc,args.periods,args.steps,args.skip,args.steps)
                row=summarize_case(inp,field,case,sc,trace,args.periods,args.steps,args.skip)
                np.savez_compressed(traj,states=trace[1],step_index=trace[2],work_history=trace[3],initial=np.asarray(inp['initial']),phase=np.asarray(inp['phase']),
                    p_corrected=np.asarray(case['p_corrected']),p_used=np.asarray(case['p_corrected']) if sc==1 else np.asarray(case['p_corrected']) * np.array([1,1,1,1,1,sc,1,1]),
                    qratio=case['qratio'],coulomb_scale=sc,periods=args.periods,steps=args.steps,skip=args.skip,frequency_f1_Hz=case['physical']['frequency_f1_Hz'])
                row['wall_seconds']=time.perf_counter()-tic;write_jsonl(scanpath,row); print(f"{case['id']} C={sc:g}: {row['status']} CR={row['counterrotating_screen']} f={row['frequency_Hz_corrected_from_cycles']} Hz dmin={row['min_separation_over_R']:.5g} wall={row['wall_seconds']:.1f}s",flush=True)
        rows=[json.loads(x) for x in scanpath.read_text(encoding='utf-8').splitlines() if x.strip()]
        with (OUT/'coulomb_scan_corrected_drag.csv').open('w',newline='',encoding='utf-8') as f:
            keys=['configuration','coulomb_scale','status','counterrotating_screen','periods_completed','frequency_Hz_corrected_from_cycles','mean_radius_over_R','mean_separation_over_R','min_separation_over_R','mean_Coulomb_energy_over_m1R2Omega2','turns','late_window_turns','balances']
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();
            for r in rows: w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(list,dict)) else r.get(k) for k in keys})
    if args.stage in ('validate','all'):
        # RK4 refinement against the 64-step scan and independent DOP853 on the scaled refine finalist.
        val=[]
        scanrows=[json.loads(x) for x in (OUT/'coulomb_scan_corrected_drag.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
        for case in cases:
            for st in (96,128):
                tic=time.perf_counter(); initial,phase,p,fam,modes,coeff,qratio,a1,a2,re1,re2,gz=arrays(inp,field,case,1.)
                raw=sim.integrate_one(initial,phase,p,fam,modes,coeff,args.periods,st,args.skip,qratio=qratio,oseen_Re_per_speed=re1,hydro_mode=0,a1_over_R=a1,a2_over_R=a2,oseen_Re2_per_speed=re2,gravity_z=gz,drag_mode=2,slip_correction=1.)
                row64=next(r for r in scanrows if r['configuration']==case['id'] and r['coulomb_scale']==1.)
                turns=np.asarray(raw[1]); windows=np.asarray(raw[2]); f1=float(case['physical']['frequency_f1_Hz'])
                val.append({'kind':'RK4_step_refinement','configuration':case['id'],'steps':st,'periods':args.periods,'turns':turns.tolist(),'frequency_Hz':(turns*f1/max(1,args.periods-args.skip)).tolist(),'window_turns':windows.tolist(),'min_distance_over_R':float(raw[7]),'completed_periods':int(raw[8]),'retained':bool(raw[9]),'difference_from_RK4_64':{'max_abs_turns':float(np.max(np.abs(turns-np.asarray(row64['turns'])))),'max_abs_min_distance':float(abs(float(raw[7])-row64['min_separation_over_R']))},'wall_seconds':time.perf_counter()-tic})
        nDOP=3000; t_eval=np.arange(0,nDOP*16+1,dtype=float)*TWOPI/16
        for case in cases:
            initial,phase,p,fam,modes,coeff,qratio,a1,a2,re1,re2,gz=arrays(inp,field,case,1.)
            def rhs_py(t,y):
                out=np.zeros(12); s=p[4]; C=p[5]; dx=y[0]-y[6];dy=y[1]-y[7];dz=y[2]-y[8];d=(dx*dx+dy*dy+dz*dz)**.5
                for i in range(2):
                    j=6*i; er,_,_=_field_np(y[j:j+3],t,p,phase,fam,modes,coeff); ef=1. if i==0 else s*qratio
                    lam=p[6] if i==0 else p[7]; re_per_speed=re1 if i==0 else re2; sg=1. if i==0 else -1.; inv=1. if i==0 else s
                    c=np.array([dx,dy,dz])*(0. if d==0 else sg*inv*C/d**3)
                    out[j:j+3]=y[j+3:j+6]
                    speed=np.linalg.norm(y[j+3:j+6]); drag=lam*(1+0.15*(re_per_speed*speed)**0.687)
                    out[j+3:j+6]=ef*np.array([er[0]+p[3]*y[j],er[1]+p[3]*y[j+1],-2*p[3]*y[j+2]])+c-drag*y[j+3:j+6]+np.array([0.,0.,gz])
                return out
            tic=time.perf_counter(); dop=solve_ivp(rhs_py,(0.,nDOP*TWOPI),initial,method='DOP853',rtol=1e-9,atol=1e-11,max_step=TWOPI/96,t_eval=t_eval)
            rktrace=simulate(inp,field,case,1.,nDOP,64,200,64)
            if dop.success:
                xyp=np.arctan2(dop.y[[1,7]],dop.y[[0,6]]); theta=np.unwrap(xyp,axis=1); dturn=(theta[:,-1]-theta[:,200*16])/(2*np.pi)
                dmin=float(np.min(np.linalg.norm(dop.y[:3]-dop.y[6:9],axis=0)))
                rkturn=np.asarray(rktrace[4]); rkmin=float(rktrace[10])
                np.savez_compressed(OUT/f'independent_dop853_{case["id"]}_C1_3000p.npz',t=dop.t,y=dop.y,rtol=1e-9,atol=1e-11)
                val.append({'kind':'independent_DOP853','configuration':case['id'],'periods':nDOP,'rtol':1e-9,'atol':1e-11,'max_step_RF_periods':1/96,'output_samples_per_RF_period':16,'success':bool(dop.success),'nfev':int(dop.nfev),'DOP_turns_after_200p':dturn.tolist(),'RK4_64_turns_after_200p':rkturn.tolist(),'turn_difference':(dturn-rkturn).tolist(),'DOP_min_distance_over_R_sampled':dmin,'RK4_segment_min_distance_over_R':rkmin,'wall_seconds':time.perf_counter()-tic})
        save_json(OUT/'numerical_validation.json',{'validation':val,'DOP853_note':'Independent Python RHS with separately coded full MFS gradient, Coulomb force, DC, gravity and Cunningham-corrected Schiller-Naumann self-drag; 3000 RF periods per base configuration.'})
    if args.stage=='balance':
        rows=[]
        for case in cases:
            for st in (96,128):
                tic=time.perf_counter(); tr=simulate(inp,field,case,1.,args.periods,st,args.skip,st)
                row=summarize_case(inp,field,case,1.,tr,args.periods,st,args.skip)
                row['kind']='RK4_energy_and_moment_balance';row['wall_seconds']=time.perf_counter()-tic
                rows.append(row);np.savez_compressed(TARGET/f'{case["id"]}_C1_rk4{st}_{args.periods}p_energy.npz',states=tr[1],step_index=tr[2],work_history=tr[3],initial=np.asarray(inp['initial']),phase=np.asarray(inp['phase']),periods=args.periods,steps=st)
                print(f"balance {case['id']} RK4/{st}: Eres={row['balances']['energy_residual']:.5g} Mres={row['balances']['angular_momentum_residual']:.5g}",flush=True)
        save_json(OUT/'refined_energy_balance.json',{'rows':rows,'units':'dimensionless energy m1 R^2 Omega1^2; dimensionless angular momentum m1 R^2 Omega1'})
    if args.stage=='long':
        rows=[]
        for case in cases:
            tic=time.perf_counter();tr=simulate(inp,field,case,1.,args.periods,args.steps,args.skip,args.steps)
            row=summarize_case(inp,field,case,1.,tr,args.periods,args.steps,args.skip);row['kind']='long_horizon_RK4';row['wall_seconds']=time.perf_counter()-tic
            rows.append(row);np.savez_compressed(TARGET/f'{case["id"]}_C1_rk4{args.steps}_{args.periods}p_long.npz',states=tr[1],step_index=tr[2],work_history=tr[3],initial=np.asarray(inp['initial']),phase=np.asarray(inp['phase']),periods=args.periods,steps=args.steps)
            print(f"long {case['id']}: {row['status']} turns={row['turns']}",flush=True)
        save_json(OUT/f'long_horizon_{args.periods}p.json',{'rows':rows})
    if args.stage=='robust':
        import copy
        outpath=OUT/'finite_perturbation_robustness.json'
        robust=json_read(outpath).get('rows',[]) if outpath.exists() else []
        done={(r['configuration'],r['perturbation']) for r in robust}
        base=np.asarray(inp['initial'],float);deg=np.pi/180
        for case in cases:
            variants=[]
            basephase=np.asarray(inp['phase'],float).copy()
            y=base.copy();y[[0,1,6,7]]*=1.01;variants.append(('both_radii_plus_1pct',y,basephase.copy()))
            y=base.copy();c,sn=np.cos(5*deg),np.sin(5*deg);x2,yy2=y[6],y[7];y[6],y[7]=c*x2-sn*yy2,sn*x2+c*yy2;variants.append(('particle2_azimuth_plus_5deg',y,basephase.copy()))
            y=base.copy()
            for j,sgn in ((0,1.),(6,-1.)):
                rad=np.hypot(y[j],y[j+1]);y[j+3]=sgn*.05*(-y[j+1]/rad);y[j+4]=sgn*.05*(y[j]/rad)
            variants.append(('opposite_tangential_kick_0p05',y,basephase.copy()))
            y=base.copy();y[2]+=.02;y[8]-=.02;variants.append(('opposite_axial_offset_0p02R',y,basephase.copy()))
            y=base.copy();ph=basephase.copy();ph[0]+=5*deg;ph[1]-=5*deg;variants.append(('RF_phases_plus_minus_5deg',y,ph))
            for i,name in ((0,'particle1'),(6,'particle2')):
                y=base.copy();y[i:i+2]*=1.01;variants.append((f'{name}_radius_plus_1pct',y,basephase.copy()))
            for j,name,sgn in ((0,'particle1',1.),(6,'particle2',-1.)):
                y=base.copy();rad=np.hypot(y[j],y[j+1]);y[j+3]=sgn*.02*y[j]/rad;y[j+4]=sgn*.02*y[j+1]/rad
                variants.append((f'{name}_radial_kick_0p02',y,basephase.copy()))
            for j,name,sgn in ((0,'particle1',1.),(6,'particle2',-1.)):
                y=base.copy();rad=np.hypot(y[j],y[j+1]);y[j+3]=sgn*.02*(-y[j+1]/rad);y[j+4]=sgn*.02*(y[j]/rad)
                variants.append((f'{name}_tangential_kick_0p02',y,basephase.copy()))
            ph=basephase.copy();ph[0]+=5*deg;variants.append(('RF1_phase_plus_5deg_only',base.copy(),ph))
            ph=basephase.copy();ph[1]-=5*deg;variants.append(('RF2_phase_minus_5deg_only',base.copy(),ph))
            for name,y,ph in variants:
                if (case['id'],name) in done: continue
                local=copy.deepcopy(inp);local['initial']=y.tolist();local['phase']=ph.tolist()
                tic=time.perf_counter();tr=simulate(local,field,case,1.,args.periods,args.steps,args.skip,args.steps)
                row=summarize_case(local,field,case,1.,tr,args.periods,args.steps,args.skip);row['perturbation']=name;row['wall_seconds']=time.perf_counter()-tic
                robust.append(row);done.add((case['id'],name));np.savez_compressed(TARGET/f'{case["id"]}_C1_{name}_rk4{args.steps}_{args.periods}p.npz',states=tr[1],step_index=tr[2],initial=y,phase=ph,periods=args.periods,steps=args.steps)
                save_json(outpath,{'rows':robust,'interpretation':'Small deterministic perturbation panel around one selected base start; not a basin-probability estimate.'})
                print(f"robust {case['id']} {name}: {row['status']} CR={row['counterrotating_screen']}",flush=True)
        save_json(outpath,{'rows':robust,'interpretation':'Small deterministic perturbation panel around one selected base start; not a basin-probability estimate.'})
    print(f"stage={args.stage} elapsed_s={time.perf_counter()-t0:.1f}",flush=True)

if __name__=='__main__': main()
