"""Global checkpointed phase/amplitude search for the 12-rod ADCFEB field.

2-D infinite-cylinder MFS only; one common RF frequency; phasors use exp(-iwt)
and gauge phi_1=0. Each candidate is normalized to sum_j |V_j|^2 = 12 U0^2.
"""
from __future__ import annotations
import csv, importlib.util, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import itertools
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
V17C4 = HERE.parents[2]
ARCH = V17C4 / "input/archive_contents/v17c_six_phase/scan.py"
FEM = V17C4 / "results/fem_sixphase/G1_G2_12electrode_6phase_P1_FEM.json"
N, A, B, K, P = 12, .05, 1.10, 72, 64
R_D = 1.0 - A
SEED = 20261118
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)
BASE_PHASE = np.asarray([0,3,2,5,4,1]*2) * np.pi/3
V_BASE = np.exp(1j*BASE_PHASE)
RCHI = np.linspace(.20, .75, 111)
OUTER_R = (.47,.50,.55,.60,.65)
INNER_R = (.30,.35,.40)

def load_scan():
    s=importlib.util.spec_from_file_location("mfs_global_screen", ARCH)
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def surface_basis(samples=128):
    th=np.arange(N)*2*np.pi/N; c=np.exp(1j*th)
    aa=th[:,None]+np.arange(P)*2*np.pi/P
    z=(c[:,None]+A*np.exp(1j*aa)).ravel()
    src=(c[:,None]+.65*A*np.exp(1j*aa)).ravel()
    sb=src.conj()[None,:]
    def G(t): return np.log(np.abs(B*B-t[:,None]*sb)/(B*np.abs(t[:,None]-src[None,:])))
    q=np.linalg.solve(G(z),np.repeat(np.eye(N),P,axis=0))
    local=np.arange(samples)*2*np.pi/samples
    t=(c[:,None]+A*np.exp(1j*local[None,:])).ravel()
    den=B*B-t[:,None]*sb; dif=t[:,None]-src[None,:]
    gx=np.real(-sb/den-1/dif); gy=np.real(-1j*sb/den-1j/dif)
    return -(gx@q),-(gy@q)

def fields_at(k,T,r,nt=512):
    th=np.arange(nt)*2*np.pi/nt; z=r*np.exp(1j*th)
    ex=np.zeros((nt,N),complex); ey=np.zeros_like(ex)
    for i,kk0 in enumerate(k):
        kk=int(kk0); n=abs(kk); base=z if kk>0 else z.conj()
        fx=-n*base[:,None]**(n-1)*T[i][None,:]
        ex+=fx; ey+=(-1j*fx) if kk>0 else (1j*fx)
    return ex,ey

def decode(X,mode):
    X=np.asarray(X,float)
    if X.ndim==1: X=X[:,None]
    if mode=="phase_equal_amplitude":
        amp=np.ones((X.shape[1],N)); phase=np.column_stack([np.zeros(X.shape[1]),X.T])
    else:
        amp=X[:N].T.copy(); phase=np.column_stack([np.zeros(X.shape[1]),X[N:].T])
    amp/=np.sqrt(np.mean(amp**2,axis=1,keepdims=True))
    v=amp*np.exp(1j*phase)
    v/=np.sqrt(np.mean(abs(v)**2,axis=1,keepdims=True))
    return v.T

def prepare(k,T,exs,eys):
    f={r:fields_at(k,T,r) for r in sorted(set(OUTER_R+INNER_R))}
    n=abs(k).astype(float); tw=np.sign(k)*n*n*(n-1)
    p=RCHI[:,None]**(2*n[None,:]-2)
    tm=p*tw[None,:]; am=p*abs(tw)[None,:]
    oi=np.flatnonzero((RCHI>=.47)&(RCHI<=.65))
    ii=np.flatnonzero((RCHI>=.20)&(RCHI<=.40))
    base=metrics(V_BASE[:,None],k,T,exs,eys,f,tm,am,oi,ii)
    return f,tm,am,oi,ii,base

def metrics(V,k,T,exs,eys,f,tm,am,oi,ii):
    c=T@V; c2=abs(c)**2
    torque=tm@c2; den=am@c2; chi=torque/np.maximum(den,1e-30)
    profile={}
    for r,(ex,ey) in f.items():
        e2=abs(ex@V)**2+abs(ey@V)**2; mean=e2.mean(axis=0)
        profile[r]=(mean,(e2.max(axis=0)-e2.min(axis=0))/np.maximum(mean,1e-30),
                    (e2.max(axis=0)-e2.min(axis=0))/4)
    surf=abs(exs@V)**2+abs(eys@V)**2
    outmean=np.mean([profile[r][0] for r in OUTER_R],axis=0)
    inmean=np.mean([profile[r][0] for r in INNER_R],axis=0)
    return {
      "chi":chi,"outer_chi":chi[oi].mean(axis=0),"inner_chi":chi[ii].mean(axis=0),
      "neg_fraction":(chi[oi]<-.15).mean(axis=0),
      "neg_margin":np.maximum(-chi[oi]-.15,0).mean(axis=0),
      "pos_fraction":(chi[ii]>.10).mean(axis=0),
      "outer_E2":outmean,"inner_E2":inmean,
      "outer_corr":np.mean([profile[r][1] for r in OUTER_R],axis=0),
      "inner_corr":np.mean([profile[r][1] for r in INNER_R],axis=0),
      "outer_barrier":np.mean([profile[r][2] for r in OUTER_R],axis=0),
      "inner_barrier":np.mean([profile[r][2] for r in INNER_R],axis=0),
      "Emax":np.sqrt(surf.max(axis=0)),
      "neighbor":np.max(abs(V-np.roll(V,-1,axis=0)),axis=0),
      "profile":profile}

def score(m,base):
    ri=np.minimum(m["outer_E2"]/base["outer_E2"][0],m["inner_E2"]/base["inner_E2"][0])
    br=m["outer_barrier"]/base["outer_barrier"][0]
    er=m["Emax"]/base["Emax"][0]
    return (4.5*m["neg_fraction"]+4*m["neg_margin"]+1.6*m["pos_fraction"]
      +np.maximum(-m["outer_chi"],0)+.65*np.maximum(m["inner_chi"],0)
      -1.05*m["outer_corr"]-.2*m["inner_corr"]-.55*br
      +.2*np.log(np.maximum(ri,1e-12))
      -8*np.maximum(.75-m["outer_E2"]/base["outer_E2"][0],0)**2
      -8*np.maximum(.75-m["inner_E2"]/base["inner_E2"][0],0)**2
      # A counter-rotating radial layout requires positive inner torque as well
      # as negative outer torque. Strongly reject all-negative / all-positive
      # solutions even when their outer band alone scores well.
      -12*np.maximum(.15-m["inner_chi"],0)
      -4*np.maximum(1.0-m["pos_fraction"],0)
      -8*np.maximum(.15+m["outer_chi"],0)
      -3*np.maximum(.75-m["neg_fraction"],0)
      -2*np.maximum(er-1.15,0)**2-1.5*np.maximum(er-1,0)
      -.5*np.maximum(m["neighbor"]-2.2,0)**2)

class Obj:
    def __init__(self,mode,k,T,exs,eys,f,tm,am,oi,ii,base):
        self.mode,self.k,self.T,self.exs,self.eys=mode,k,T,exs,eys
        self.f,self.tm,self.am,self.oi,self.ii,self.base=f,tm,am,oi,ii,base
    def ev(self,X):
        V=decode(X,self.mode)
        m=metrics(V,self.k,self.T,self.exs,self.eys,self.f,self.tm,self.am,self.oi,self.ii)
        return V,m,score(m,self.base)
    def __call__(self,X): return -self.ev(X)[2]

def save_npz(path,**kw):
    tmp=path.with_suffix(".tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**kw)
    tmp.replace(path)

def checkpoint(directory,mode,chunk,res,seed):
    save_npz(directory/f"checkpoint_{mode}_{chunk:02d}.npz",
      population=res.population,energies=res.population_energies,best_x=res.x,
      best_fun=np.asarray(res.fun),nfev=np.asarray(res.nfev),chunk=np.asarray(chunk),
      seed=np.asarray(seed))

def main():
    scan=load_scan()
    with threadpool_limits(limits=1):
        k,T,bres=scan.transfer(N,A,B,p=P,K=K)
        exs,eys=surface_basis()
    f,tm,am,oi,ii,base=prepare(k,T,exs,eys)
    saved=json.loads(FEM.read_text(encoding="utf-8"))["geometries"]["ADCFEBADCFEB"]["mesh_levels"][-1]["mfs_coefficients"]
    sd={int(kk):complex(float(re),float(im)) for kk,re,im in saved}
    cb=T@V_BASE
    coeff_delta=max(abs(sd[int(kk)]-cb[i]) for i,kk in enumerate(k))
    baseinfo={key:float(v[0]) for key,v in base.items()
              if isinstance(v,np.ndarray) and v.ndim==1 and len(v)==1}
    baseinfo.update({"MFS_boundary_residual":float(max(abs(bres@V_BASE))),
                     "max_coeff_delta_saved":float(coeff_delta)})
    (OUT/"baseline_validation.json").write_text(json.dumps(baseinfo,indent=2),encoding="utf-8")
    print("ADCFEB",json.dumps(baseinfo),flush=True)

    # Seed the global populations with all archived first-phase-A repeated
    # six-phase permutations. This preserves known sign-changing patterns in
    # the search instead of relying on a low-probability random hit.
    with (ARCH.parent / "permutations.csv").open(newline="", encoding="utf-8") as fh:
        patterns=[row["sequence"] for row in csv.DictReader(fh)]
    phase_index={ch:i for i,ch in enumerate("ABCDEF")}
    phase_seed=np.asarray([[phase_index[ch]*np.pi/3 for ch in name[:N]] for name in patterns])
    phase_seed[:,0]=0.0
    phase_seed=(phase_seed+np.pi)%(2*np.pi)-np.pi
    phase_seed[:,0]=0.0
    configs=[
      ("phase_equal_amplitude",11,12,8,SEED+101),
      ("phase_and_amplitude",23,13,8,SEED+202)]
    all_summaries=[]
    for mode,dim,power,popsize,seed in configs:
        od=OUT/mode; od.mkdir(parents=True,exist_ok=True)
        bounds=[(-np.pi,np.pi)]*dim if mode=="phase_equal_amplitude" else [(.70,1.30)]*12+[(-np.pi,np.pi)]*11
        obj=Obj(mode,k,T,exs,eys,f,tm,am,oi,ii,base)
        sob=qmc.Sobol(d=dim,scramble=True,seed=seed).random_base2(power)
        lo=np.array([x[0] for x in bounds]); hi=np.array([x[1] for x in bounds])
        X=lo+sob*(hi-lo)
        discrete=(phase_seed[:,1:] if mode=="phase_equal_amplitude"
                  else np.column_stack([np.ones(len(phase_seed)*N).reshape(len(phase_seed),N),phase_seed[:,1:]]))
        X=np.vstack([X,discrete])
        VV,MM,SS=obj.ev(X.T)
        npop=popsize*dim; rng=np.random.default_rng(seed+1)
        order=np.argsort(SS)[::-1]
        feasible=np.flatnonzero((MM["inner_chi"]>.10)&(MM["pos_fraction"]>=.75)&
                                (MM["outer_chi"]<-.15)&(MM["neg_fraction"]>=.15/.18))
        feasible=feasible[np.argsort(SS[feasible])[::-1]] if len(feasible) else feasible
        ntop=npop//2
        chosen_top=feasible[:min(ntop,len(feasible))]
        remaining=max(0,ntop-len(chosen_top))
        fallback=[i for i in order if i not in set(chosen_top.tolist())]
        if remaining:
            chosen_top=np.r_[chosen_top,np.asarray(fallback[:remaining],int)]
        pool=order[:min(4*npop,len(order))]
        pool=np.asarray([i for i in pool if i not in set(chosen_top.tolist())],int)
        take=npop-len(chosen_top)
        tail=rng.choice(pool,size=take,replace=False) if take else np.empty(0,int)
        chosen=np.r_[chosen_top,tail]
        pop=X[chosen]
        save_npz(od/"sobol_initial_checkpoint.npz",sobol_points=sob,candidate_X=X,
                 candidate_scores=SS,initial_population=pop,seed=np.asarray(seed))
        with (od/"sobol_top250.csv").open("w",newline="",encoding="utf-8") as fh:
            w=csv.writer(fh); w.writerow(["rank","score","variables","phasors"])
            VV=decode(X.T,mode)
            for rank,j in enumerate(order[:250],1):
                w.writerow([rank,float(SS[j]),json.dumps(X[j].tolist()),
                    json.dumps([[float(z.real),float(z.imag)] for z in VV[:,j]])])
        print(mode,"Sobol best",float(SS[order[0]]),"sign_band_feasible",len(feasible),flush=True)
        last=None
        checkpoint_files=sorted(od.glob(f"checkpoint_{mode}_*.npz"))
        start_chunk=1
        if checkpoint_files:
            latest_path=checkpoint_files[-1]
            latest_chunk=int(latest_path.stem.rsplit("_",1)[-1])
            state=np.load(latest_path)
            pop=np.asarray(state["population"])
            start_chunk=latest_chunk+1
            if latest_chunk>=3:
                last=SimpleNamespace(population=pop,
                    population_energies=np.asarray(state["energies"]),
                    x=np.asarray(state["best_x"]),fun=float(state["best_fun"]),
                    nfev=int(state["nfev"]))
            chunk_reports=[]
            for old_chunk in range(1,latest_chunk+1):
                old_summary=od/f"chunk_{old_chunk:02d}_summary.json"
                if old_summary.exists():
                    old=json.loads(old_summary.read_text(encoding="utf-8"))
                    cp=od/f"checkpoint_{mode}_{old_chunk:02d}.npz"
                    if cp.exists():
                        st=np.load(cp); ps=int(len(st["population"]))
                        old.setdefault("vectorized_objective_calls",int(old.get("nfev",0)))
                        old.setdefault("population_size",ps)
                        old.setdefault("estimated_candidate_score_evaluations",
                                        int(old.get("nfev",0))*ps)
                        old_summary.write_text(json.dumps(old,indent=2),encoding="utf-8")
                    chunk_reports.append(old)
            print(mode,"resume after checkpoint",latest_chunk,flush=True)
        else:
            chunk_reports=[]
        for chunk in range(start_chunk,4):
            sd=seed+1000+chunk
            res=differential_evolution(obj,bounds,strategy="best1bin",maxiter=15,
              popsize=popsize,tol=.002,mutation=(.45,1.0),recombination=.75,
              init=pop,x0=pop[np.argmin(obj(pop.T))],polish=False,seed=sd,
              updating="deferred",workers=1,vectorized=True,disp=True)
            checkpoint(od,mode,chunk,res,sd)
            pop=res.population; last=res
            summ={"mode":mode,"chunk":chunk,"seed":sd,"nfev":int(res.nfev),
              "vectorized_objective_calls":int(res.nfev),"population_size":int(len(res.population)),
              "estimated_candidate_score_evaluations":int(res.nfev*len(res.population)),
              "best_score":float(-res.fun),"success":bool(res.success),"message":str(res.message)}
            (od/f"chunk_{chunk:02d}_summary.json").write_text(json.dumps(summ,indent=2),encoding="utf-8")
            chunk_reports.append(summ)
            print(mode,summ,flush=True)
        if last is None: continue
        pool=np.vstack([last.population,X[order[:100]]])
        Vt,Mt,St=obj.ev(pool.T)
        coeff=T@Vt
        order2=np.argsort(St)[::-1]; selected=[]; seen=[]
        sign_feasible=(Mt["inner_chi"]>.10)&(Mt["pos_fraction"]>=.75)&\
            (Mt["outer_chi"]<-.15)&(Mt["neg_fraction"]>=.15/.18)&\
            (Mt["outer_E2"]>=.75*base["outer_E2"][0])&\
            (Mt["inner_E2"]>=.75*base["inner_E2"][0])&\
            (Mt["Emax"]<=1.20*base["Emax"][0])
        sign_ids_raw=np.flatnonzero(sign_feasible)
        def unique_ids(ids):
            kept=[]; vectors=[]
            for ix in ids:
                if any(np.linalg.norm(Vt[:,ix]-q)<.04 for q in vectors): continue
                kept.append(int(ix)); vectors.append(Vt[:,ix])
            return np.asarray(kept,dtype=int)
        sign_ids=unique_ids(sign_ids_raw)
        strict_feasible=sign_feasible&\
            (Mt["outer_corr"]<=base["outer_corr"][0]+1e-10)&\
            (Mt["outer_barrier"]<=base["outer_barrier"][0]+1e-10)
        strict_ids=unique_ids(np.flatnonzero(strict_feasible))
        for j in order2:
            if not sign_feasible[j]: continue
            v=Vt[:,j]
            if any(np.linalg.norm(v-q)<.04 for q in seen): continue
            selected.append(int(j)); seen.append(v)
            if len(selected)>=50: break
        # Pareto front among sign-band-feasible candidates: maximize negative
        # band width, negative outer torque strength, positive inner torque and
        # field floor; minimize barrier/corrugation and peak surface E.
        pareto_obj=np.column_stack([
            Mt["neg_fraction"],-Mt["outer_chi"],Mt["inner_chi"],
            np.minimum(Mt["outer_E2"]/base["outer_E2"][0],Mt["inner_E2"]/base["inner_E2"][0]),
            -Mt["outer_corr"],-Mt["outer_barrier"]/base["outer_barrier"][0],
            -Mt["Emax"]/base["Emax"][0]])
        pareto=[]
        for i in sign_ids:
            others=pareto_obj[sign_ids]
            dominated=np.any(np.all(others>=pareto_obj[i]-1e-12,axis=1)&
                             np.any(others>pareto_obj[i]+1e-12,axis=1))
            if not dominated: pareto.append(int(i))
        pareto_set=set(pareto); strict_set=set(strict_ids.tolist())
        for j in pareto:
            if int(j) in selected: continue
            if any(np.linalg.norm(Vt[:,j]-Vt[:,q])<.04 for q in selected): continue
            selected.append(int(j))
        candidates=[]
        for rank,j in enumerate(selected,1):
            phases=np.mod(np.angle(Vt[:,j]),2*np.pi)
            chi=Mt["chi"][:,j]
            roots=[float((RCHI[i]+RCHI[i+1])/2) for i in range(len(RCHI)-1) if chi[i]*chi[i+1]<0]
            c={
             "rank":rank,"score":float(St[j]),
             "sign_band_feasible":True,
             "strict_corrugation_barrier_feasible":bool(strict_feasible[j]),
             "pareto_front":bool(j in pareto_set),
             "amplitudes_rms_normalized":abs(Vt[:,j]).tolist(),
             "phases_absolute_deg_gauge_phi1_zero":(phases*180/np.pi).tolist(),
             "outer_mean_chirality_r47_r65":float(Mt["outer_chi"][j]),
             "outer_negative_fraction_chi_lt_m0p15":float(Mt["neg_fraction"][j]),
             "outer_negative_margin":float(Mt["neg_margin"][j]),
             "inner_mean_chirality_r20_r40":float(Mt["inner_chi"][j]),
             "inner_positive_fraction_chi_gt_p0p1":float(Mt["pos_fraction"][j]),
             "outer_mean_E2":float(Mt["outer_E2"][j]),"inner_mean_E2":float(Mt["inner_E2"][j]),
             "outer_corrugation":float(Mt["outer_corr"][j]),"inner_corrugation":float(Mt["inner_corr"][j]),
             "outer_barrier_per_kappa2":float(Mt["outer_barrier"][j]),
             "surface_Emax_U0_over_D":float(Mt["Emax"][j]),
             "max_neighbor_voltage_difference_U0":float(Mt["neighbor"][j]),
             "MFS_boundary_residual":float(np.max(np.abs(bres@Vt[:,j]))),
             "torque_roots_r20_r75":roots,
             "radial_chirality":[[float(r),float(x)] for r,x in zip(RCHI,chi)],
             "full_spectrum":[[int(kk),float(x.real),float(x.imag)] for kk,x in zip(k,coeff[:,j])],
             "voltage_phasors":[[float(z.real),float(z.imag)] for z in Vt[:,j]],
             "angular_profiles":{str(r):{"mean_E2":float(Mt["profile"][r][0][j]),
                 "corrugation":float(Mt["profile"][r][1][j]),
                 "barrier_per_kappa2":float(Mt["profile"][r][2][j])} for r in sorted(Mt["profile"])}
            }
            candidates.append(c)
        # Keep high-scoring rejected patterns separate so an all-negative
        # radial torque field cannot be mistaken for a counter-rotation design.
        rejected=[]
        rej_order=[int(j) for j in order2 if not sign_feasible[j]][:20]
        for j in rej_order:
            reasons=[]
            if Mt["inner_chi"][j]<=.10: reasons.append("inner_mean_chi_not_positive_enough")
            if Mt["pos_fraction"][j]<.75: reasons.append("inner_positive_fraction_below_0p75")
            if Mt["outer_chi"][j]>=-.15: reasons.append("outer_mean_chi_not_negative_enough")
            if Mt["neg_fraction"][j]<.15/.18: reasons.append("negative_width_below_0p15R")
            if Mt["outer_E2"][j]<.75*base["outer_E2"][0]: reasons.append("outer_RF_intensity_below_floor")
            if Mt["inner_E2"][j]<.75*base["inner_E2"][0]: reasons.append("inner_RF_intensity_below_floor")
            if Mt["Emax"][j]>1.20*base["Emax"][0]: reasons.append("surface_Emax_above_1p20_baseline")
            rejected.append({"score":float(St[j]),"reasons":reasons,
              "inner_mean_chirality":float(Mt["inner_chi"][j]),
              "inner_positive_fraction":float(Mt["pos_fraction"][j]),
              "outer_mean_chirality":float(Mt["outer_chi"][j]),
              "outer_negative_fraction":float(Mt["neg_fraction"][j]),
              "outer_corrugation":float(Mt["outer_corr"][j]),
              "surface_Emax_U0_over_D":float(Mt["Emax"][j])})
        (od/"rejected_top20.json").write_text(json.dumps(rejected,indent=2),encoding="utf-8")
        # Candidate phasors are stored as [[Re,Im],...]; retain full records
        # for all Pareto points as well as the score-ranked shortlist.
        pareto_json=[]
        for j in pareto[:50]:
            target=Vt[:,j]
            match=next((c for c in candidates if np.linalg.norm(
                np.asarray([complex(*z) for z in c["voltage_phasors"]])-target)<1e-10),None)
            if match is not None: pareto_json.append(match)
        report={"method":"Sobol global initialization + checkpointed differential evolution",
          "mode":mode,"sobol_seed":seed,"optimizer_seed_base":seed+1001,
          "sobol_samples":len(sob),"discrete_phase_permutation_seeds":int(len(phase_seed)),
          "initial_sign_band_feasible_count":int(len(feasible)),
          "initial_candidate_count_including_permutation_seeds":int(len(X)),
          "dimensions":dim,
          "optimizer":{"strategy":"best1bin","chunks":3,"generations_per_chunk":15,
            "population_size":int(len(last.population)),"mutation":[.45,1.0],
            "recombination":.75,"polish":False,"checkpoint_files":True,
            "objective":"vectorized multi-criterion with hard sign-band feasibility penalty",
            "chunk_summaries":chunk_reports},
          "objective_weights_and_guards":{"outer_negative_fraction":4.5,
            "outer_negative_margin":4.0,"inner_positive_fraction":1.6,
            "outer_corrugation_penalty":1.05,"inner_corrugation_penalty":.20,
            "outer_barrier_relative_penalty":.55,"field_floor_ratio":.75,
            "Emax_soft_threshold_ratio":1.15},
          "geometry":{"N":N,"a_D":A,"B_D":B,"R_D":R_D,"MFS_p":P,"Kmax":K},
          "voltage_constraint":{"common_frequency":True,"gauge_phi1_deg":0,
             "sum_abs_voltage_squared_U0_squared":12,
             "RMS_voltage_per_electrode_U0":1.0,
             "raw_amplitude_bounds":[.70,1.30] if mode=="phase_and_amplitude" else [1,1],
             "phase_bounds_rad":[-np.pi,np.pi]},
          "baseline_ADCFEB":baseinfo,
          "feasibility_filter":{"inner_mean_chi_gt_0p10":True,"inner_fraction_chi_gt_0p10_at_least_0p75":True,
            "outer_mean_chi_lt_minus_0p15":True,"outer_chi_lt_minus_0p15_width_at_least_0p15R":True,
            "inner_and_outer_mean_E2_at_least_0p75_baseline":True,"surface_Emax_at_most_1p20_baseline":True,
            "strict_candidate_also_requires_outer_corrugation_and_barrier_not_above_baseline":True},
          "evaluated_pool_counts":{"total_candidate_instances":int(len(St)),
            "sign_band_feasible_unique_phasors":int(len(sign_ids)),
            "strict_corrugation_barrier_feasible":int(len(strict_ids)),"pareto_count":int(len(pareto))},
          "top_candidates":candidates,"pareto_candidates":pareto_json,
          "strict_candidates_only":[c for c in candidates if c["strict_corrugation_barrier_feasible"]],
          "rejected_high_score_candidates":rejected,
          "scope":"2-D infinite-cylinder MFS phasor screening; no particle trajectories/stability/3-D end fields."}
        (od/"top_candidates_full_spectrum.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
        with (od/"top_candidates.csv").open("w",newline="",encoding="utf-8") as fh:
            fields=["rank","score","outer_mean_chirality_r47_r65","outer_negative_fraction_chi_lt_m0p15",
             "inner_mean_chirality_r20_r40","inner_positive_fraction_chi_gt_p0p1",
             "outer_corrugation","inner_corrugation","outer_barrier_per_kappa2",
             "outer_mean_E2","inner_mean_E2","surface_Emax_U0_over_D",
             "max_neighbor_voltage_difference_U0","amplitudes_rms_normalized",
             "phases_absolute_deg_gauge_phi1_zero","torque_roots_r20_r75"]
            w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
            for c in candidates:
                w.writerow({key:json.dumps(c[key],separators=(",",":")) if isinstance(c[key],list) else c[key] for key in fields})
        all_summaries.append({"mode":mode,"best":candidates[0]})
        print(mode,"best",candidates[0]["score"],candidates[0]["outer_mean_chirality_r47_r65"],
              candidates[0]["outer_corrugation"],flush=True)
    (OUT/"GLOBAL_SUMMARY.json").write_text(json.dumps({"seed":SEED,"baseline":baseinfo,
      "results":all_summaries,"scope":"2D MFS only; full RF-DC needed"},indent=2,ensure_ascii=False),encoding="utf-8")

if __name__=="__main__": main()
