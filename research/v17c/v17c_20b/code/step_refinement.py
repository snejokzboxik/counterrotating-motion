import json,math
import numpy as np
import attractor_search as a

OUT=a.OUT
def main():
 cfg=a.read_json(OUT/'candidate_parameters.json');field=a.load_field();start=cfg['matched_common_start'];rows=[]
 for c in cfg['candidates']:
  p,q=a.arrays(c,field)
  raw=a.PAIR._integrate_trace(np.asarray(start['initial'],float),np.asarray(start['phase'],float),p,field['family'],field['k'],field['coeff'],q,
    c['a1_over_R'],c['a2_over_R'],c['re1_per_speed'],c['re2_per_speed'],c['gravity_z'],3000,128,200,1280)
  end,states,idx,wh,turns,windows,rmin,rmax,zmax,vmax,dmin,avgr,avd,avuc,winrm,windm,wrk,completed,status=raw
  np.savez_compressed(OUT/f"rk4_128_{c['legacy_candidate']}_3000p.npz",states=states,step_index=idx,initial=start['initial'],phase=start['phase'],periods=3000,steps=128)
  dop=np.load(OUT/f"dop853_{c['legacy_candidate']}_3000p.npz")
  baseline=np.load(OUT/f"baseline_{c['id']}_15000p_rk464.npz")
  dop_end=dop['states'][-1];base_idx=np.argmin(abs(baseline['step_index']/64-3000));base_end=baseline['states'][base_idx]
  r={'candidate':c['id'],'steps':128,'periods':3000,'status':int(status),'turns_after_200p':np.asarray(turns).tolist(),
    'frequency_Hz':(np.asarray(turns)*c['f1_Hz']/(3000-200)).tolist(),'rmin_R':np.asarray(rmin).tolist(),'rmax_R':np.asarray(rmax).tolist(),
    'minimum_pair_distance_R':float(dmin),'max_abs_state_difference_rk4_64_vs_128':float(np.max(abs(base_end-end))),
    'max_abs_state_difference_rk4_128_vs_dop853':float(np.max(abs(dop_end-end))),
    'rms_state_difference_rk4_128_vs_dop853':float(np.sqrt(np.mean((dop_end-end)**2)))}
  rows.append(r);print(c['legacy_candidate'],r,flush=True)
 a.write_json(OUT/'step_refinement_3000p.json',{'rows':rows,'note':'Same initial condition, full complex MFS, corrected drag; DOP853 states sampled at matching 10-period grid.'})
if __name__=='__main__':main()
