"""Continuous, phase-preserving RF/DC envelopes over the validated V17C.9 RHS.

Protocol times are measured in periods of the primary RF component.  The
terminal coefficients are exactly those in ``p``; envelope factors multiply
the two RF amplitudes and the DC coefficient only.  Carrier phases are never
reset.  The constant protocol calls the original RHS unchanged, for a strict
P0 equivalence check against ``baseline.basin_scan.integrate_one``.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "v17c_9" / "baseline"
sys.path.insert(0, str(BASE))
from basin_scan import _finite_body_contact, _rhs as _rhs_fixed  # noqa: E402

TWOPI = 2.0 * math.pi
F1_HZ = 100_000.0
RMIN_CLASS = 0.02
ESCAPE_R = 1.0
ESCAPE_Z = 2.0

# Protocol vector indices. See protocol_spec() below.
KIND = 0
TA, TB, TDC, DELAY_B, DELAY_DC = 1, 2, 3, 4, 5
G0A, G0B, G0DC, SHAPE, POWER, DC_MODE = 6, 7, 8, 9, 10, 11
CAP_A, CAP_B, CAP_DC, CAP_RAMP, CAP_HOLD, CAP_TRANS, TRANS_SHAPE = range(12, 19)


def protocol_spec():
    return {
        "units": "time knots and durations are cycles of f1; carrier phases remain continuous",
        "vector_fields": [
            "kind (0=constant, 1=direct ramps with A starting at zero, 2=two-stage capture, 3=direct ramps with B starting first)",
            "T_A_cycles", "T_B_cycles", "T_DC_cycles", "delay_B_cycles", "delay_DC_cycles",
            "gA_start", "gB_start", "gDC_start", "shape_id", "power_p", "dc_mode",
            "A_capture_multiplier", "B_capture_multiplier", "DC_capture_multiplier",
            "capture_ramp_cycles", "capture_hold_cycles", "capture_to_final_cycles", "transition_shape_id",
        ],
        "shape_ids": {"0": "linear", "1": "half-cosine", "2": "cubic smoothstep", "3": "quintic smootherstep", "4": "s^p"},
        "dc_mode": {"0": "independent DC envelope", "1": "gDC=gA*gB (equals gRF^2 for synchronized RF ramps)"},
        "capture_definition": "ramp each start level to its capture multiplier, hold, then ramp to 1; all segments are continuous",
    }


@njit(cache=True, nogil=True)
def _shape(shape_id, s, power):
    if s <= 0.0:
        return 0.0
    if s >= 1.0:
        return 1.0
    if shape_id == 0:
        return s
    if shape_id == 1:
        return 0.5 * (1.0 - math.cos(math.pi * s))
    if shape_id == 2:
        return s * s * (3.0 - 2.0 * s)
    if shape_id == 3:
        return s * s * s * (s * (6.0 * s - 15.0) + 10.0)
    return s ** power


@njit(cache=True, nogil=True)
def _ramp(start, end, cycle, delay, duration, shape_id, power):
    if duration <= 0.0:
        return end
    s = (cycle - delay) / duration
    if s <= 0.0:
        return start
    if s >= 1.0:
        return end
    return start + (end - start) * _shape(shape_id, s, power)


@njit(cache=True, nogil=True)
def _capture(start, cap, cycle, ramp_cycles, hold_cycles, transition_cycles,
             shape_id, transition_shape, power):
    if ramp_cycles > 0.0 and cycle < ramp_cycles:
        return _ramp(start, cap, cycle, 0.0, ramp_cycles, shape_id, power)
    after_ramp = cycle - ramp_cycles
    if after_ramp <= hold_cycles:
        return cap
    if transition_cycles <= 0.0:
        return 1.0
    return _ramp(cap, 1.0, after_ramp, hold_cycles, transition_cycles,
                 transition_shape, power)


@njit(cache=True, nogil=True)
def _envelopes(tau, protocol):
    kind = int(protocol[KIND])
    if kind == 0:
        return 1.0, 1.0, 1.0
    cycle = tau / TWOPI
    shape_id = int(protocol[SHAPE])
    power = protocol[POWER]
    if kind == 1:
        ga = _ramp(protocol[G0A], 1.0, cycle, 0.0, protocol[TA], shape_id, power)
        gb = _ramp(protocol[G0B], 1.0, cycle, protocol[DELAY_B], protocol[TB], shape_id, power)
        if int(protocol[DC_MODE]) == 1:
            gd = ga * gb
        else:
            gd = _ramp(protocol[G0DC], 1.0, cycle, protocol[DELAY_DC], protocol[TDC], shape_id, power)
        return ga, gb, gd
    if kind == 3:
        # B starts immediately; A is delayed by DELAY_DC. DELAY_B is used for
        # an independent DC delay unless DC follows the RF product.
        ga = _ramp(protocol[G0A], 1.0, cycle, protocol[DELAY_DC], protocol[TA], shape_id, power)
        gb = _ramp(protocol[G0B], 1.0, cycle, 0.0, protocol[TB], shape_id, power)
        if int(protocol[DC_MODE]) == 1:
            gd = ga * gb
        else:
            gd = _ramp(protocol[G0DC], 1.0, cycle, protocol[DELAY_B], protocol[TDC], shape_id, power)
        return ga, gb, gd
    ga = _capture(protocol[G0A], protocol[CAP_A], cycle, protocol[CAP_RAMP],
                  protocol[CAP_HOLD], protocol[CAP_TRANS], shape_id,
                  int(protocol[TRANS_SHAPE]), power)
    gb = _capture(protocol[G0B], protocol[CAP_B], cycle, protocol[CAP_RAMP],
                  protocol[CAP_HOLD], protocol[CAP_TRANS], shape_id,
                  int(protocol[TRANS_SHAPE]), power)
    gd = _capture(protocol[G0DC], protocol[CAP_DC], cycle, protocol[CAP_RAMP],
                  protocol[CAP_HOLD], protocol[CAP_TRANS], shape_id,
                  int(protocol[TRANS_SHAPE]), power)
    return ga, gb, gd


@njit(cache=True, nogil=True)
def _rhs_enveloped(t, y, p, protocol, phase, family, k, coeff, qratio, out,
                   oseen_Re_per_speed=0.0, hydro_mode=0, a1_over_R=0.0,
                   a2_over_R=0.0, oseen_Re2_per_speed=-1.0):
    ga, gb, gd = _envelopes(t, protocol)
    p_now = p.copy()
    p_now[0] *= ga
    p_now[1] *= gb
    p_now[3] *= gd
    _rhs_fixed(t, y, p_now, phase, family, k, coeff, qratio, out,
               oseen_Re_per_speed, hydro_mode, a1_over_R, a2_over_R,
               oseen_Re2_per_speed)


@njit(cache=True, nogil=True)
def _integrate_core(initial, phase, p, protocol, family, k, coeff, periods,
                    steps, skip, qratio, a1_over_R, a2_over_R, log_every):
    dt = TWOPI / steps
    y = initial.copy()
    y1 = np.empty(12); y2 = np.empty(12); y3 = np.empty(12)
    y4 = np.empty(12); tmp = np.empty(12); yn = np.empty(12)
    total_turns = np.zeros(2)
    window_turns = np.zeros((3, 2))
    min_radius = np.array([math.hypot(y[0], y[1]), math.hypot(y[6], y[7])])
    max_radius = min_radius.copy()
    max_abs_dtheta = np.zeros(2)
    max_abs_z = np.array([abs(y[2]), abs(y[8])])
    dx = y[0]-y[6]; dy = y[1]-y[7]; dz = y[2]-y[8]
    min_distance = math.sqrt(dx*dx+dy*dy+dz*dz)
    particle_contact = False
    electrode_contact = False
    escaped = False
    singular = False
    steps_done = 0
    if log_every > 0:
        nmax = periods*steps//log_every + 3
        log_t = np.empty(nmax, dtype=np.float64)
        log_y = np.empty((nmax, 12), dtype=np.float64)
        log_count = 1
        log_t[0] = 0.0
        log_y[0] = y
    else:
        log_t = np.empty(0, dtype=np.float64)
        log_y = np.empty((0, 12), dtype=np.float64)
        log_count = 0
    window_edges = np.array([skip, skip+(periods-skip)//3,
                             skip+2*(periods-skip)//3, periods])
    total_steps = periods*steps
    for step in range(total_steps):
        t = step*dt
        _rhs_enveloped(t,y,p,protocol,phase,family,k,coeff,qratio,y1,0.0,0,a1_over_R,a2_over_R,-1.0)
        for j in range(12): tmp[j]=y[j]+0.5*dt*y1[j]
        dx=tmp[0]-tmp[6]; dy=tmp[1]-tmp[7]; dz=tmp[2]-tmp[8]
        min_distance=min(min_distance,math.sqrt(dx*dx+dy*dy+dz*dz))
        contact=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact != 0:
            particle_contact = contact == 1; electrode_contact = contact == 2
            steps_done=step+1
            if log_every > 0:
                log_t[log_count]=t+0.5*dt; log_y[log_count]=tmp; log_count+=1
            break
        _rhs_enveloped(t+0.5*dt,tmp,p,protocol,phase,family,k,coeff,qratio,y2,0.0,0,a1_over_R,a2_over_R,-1.0)
        for j in range(12): tmp[j]=y[j]+0.5*dt*y2[j]
        dx=tmp[0]-tmp[6]; dy=tmp[1]-tmp[7]; dz=tmp[2]-tmp[8]
        min_distance=min(min_distance,math.sqrt(dx*dx+dy*dy+dz*dz))
        contact=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact != 0:
            particle_contact = contact == 1; electrode_contact = contact == 2
            steps_done=step+1
            if log_every > 0:
                log_t[log_count]=t+0.5*dt; log_y[log_count]=tmp; log_count+=1
            break
        _rhs_enveloped(t+0.5*dt,tmp,p,protocol,phase,family,k,coeff,qratio,y3,0.0,0,a1_over_R,a2_over_R,-1.0)
        for j in range(12): tmp[j]=y[j]+dt*y3[j]
        dx=tmp[0]-tmp[6]; dy=tmp[1]-tmp[7]; dz=tmp[2]-tmp[8]
        min_distance=min(min_distance,math.sqrt(dx*dx+dy*dy+dz*dz))
        contact=_finite_body_contact(tmp,a1_over_R,a2_over_R)
        if contact != 0:
            particle_contact = contact == 1; electrode_contact = contact == 2
            steps_done=step+1
            if log_every > 0:
                log_t[log_count]=t+dt; log_y[log_count]=tmp; log_count+=1
            break
        _rhs_enveloped(t+dt,tmp,p,protocol,phase,family,k,coeff,qratio,y4,0.0,0,a1_over_R,a2_over_R,-1.0)
        for j in range(12): yn[j]=y[j]+dt*(y1[j]+2*y2[j]+2*y3[j]+y4[j])/6.0
        r0=math.hypot(y[0],y[1]); r1=math.hypot(yn[0],yn[1])
        r2=math.hypot(y[6],y[7]); r3=math.hypot(yn[6],yn[7])
        min_radius[0]=min(min_radius[0],r0,r1); min_radius[1]=min(min_radius[1],r2,r3)
        max_radius[0]=max(max_radius[0],r0,r1); max_radius[1]=max(max_radius[1],r2,r3)
        max_abs_z[0]=max(max_abs_z[0],abs(y[2]),abs(yn[2])); max_abs_z[1]=max(max_abs_z[1],abs(y[8]),abs(yn[8]))
        dx=yn[0]-yn[6]; dy=yn[1]-yn[7]; dz=yn[2]-yn[8]
        d2=dx*dx+dy*dy+dz*dz
        min_distance=min(min_distance,math.sqrt(d2))
        if a1_over_R+a2_over_R>0.0 and d2<(a1_over_R+a2_over_R)**2:
            particle_contact=True; steps_done=step+1; y[:]=yn
            if log_every>0 and (step+1)%log_every==0:
                log_t[log_count]=t+dt; log_y[log_count]=y; log_count+=1
            break
        if ((a1_over_R>0.0 and (max_radius[0]+a1_over_R>ESCAPE_R or max_abs_z[0]+a1_over_R>ESCAPE_Z)) or
            (a2_over_R>0.0 and (max_radius[1]+a2_over_R>ESCAPE_R or max_abs_z[1]+a2_over_R>ESCAPE_Z))):
            electrode_contact=True; steps_done=step+1; y[:]=yn
            if log_every>0 and (step+1)%log_every==0:
                log_t[log_count]=t+dt; log_y[log_count]=y; log_count+=1
            break
        if d2<1e-10:
            singular=True; steps_done=step+1
            break
        cycle=step//steps
        if cycle>=skip:
            win=2
            if cycle<window_edges[1]: win=0
            elif cycle<window_edges[2]: win=1
            for i in range(2):
                b=6*i
                rr=y[b]*y[b]+y[b+1]*y[b+1]
                rn=yn[b]*yn[b]+yn[b+1]*yn[b+1]
                if rr>RMIN_CLASS*RMIN_CLASS and rn>RMIN_CLASS*RMIN_CLASS:
                    cross=y[b]*yn[b+1]-y[b+1]*yn[b]
                    dot=y[b]*yn[b]+y[b+1]*yn[b+1]
                    dtheta=math.atan2(cross,dot)/TWOPI
                    total_turns[i]+=dtheta; window_turns[win,i]+=dtheta
                    max_abs_dtheta[i]=max(max_abs_dtheta[i],abs(dtheta)*TWOPI)
        if not np.all(np.isfinite(yn)):
            singular=True; steps_done=step+1
            break
        if max_radius[0]>ESCAPE_R or max_radius[1]>ESCAPE_R or max_abs_z[0]>ESCAPE_Z or max_abs_z[1]>ESCAPE_Z:
            escaped=True; steps_done=step+1; y[:]=yn
            if log_every>0 and (step+1)%log_every==0:
                log_t[log_count]=t+dt; log_y[log_count]=y; log_count+=1
            break
        y,yn=yn,y
        steps_done=step+1
        if log_every>0 and (step+1)%log_every==0:
            log_t[log_count]=t+dt; log_y[log_count]=y; log_count+=1
    completed=steps_done//steps
    retained=not escaped and not singular and not particle_contact and not electrode_contact and completed>=periods
    axis_ambiguous=min_radius[0]<RMIN_CLASS or min_radius[1]<RMIN_CLASS
    signs=np.zeros((3,2)); stable=True
    for w in range(3):
        for i in range(2):
            signs[w,i]=0.0 if abs(window_turns[w,i])<0.05 else math.copysign(1.0,window_turns[w,i])
            if signs[w,i]==0.0: stable=False
    opposite=(signs[0,0]*signs[0,1]<0.0 and signs[1,0]*signs[1,1]<0.0 and signs[2,0]*signs[2,1]<0.0)
    enough=abs(total_turns[0])>=10.0 and abs(total_turns[1])>=10.0
    if singular: category="point_singularity_unresolved"
    elif particle_contact: category="finite_particle_contact"
    elif electrode_contact: category="finite_particle_wall_contact"
    elif escaped: category="lost_retention"
    elif axis_ambiguous: category="axis_ambiguous"
    elif retained and enough and stable and opposite: category="CR_screen"
    elif retained and enough and stable and signs[0,0]*signs[0,1]>0.0: category="same_direction"
    elif retained and abs(total_turns[0])<0.25 and abs(total_turns[1])<0.25: category="localized_or_slow"
    elif retained: category="inconclusive"
    else: category="incomplete"
    freq=total_turns*F1_HZ/max(1,completed-skip)
    return (y,total_turns,window_turns,min_radius,max_radius,max_abs_z,max_abs_dtheta,
            min_distance,completed,retained,escaped,singular,category,freq,
            log_t[:log_count].copy(),log_y[:log_count].copy())


def constant_protocol() -> np.ndarray:
    a = np.zeros(19, dtype=np.float64)
    a[KIND] = 0
    a[POWER] = 1.0
    return a


def ramp_protocol(*, duration: float, shape: int, power: float = 1.0,
                  g0: float = 0.0, dc_mode: int = 0,
                  duration_b: float | None = None, duration_dc: float | None = None,
                  delay_b: float = 0.0, delay_dc: float = 0.0,
                  g0_b: float | None = None, g0_dc: float | None = None) -> np.ndarray:
    a = np.zeros(19, dtype=np.float64)
    a[KIND] = 1; a[TA] = duration; a[TB] = duration if duration_b is None else duration_b
    a[TDC] = duration if duration_dc is None else duration_dc
    a[DELAY_B] = delay_b; a[DELAY_DC] = delay_dc
    a[G0A] = g0; a[G0B] = g0 if g0_b is None else g0_b
    a[G0DC] = g0 if g0_dc is None else g0_dc
    a[SHAPE] = shape; a[POWER] = power; a[DC_MODE] = dc_mode
    return a


def b_first_protocol(*, duration_a: float, duration_b: float, duration_dc: float,
                     delay_a: float, delay_dc: float, shape: int,
                     power: float = 1.0, g0: float = 0.0,
                     dc_mode: int = 0) -> np.ndarray:
    """Ramp B first, then A; both RF carriers retain their absolute phase."""
    a = np.zeros(19, dtype=np.float64)
    a[KIND] = 3
    a[TA] = duration_a; a[TB] = duration_b; a[TDC] = duration_dc
    a[DELAY_DC] = delay_a; a[DELAY_B] = delay_dc
    a[G0A] = g0; a[G0B] = g0; a[G0DC] = g0
    a[SHAPE] = shape; a[POWER] = power; a[DC_MODE] = dc_mode
    return a


def capture_protocol(*, capture_a: float, capture_b: float, capture_dc: float,
                     ramp_cycles: float, hold_cycles: float, transition_cycles: float,
                     shape: int = 3, transition_shape: int = 3, power: float = 1.0,
                     start_a: float = 0.0, start_b: float = 0.0,
                     start_dc: float = 0.0) -> np.ndarray:
    a = np.zeros(19, dtype=np.float64)
    a[KIND] = 2
    a[G0A] = start_a; a[G0B] = start_b; a[G0DC] = start_dc
    a[SHAPE] = shape; a[POWER] = power
    a[CAP_A] = capture_a; a[CAP_B] = capture_b; a[CAP_DC] = capture_dc
    a[CAP_RAMP] = ramp_cycles; a[CAP_HOLD] = hold_cycles
    a[CAP_TRANS] = transition_cycles; a[TRANS_SHAPE] = transition_shape
    return a


def protocol_duration_cycles(protocol: np.ndarray) -> float:
    if int(protocol[KIND]) == 0:
        return 0.0
    if int(protocol[KIND]) == 1:
        return max(float(protocol[TA]), float(protocol[DELAY_B] + protocol[TB]),
                   float(protocol[DELAY_DC] + protocol[TDC]))
    if int(protocol[KIND]) == 3:
        return max(float(protocol[DELAY_DC] + protocol[TA]), float(protocol[TB]),
                   float(protocol[DELAY_B] + protocol[TDC]))
    return float(protocol[CAP_RAMP] + protocol[CAP_HOLD] + protocol[CAP_TRANS])


def integrate(initial, phase, p, protocol, family, k, coeff, periods, steps, skip,
              qratio, a1_over_R=0.0002, a2_over_R=0.00189, log_every=0):
    result = _integrate_core(np.asarray(initial, dtype=np.float64), np.asarray(phase, dtype=np.float64),
        np.asarray(p, dtype=np.float64), np.asarray(protocol, dtype=np.float64),
        np.asarray(family, dtype=np.int64), np.asarray(k, dtype=np.int64),
        np.asarray(coeff, dtype=np.complex128), int(periods), int(steps), int(skip),
        float(qratio), float(a1_over_R), float(a2_over_R), int(log_every))
    keys = ("final_state","turns","window_turns","r_min","r_max","abs_z_max",
            "max_abs_inner_step_dtheta_rad","minimum_separation","completed_periods",
            "retained","escaped","singular_unresolved","category","frequency_Hz",
            "log_tau","log_state")
    return dict(zip(keys, result))
