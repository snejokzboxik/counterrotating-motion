"""Short, phase-continuous full-RK4 capture to audit M3 micromotion."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT/"research"/"v17c"/"v17c_17"))
import analytical_model as model  # noqa: E402
import run_pair_coulomb_scan as pair  # noqa: E402


def main():
    data = model.load_inputs()
    source = ROOT/"research"/"v17c"/"v17c_20b"/"results"/"refine12_240k_corrected_drag_trajectory.npz"
    archived = np.load(source)
    y0 = np.asarray(archived["states"][-1], dtype=float)
    tau_end = float(archived["step_index"][-1])*2*np.pi/float(archived["steps"])
    phase0 = np.asarray(archived["phase"], dtype=float)
    phase = (phase0 + np.array([tau_end, data["exact_p"][2]*tau_end])) % (2*np.pi)
    p = np.asarray(data["exact_p"], dtype=float)
    modes = data["modes"]
    family = np.asarray([m.family for m in modes], dtype=np.int64)
    azimuthal_k = np.asarray([m.k for m in modes], dtype=np.int64)
    coefficients = np.asarray([m.coefficient for m in modes], dtype=np.complex128)
    phys = data["candidate"]["physical_parameter_mapping_from_V20B"]
    raw = pair._integrate_trace(
        y0.copy(), phase, p, family, azimuthal_k, coefficients,
        data["charge_ratio_q2_q1"], float(phys["a1_over_R"]), float(phys["a2_over_R"]),
        data["Re_per_speed"][0], data["Re_per_speed"][1], data["gravity_z"],
        1000, 64, 0, 1)
    end, states, step_idx, work, turns, windows, rmin, rmax, zmax, vmax, dmin, avgr, avd, avuc, winrm, windm, wrk, completed, status = raw
    states = np.asarray(states)
    steps_per_period = 64
    tau = np.arange(states.shape[0], dtype=float)*(2*np.pi/steps_per_period)
    analytical = json.loads((BASE/"results"/"analytical_predictions.json").read_text(encoding="utf-8"))
    m3 = analytical["models"]["M3_full_MFS_SN_torus_Galerkin"]
    m3_particles = m3["inner_harmonic_balance"]["particles"]
    target_frequency = data["target"]["late_frequency_Hz"]
    target_radius = data["target"]["late_mean_radius_R"]
    records = []
    series = {}
    for i in range(2):
        xy = states[:,6*i]+1j*states[:,6*i+1]
        theta0 = float(np.angle(xy[0]))
        omega_tau = float(target_frequency[i]/data["f1_Hz"])
        rotating = xy*np.exp(-1j*(theta0+omega_tau*tau))
        disp = rotating-target_radius[i]
        # Discard secular/near-DC drift; the retained band is well above the
        # M3 slow-mode cutoff and below the 32-cycle/RF1 Nyquist frequency.
        freq_full = np.fft.fftfreq(len(disp), d=1.0/steps_per_period)
        spectrum = np.fft.fft(disp-np.mean(disp))
        spectrum[np.abs(freq_full) < 0.05] = 0
        filtered = np.fft.ifft(spectrum)
        window = np.hanning(len(disp))
        spec_win = np.fft.fft((disp-np.mean(disp))*window)
        amp = 2*np.abs(spec_win)/max(1.0,np.sum(window))
        positive = np.flatnonzero((freq_full >= 0.05) & (freq_full > 0))
        peaks = positive[np.argsort(amp[positive])[-12:][::-1]]
        peak_rows = [{"cycles_per_RF1_period":float(freq_full[j]),"amplitude_R":float(amp[j])}
                     for j in peaks]
        carrier = []
        omega_dimless = omega_tau
        for m in modes:
            sigma = m.frequency-m.k*omega_dimless
            carrier.append({"family":m.family+1,"k":m.k,
                            "absolute_cycles_per_RF1_period":float(abs(sigma)),
                            "signed_rotating_cycles_per_RF1_period":float(sigma)})
        rms_hp = float(np.sqrt(np.mean(np.abs(filtered)**2)))
        m3_rms = float(m3_particles[i]["rms_micromotion"])
        records.append({
            "particle":i+1,"status":int(status),"completed_periods":int(completed),
            "sample_count":int(len(disp)),"steps_per_period":steps_per_period,
            "absolute_start_RF_phases_rad":phase.tolist(),
            "late_reference_radius_R":float(target_radius[i]),
            "late_reference_frequency_Hz":float(target_frequency[i]),
            "highpass_cutoff_cycles_per_RF1_period":0.05,
            "full_RK4_highpass_micromotion_RMS_R":rms_hp,
            "M3_phase_torus_micromotion_RMS_R":m3_rms,
            "relative_difference_percent":float(100*(rms_hp-m3_rms)/m3_rms),
            "dominant_highpass_spectral_peaks":peak_rows,
            "M3_linear_RF_carrier_offsets":carrier,
        })
        series[f"particle{i+1}_rotating_frame_displacement"] = np.column_stack((disp.real,disp.imag))
    out = {
        "study":"V17C.21 short full-RHS micromotion audit",
        "source_240k_npz_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "rhs":"V17C.17 pair RK4, full eight-mode RF, Coulomb, common V20B gravity, Cunningham-corrected Stokes × Schiller–Naumann; polarization off.",
        "method":"1000 RF1-period, 64-step/period replay from the archived late state; external RF phases continued at absolute time. Rotating-frame displacement high-pass filtered above 0.05 cycles per RF1 period.",
        "rows":records,
        "qualification":"Short-window high-pass RMS is not identical to full torus RMS: it samples a finite segment and removes sub-cutoff modulation. Spectral lines are resolved numerically but do not prove quasiperiodicity.",
    }
    (BASE/"results"/"micromotion_spectrum_audit.json").write_text(
        json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    np.savez_compressed(BASE/"results"/"micromotion_spectrum_audit.npz",
                        tau=tau,phase=phase,initial_state=y0,states=states,**series)
    print(json.dumps(out,indent=2,ensure_ascii=True))


if __name__ == "__main__":
    main()
