"""Reproducible dimensional screen for the V17C.16 trial pair.

All input quantities below are dimensional measurements/assumptions. The
Local15 field coefficients are used only as a frozen transfer example, not as
an optimized or physically validated solution for the new pair.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

EPS0 = 8.854_187_8128e-12
E_CHARGE = 1.602_176_634e-19
K_B = 1.380_649e-23
G = 9.80665
TORR_PA = 133.32236842105263

PAIR = {
    "radii_m": [35e-6, 45e-6],
    "particle_densities_kg_m3": [1050.0, 1050.0],
    "charges_e": [400_000.0, 290_121.0],
    "temperature_K": 293.15,
    "air_pressure_Pa": 101_325.0,
    "air_density_kg_m3": 1.2,
    "air_dynamic_viscosity_Pa_s": 1.81e-5,
    "air_mean_free_path_m": 65.1e-9,
    "f1_Hz": 47.5,
    "frequency_ratio_nu": 3.1118,
    "radial_scale_R_m": 5e-3,
}

# Frozen Local15 ideal-field coefficients and audited V17C 24-rod transfer data.
# h2/h4 map peak electrode voltage to the leading potential harmonic at R.
LOCAL15 = {"A": 0.03716194907442204,
           "B": 0.1626938199343468,
           "beta": 0.017273303124515367,
           "lambda1_abstract": 0.20975541409005094,
           "lambda2_abstract": 0.12617311138613094,
           "nu": 3.1118029452860356}
MFS = {"electrodes": 24, "R_over_D": 0.95, "rod_radius_over_D": 0.05,
       "shield_radius_over_D": 1.10,
       "h_plus2": 0.870316891091623,
       "h_minus4": 0.7982172624273378,
       "rod_surface_E_per_V_at_D_1m_V_m": [26.363126866924716,
                                           26.442998019675162],
       "shield_surface_E_per_V_at_D_1m_V_m": [17.348310458627335,
                                               17.29602544161173]}


def _sphere_mass(radius: float, density: float) -> float:
    return 4.0 * math.pi * density * radius**3 / 3.0


def build_screen() -> dict:
    a = PAIR["radii_m"]
    rho = PAIR["particle_densities_kg_m3"]
    q = [z * E_CHARGE for z in PAIR["charges_e"]]
    m = [_sphere_mass(a[i], rho[i]) for i in range(2)]
    eta, rho_f = PAIR["air_dynamic_viscosity_Pa_s"], PAIR["air_density_kg_m3"]
    nu_f = eta / rho_f
    R = PAIR["radial_scale_R_m"]
    omega1 = 2.0 * math.pi * PAIR["f1_Hz"]
    omega2 = omega1 * PAIR["frequency_ratio_nu"]
    q_over_m = [q[i] / m[i] for i in range(2)]
    lambda_stokes = [6.0 * math.pi * eta * a[i] / (m[i] * omega1)
                     for i in range(2)]
    tau_particle = [m[i] / (6.0 * math.pi * eta * a[i]) for i in range(2)]
    kn = [PAIR["air_mean_free_path_m"] / a[i] for i in range(2)]
    # Kim et al. correlation, with Kn = mean free path / particle radius.
    # Keep the convention explicit because diameter-based Kn changes the
    # coefficients and caused a previous parameterization mismatch.
    cc = [1.0 + kn[i] * (1.165 + 0.483 * math.exp(-0.997 / kn[i]))
          for i in range(2)]
    e_surface = [q[i] / (4.0 * math.pi * EPS0 * a[i]**2) for i in range(2)]
    ps_epsilon_r = 2.6
    pauthenier_factor = 3.0*ps_epsilon_r/(ps_epsilon_r+2.0)

    # Local15 coefficient mapping with particle 1 as the reference species.
    vscale = m[0] * R**2 * omega1**2 / q[0]
    U1 = LOCAL15["A"] * vscale / MFS["h_plus2"]
    U2 = LOCAL15["B"] * vscale / MFS["h_minus4"]
    V2 = LOCAL15["beta"] * vscale
    D = R / MFS["R_over_D"]
    gap_rod_shield = (MFS["shield_radius_over_D"] - 1.0
                      - MFS["rod_radius_over_D"]) * D
    gap_adjacent = (2.0 * math.sin(math.pi / MFS["electrodes"])
                    - 2.0 * MFS["rod_radius_over_D"]) * D
    field_scale = 1.0 / D
    E_rod_rf1 = MFS["rod_surface_E_per_V_at_D_1m_V_m"][0] * field_scale * U1
    E_rod_rf2 = MFS["rod_surface_E_per_V_at_D_1m_V_m"][1] * field_scale * U2
    E_shield_rf1 = MFS["shield_surface_E_per_V_at_D_1m_V_m"][0] * field_scale * U1
    E_shield_rf2 = MFS["shield_surface_E_per_V_at_D_1m_V_m"][1] * field_scale * U2

    mass_ratio = m[1] / m[0]
    charge_ratio = q[1] / q[0]
    alpha = [1.0, q_over_m[1] / q_over_m[0]]
    kappa_base = q[0]**2 / (4.0 * math.pi * EPS0 * m[0] * R**3 * omega1**2)
    C12 = kappa_base * charge_ratio
    gravity_hat = G / (R * omega1**2)
    # z equilibria in the ideal DC quadrupole if its axis is vertical.
    z_eq_R = [-gravity_hat * (1.0 - rho_f/rho[i])
              / (2.0 * LOCAL15["beta"] * alpha[i])
              for i in range(2)]
    e_balance = [G / q_over_m[i] for i in range(2)]

    # Use the published Local15 linear-response displacements only as a
    # screening proxy for RF micromotion speed; these are not new trajectories.
    micromotion = []
    try:
        v17c_root = Path(__file__).resolve().parents[2]
        old = json.loads((v17c_root / "v17c_15" / "results" /
                          "moving_orbit_correction_local15.json").read_text(
                              encoding="utf-8"))
        for i, name in enumerate(("species1", "species2")):
            row = old["species"][name]
            velocity_amps = [abs(mode["sigma"]) * mode["micromotion_amplitude"]
                             for mode in row["mode_responses"]]
            slow = abs(row["moving_orbit_corrected"]["rho"]
                       * row["moving_orbit_corrected"]["omega"])
            u_rms = math.sqrt(slow**2 + sum(v*v / 2.0 for v in velocity_amps))
            u_bound = slow + sum(velocity_amps)
            re_per_unit_speed = 2.0 * rho_f * a[i] * R * omega1 / eta
            re_rms = re_per_unit_speed * u_rms
            re_bound = re_per_unit_speed * u_bound
            micromotion.append({
                "source": "V17C.15 frozen Local15 moving-orbit linear response; screening only",
                "dimensionless_RF_velocity_amplitudes": velocity_amps,
                "dimensionless_slow_velocity": slow,
                "physical_RMS_speed_m_s": u_rms * R * omega1,
                "diameter_Re_RMS_proxy": re_rms,
                "Oseen_fraction_3Re_over_16_RMS_proxy": 3.0 * re_rms / 16.0,
                "diameter_Re_conservative_phase_bound": re_bound,
                "Oseen_fraction_conservative_bound": 3.0 * re_bound / 16.0,
            })
    except (OSError, KeyError, json.JSONDecodeError):
        micromotion = [{"status": "V17C.15 response file unavailable"} for _ in range(2)]

    # Rough Paschen plane-parallel scale only. It is not a prediction for the
    # nonuniform RF electrode field or a discharge clearance certification.
    def paschen_voltage(gap_m: float) -> float:
        pd_torr_cm = (PAIR["air_pressure_Pa"] / TORR_PA) * gap_m * 100.0
        Ap, Bp, gamma = 15.0, 365.0, 0.01
        den = math.log(Ap * pd_torr_cm) - math.log(math.log(1.0 + 1.0/gamma))
        return Bp * pd_torr_cm / den if den > 0 else float("nan")

    return {
        "status": "dimensional screen; no coupled orbit or breakdown clearance is claimed",
        "input": PAIR,
        "constants": {"epsilon0_F_m": EPS0, "elementary_charge_C": E_CHARGE,
                      "gravity_m_s2": G, "air_kinematic_viscosity_m2_s": nu_f},
        "charging_screen": {
            "polystyrene_relative_permittivity_illustrative": ps_epsilon_r,
            "Pauthenier_dielectric_factor_3epsilon_over_epsilon_plus_2": pauthenier_factor,
            "caveat": "Equivalent uniform charger field from the Pauthenier saturation relation; not a measured charging protocol for these 70-90 um beads.",
        },
        "particles": [{
            "radius_m": a[i], "diameter_m": 2.0*a[i],
            "density_kg_m3": rho[i], "mass_kg": m[i],
            "charge_C": q[i], "charge_e": PAIR["charges_e"][i],
            "q_over_m_C_kg": q_over_m[i],
            "isolated_surface_field_V_m": e_surface[i],
            "field_fraction_of_3MV_m_reference": e_surface[i] / 3.0e6,
            "illustrative_Pauthenier_charger_field_for_target_charge_V_m":
                e_surface[i]/pauthenier_factor,
            "lambda_stokes_at_Omega1": lambda_stokes[i],
            "particle_velocity_relaxation_time_s": tau_particle[i],
            "particle_Stokes_number_at_omega1": omega1*tau_particle[i],
            "particle_Stokes_number_at_omega2": omega2*tau_particle[i],
            "Kn_mean_free_path_over_radius": kn[i],
            "Cunningham_slip_factor": cc[i],
            "unsteady_Stokes_parameter_a_sqrt_omega_over_nu": [
                a[i]*math.sqrt(omega1/nu_f), a[i]*math.sqrt(omega2/nu_f)],
            "added_mass_over_Stokes_drag_fraction_alpha2_over_9": [
                (a[i]**2*omega1/nu_f)/9.0, (a[i]**2*omega2/nu_f)/9.0],
            "gravity_terminal_speed_Stokes_m_s":
                2.0*(rho[i]-rho_f)*G*a[i]**2/(9.0*eta),
            "uniform_field_required_to_balance_weight_V_m": e_balance[i],
            "gravity_equilibrium_z_over_R_in_Local15_DC_quad": z_eq_R[i],
            "local15_RF_micromotion_screen": micromotion[i],
        } for i in range(2)],
        "ratios": {"mass2_over_mass1": mass_ratio,
                   "charge2_over_charge1": charge_ratio,
                   "(q2/m2)/(q1/m1)": alpha[1],
                   "lambda2_over_lambda1_from_same_air":
                       lambda_stokes[1] / lambda_stokes[0],
                   "Local15_lambda2_over_lambda1":
                       LOCAL15["lambda2_abstract"] / LOCAL15["lambda1_abstract"],
                   "dimensionless_mass2": mass_ratio,
                   "dimensionless_charge2": charge_ratio,
                   "dimensionless_Coulomb_C12": C12,
                   "dimensionless_Coulomb_acceleration_particle2": C12/mass_ratio},
        "field_and_electrodes": {
            "f1_Hz": PAIR["f1_Hz"], "f2_Hz": PAIR["f1_Hz"]*PAIR["frequency_ratio_nu"],
            "omega1_rad_s": omega1, "omega2_rad_s": omega2,
            "R_m": R, "D_m": D,
            "Local15_A_B_beta": LOCAL15,
            "leading_transfer_h_plus2": MFS["h_plus2"],
            "leading_transfer_h_minus4": MFS["h_minus4"],
            "U1_electrode_peak_V": U1, "U2_electrode_peak_V": U2,
            "V2_DC_ideal_quadrupole_coefficient_V_not_electrode_voltage": V2,
            "U2_over_U1": U2/U1,
            "rod_to_shield_gap_m": gap_rod_shield,
            "neighbor_rod_gap_m": gap_adjacent,
            "conservative_RF_surface_E_rod_family_1_V_m": E_rod_rf1,
            "conservative_RF_surface_E_rod_family_2_V_m": E_rod_rf2,
            "conservative_RF_surface_E_rod_sum_V_m": E_rod_rf1+E_rod_rf2,
            "conservative_RF_surface_E_shield_family_1_V_m": E_shield_rf1,
            "conservative_RF_surface_E_shield_family_2_V_m": E_shield_rf2,
            "conservative_RF_surface_E_shield_sum_V_m": E_shield_rf1+E_shield_rf2,
            "rod_shield_pd_Torr_cm": PAIR["air_pressure_Pa"]/TORR_PA*gap_rod_shield*100.0,
            "adjacent_rod_pd_Torr_cm": PAIR["air_pressure_Pa"]/TORR_PA*gap_adjacent*100.0,
            "rod_shield_parallel_plate_Paschen_V_screen_only": paschen_voltage(gap_rod_shield),
            "adjacent_rod_parallel_plate_Paschen_V_screen_only": paschen_voltage(gap_adjacent),
            "RF_electrode_difference_upper_bound_V":
                2.0*U1*math.sin(math.pi/24.0)+U2,
            "single_rod_to_ground_peak_upper_bound_V": U1+U2,
        },
        "gravity_and_coulomb": {
            "dimensionless_g_over_R_Omega1_squared": gravity_hat,
            "weight_N": [m[i]*G for i in range(2)],
            "Coulomb_force_at_separation_R_N":
                q[0]*q[1]/(4.0*math.pi*EPS0*R**2),
            "Coulomb_force_at_sphere_contact_N":
                q[0]*q[1]/(4.0*math.pi*EPS0*(a[0]+a[1])**2),
            "charge_at_3MV_m_uniform_sphere_reference_e":
                [4.0*math.pi*EPS0*a[i]**2*3.0e6/E_CHARGE for i in range(2)],
        },
        "caveats": [
            "The 3 MV/m comparison and Paschen voltage are reference screens, not breakdown criteria for this geometry.",
            "The MFS surface bounds use a conservative sum of independent family maxima; exact two-phase maxima require the full complex surface-field bases.",
            "The micromotion velocity estimate comes from the frozen Local15 linear response and must be replaced by velocity statistics from the new exact two-particle trajectories.",
            "A Cunningham factor is small at these Kn values; it is not multiplied into Oseen as though that product were a controlled composite theory.",
        ],
    }


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "results" / "dimensional_screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    data = build_screen()
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(output)
    print(json.dumps(data["ratios"], indent=2))


if __name__ == "__main__":
    main()
