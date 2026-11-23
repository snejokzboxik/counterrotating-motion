from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "research" / "v17c" / "v17c_20a" / "results"
FIG = ROOT / "research" / "v17c" / "v17c_20a" / "figures"
TARGETS = (2000, 5000, 10000, 20000, 50000)


def read_result(candidate: str, periods: int, source_period: int, steps=32, qr=20):
    run_id = f"{candidate}_{source_period}p_seed__100p_transient__{periods}p_qr__s{steps}_q{qr}"
    path = OUT / "lyapunov" / run_id / "result.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fit_models(T: np.ndarray, lam: np.ndarray):
    x = 1.0 / T
    c0 = float(np.dot(x, lam) / np.dot(x, x))
    pred0 = c0 * x
    X = np.column_stack((np.ones_like(x), x))
    coef, *_ = np.linalg.lstsq(X, lam, rcond=None)
    pred1 = X @ coef

    def stats(pred, k):
        rss = max(float(np.sum((lam - pred) ** 2)), 1e-300)
        n = len(T)
        return {"rss": rss, "aic_no_constant": float(n * math.log(rss / n) + 2 * k),
                "max_abs_residual": float(np.max(np.abs(lam - pred)))}

    return {
        "c_over_T": {"c": c0, **stats(pred0, 1)},
        "lambda_inf_plus_c_over_T": {"lambda_inf": float(coef[0]), "c": float(coef[1]), **stats(pred1, 2)},
        "interpretation": "Finite-window regression only; model ranking over four horizons is descriptive, not an asymptotic proof.",
    }


def main():
    all_rows = []
    spectra_by_candidate = {}
    fit_output = {}
    cases = [("candidate52", 50000), ("local15", 50000), ("refine12_sobol17", 240000)]
    for candidate, source in cases:
        rows = []
        for horizon in TARGETS:
            periods = 50000 if horizon == 50000 and candidate in ("candidate52", "local15") else 20000
            if candidate == "refine12_sobol17" and horizon == 50000:
                periods = 50000
            result = read_result(candidate, periods, source)
            if result is None:
                continue
            row = next((r for r in result["rows"] if abs(r["qr_periods_after_transient"] - horizon) < 1e-9), None)
            if row is None:
                continue
            spectrum = np.asarray(row["spectrum_per_tau_descending"], dtype=float)
            rows.append({"horizon_periods": horizon, "spectrum_per_tau_descending": spectrum.tolist(),
                         "volume_balance_abs_error_per_tau": row["volume_balance_abs_error_per_tau"],
                         "sum_exponents_per_tau": row["sum_exponents_per_tau"],
                         "integrated_divergence_per_tau": row["integrated_divergence_per_tau"]})
        # Use identical 2k, 5k, 10k, 20k windows for the two finite-window fits.
        base_rows = [r for r in rows if r["horizon_periods"] <= 20000]
        T = np.asarray([r["horizon_periods"] for r in base_rows], dtype=float)
        ex = np.asarray([r["spectrum_per_tau_descending"] for r in base_rows], dtype=float)
        T_all = np.asarray([r["horizon_periods"] for r in rows], dtype=float)
        ex_all = np.asarray([r["spectrum_per_tau_descending"] for r in rows], dtype=float)
        fit_output[candidate] = {
            "horizons_periods": T.astype(int).tolist(),
            "exponents_at_horizons_per_tau": ex.tolist(),
            "per_exponent": [fit_models(T, ex[:, i]) for i in range(ex.shape[1])],
            "top_two_models": [fit_models(T, ex[:, i]) for i in range(min(2, ex.shape[1]))],
            "all_horizons_model_comparison_including_50k": {
                "horizons_periods": T_all.astype(int).tolist(),
                "top_two_models": [fit_models(T_all, ex_all[:, i]) for i in range(min(2, ex_all.shape[1]))],
                "all_exponent_models": [fit_models(T_all, ex_all[:, i]) for i in range(ex_all.shape[1])],
            },
            "divergence_check": [{k: r[k] for k in ("horizon_periods", "sum_exponents_per_tau", "integrated_divergence_per_tau", "volume_balance_abs_error_per_tau")} for r in rows],
            "extra_50k": next((r["spectrum_per_tau_descending"] for r in rows if r["horizon_periods"] == 50000), None),
        }
        spectra_by_candidate[candidate] = rows

    # Step, QR interval and independent-start checks for refine12 at 2,000 periods.
    sensitivity = {}
    sensitivity_specs = [
        ("step64_qr20", "refine12_sobol17", 240000, 2000, 64, 20),
        ("step32_qr5", "refine12_sobol17", 240000, 2000, 32, 5),
        ("late_start120k", "refine12_sobol17", 120000, 2000, 32, 20),
    ]
    for key, candidate, source, periods, steps, qr in sensitivity_specs:
        r = read_result(candidate, periods, source, steps=steps, qr=qr)
        if r is not None:
            sensitivity[key] = {"spectrum_per_tau_descending": r["lyapunov_spectrum_per_tau_descending"],
                                "volume_balance_abs_error_per_tau": r["volume_balance_abs_error_per_tau"],
                                "run_id": r["run_id"]}

    summary = {
        "source_commit": "d1bf252649e174166becf464269318b5f97acd5d",
        "method": "Conditional 12D mechanical QR, RK4 tangent and base integration; continuous RF phases reconstructed from archived absolute period; 100p discarded transition; conditional tangent phase fixed.",
        "fits": fit_output,
        "numerical_sensitivity": sensitivity,
        "caveats": ["The slopes are finite-horizon conditional exponents, not proofs of existence or attraction of a torus.",
                    "A near-zero exponent may represent a neutral orbital phase, very weak contraction, or unresolved transient; direct set-attraction tests remain necessary.",
                    "The volume identity is tested against an independently accumulated analytic divergence along RK4 stages; it does not validate the full physical model."],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "lyapunov_horizon_fits.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "lyapunov_horizon_fits.csv").write_text(
        "candidate,horizon_periods,lambda1_per_tau,lambda2_per_tau,lambda_sum_per_tau,divergence_per_tau,volume_error_per_tau\n" +
        "".join(f"{candidate},{r['horizon_periods']},{r['spectrum_per_tau_descending'][0]:.12g},{r['spectrum_per_tau_descending'][1]:.12g},{r['sum_exponents_per_tau']:.12g},{r['integrated_divergence_per_tau']:.12g},{r['volume_balance_abs_error_per_tau']:.12g}\n" for candidate, rows in spectra_by_candidate.items() for r in rows),
        encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib unavailable; numeric outputs created: {exc}")
        return

    FIG.mkdir(parents=True, exist_ok=True)
    colors = {"candidate52": "#0072B2", "local15": "#D55E00", "refine12_sobol17": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), constrained_layout=True)
    for candidate, rows in spectra_by_candidate.items():
        c = colors[candidate]
        for idx, marker, label in ((0, "o", r"$\lambda_1$"), (1, "s", r"$\lambda_2$")):
            good = [r for r in rows if r["horizon_periods"] <= 20000]
            if not good:
                continue
            x = np.asarray([r["horizon_periods"] for r in good], dtype=float)
            y = np.asarray([r["spectrum_per_tau_descending"][idx] for r in good])
            axes[0].scatter(x, y, color=c, marker=marker, s=38,
                            label=f"{candidate}, {label}")
        extra = [r for r in rows if r["horizon_periods"] == 50000]
        if extra:
            axes[0].scatter([50000, 50000], extra[0]["spectrum_per_tau_descending"][:2],
                            color=c, marker="D", s=48, label=f"{candidate}, 50k")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("QR horizon after transient (RF periods)")
    axes[0].set_ylabel(r"Conditional mechanical exponent $\lambda$ ($\tau^{-1}$)")
    axes[0].set_title("Near-neutral mechanical directions")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)

    for candidate, rows in spectra_by_candidate.items():
        good = [r for r in rows if r["horizon_periods"] <= 20000]
        if not good:
            continue
        h = np.asarray([r["horizon_periods"] for r in good], dtype=float)
        err = np.asarray([r["volume_balance_abs_error_per_tau"] for r in good])
        axes[1].scatter(h, err, color=colors[candidate], label=candidate, s=32)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("QR horizon after transient (RF periods)")
    axes[1].set_ylabel(r"$|\sum_i\lambda_i-\langle\mathrm{tr}J\rangle|$ ($\tau^{-1}$)")
    axes[1].set_title("Liouville volume-balance check")
    axes[1].grid(True, alpha=0.2, which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("V17C.20A — long-horizon conditional QR audit", fontsize=12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"FIG01_lyapunov_long_horizon.{ext}", dpi=600 if ext == "png" else None)
    plt.close(fig)
    print(json.dumps({"summary": str(OUT / "lyapunov_horizon_fits.json"),
                      "figure": str(FIG / "FIG01_lyapunov_long_horizon.png"),
                      "sensitivity": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
