import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "estimate_v13_3_mass_fwhm.py"
SPEC = importlib.util.spec_from_file_location("estimate_v13_3_mass_fwhm", MODULE_PATH)
FWHM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FWHM)


def test_single_bounded_peak_is_resolved_on_dense_grid():
    x = np.linspace(-2.0, 2.0, 81)
    y = np.exp(-np.log(2.0) * x**2)
    peak, mu_peak, left, right, width, lobes, resolved, _ = FWHM.crossings(x, y)
    assert peak == 1.0
    assert abs(mu_peak) < 1e-14
    assert abs(left + 1.0) < 0.003
    assert abs(right - 1.0) < 0.003
    assert abs(width - 2.0) < 0.006
    assert lobes == 1
    assert resolved


def test_disconnected_half_height_bands_are_flagged_as_multilobed():
    x = np.arange(9, dtype=float)
    y = np.array([0.0, 0.6, 1.0, 0.4, 0.1, 0.6, 1.0, 0.6, 0.0])
    _, _, left, right, width, lobes, resolved, _ = FWHM.crossings(x, y)
    assert np.isfinite(left) and np.isfinite(right) and np.isfinite(width)
    assert lobes == 2
    assert not resolved


def test_unbounded_peak_does_not_receive_a_finite_fwhm():
    x = np.arange(5, dtype=float)
    y = np.array([0.1, 0.4, 0.8, 1.0, 0.7])
    _, _, left, right, width, lobes, resolved, _ = FWHM.crossings(x, y)
    assert np.isfinite(left)
    assert np.isnan(right)
    assert np.isnan(width)
    assert lobes == 1
    assert not resolved
