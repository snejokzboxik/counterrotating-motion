import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_v13_3_atlas_products.py"
_SPEC = importlib.util.spec_from_file_location("v13_3_products", _PATH)
_PRODUCTS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PRODUCTS)


def test_physical_scaling_reproduces_v13_reference_drives():
    rows = _PRODUCTS.voltage_and_feasibility()
    row = next(r for r in rows if abs(r["alpha"]-.025)<1e-12
               and abs(r["beta"]-.175)<1e-12 and abs(r["eta"]-.05)<1e-12)
    assert np.isclose(row["U2_V"], 190.0, rtol=2e-12)
    assert np.isclose(row["U4_V"], 13.5448, rtol=2e-5)
    assert row["max_electrode_peak_abs_V"] < 250.0


def test_voltage_limit_marks_large_octupole_as_infeasible():
    rows = _PRODUCTS.voltage_and_feasibility()
    row = next(r for r in rows if abs(r["alpha"]-.025)<1e-12
               and abs(r["beta"]-.175)<1e-12 and abs(r["eta"]-.2)<1e-12)
    assert np.isclose(row["U4_V"], 54.1792, rtol=2e-5)
    assert row["max_electrode_peak_abs_V"] > 250.0
    assert not row["within_250V"]
