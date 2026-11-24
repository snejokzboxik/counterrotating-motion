"""Independent checks of the V20B-to-Reynolds coordinate transformation."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theory.nondimensionalization import (  # noqa: E402
    Trap, energy_and_angular_momentum, energy_and_angular_momentum_reynolds,
    from_reynolds_state, original_rhs, power_and_torque_balance,
    reynolds_rhs, to_reynolds_state,
)


def archived_rhs():
    """Compile only the separate V20B control RHS, avoiding its Numba runner imports."""
    code = ROOT / "research/v17c/v17c_20b/code"
    source = code / "dop853_independent.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "rhs_independent"]
    assert len(functions) == 1
    isolated = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"np": np, "math": math}
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["rhs_independent"]


def states():
    rng = np.random.default_rng(20261124)
    for _ in range(12):
        r1, r2 = rng.uniform(0.25, 0.75, 2)
        theta = rng.uniform(-np.pi, np.pi, 2)
        pos1 = np.asarray([r1 * np.cos(theta[0]), r1 * np.sin(theta[0]), rng.uniform(-0.7, 0.1)])
        pos2 = np.asarray([r2 * np.cos(theta[1]), r2 * np.sin(theta[1]), rng.uniform(-1.0, -0.3)])
        vel = rng.uniform(-0.06, 0.06, (2, 3))
        yield np.concatenate((pos1, vel[0], pos2, vel[1])), rng.uniform(0, 20), rng.uniform(0, 2*np.pi, 2)


def test_reynolds_map_and_archived_rhs():
    trap = Trap.refine12(ROOT)
    field = {"family": trap.family, "k": trap.modes, "coeff": trap.coeff}
    rhs_v20b = archived_rhs()
    for x, tau, phases in states():
        z = to_reynolds_state(x, trap)
        assert np.allclose(from_reynolds_state(z, trap), x, rtol=0, atol=2e-16)
        raw = original_rhs(tau, x, trap, phases)
        reference = rhs_v20b(tau, x, trap.p, phases, field, trap.qratio,
                             trap.re_star[0], trap.re_star[1], trap.gravity_z)
        assert np.allclose(raw, reference, rtol=4e-14, atol=4e-15)
        transformed = reynolds_rhs(tau, z, trap, phases).reshape(2, 6)
        assert np.allclose(transformed / trap.re_star[:, None], raw.reshape(2, 6),
                           rtol=5e-14, atol=5e-15)


def test_reynolds_speed_geometry_and_conservation_balance():
    trap = Trap.refine12(ROOT)
    for x, tau, phases in states():
        z = to_reynolds_state(x, trap)
        assert np.allclose(np.linalg.norm(z.reshape(2, 6)[:, 3:], axis=1),
                           trap.re_star * np.linalg.norm(x.reshape(2, 6)[:, 3:], axis=1))
        h1, l1 = energy_and_angular_momentum(x, trap, tau, phases)
        h2, l2 = energy_and_angular_momentum_reynolds(z, trap, tau, phases)
        assert np.allclose([h1, l1], [h2, l2], rtol=5e-14, atol=5e-15)
        f = original_rhs(tau, x, trap, phases)
        step = 2e-7
        hp, lp = energy_and_angular_momentum(x + step * f, trap, tau + step, phases)
        hm, lm = energy_and_angular_momentum(x - step * f, trap, tau - step, phases)
        power, torque = power_and_torque_balance(x, trap, tau, phases)
        assert abs((hp - hm) / (2 * step) - power) < 2e-7
        assert abs((lp - lm) / (2 * step) - torque) < 2e-8


def test_coulomb_torques_cancel_only_after_mass_weighting():
    trap = Trap.refine12(ROOT)
    s, c = trap.p[4], trap.p[5]
    for x, _, _ in states():
        y = x.reshape(2, 6)
        delta = y[0, :3] - y[1, :3]
        force_1 = c * delta / np.linalg.norm(delta)**3
        acceleration_2 = -s * force_1
        weighted = (np.cross(y[0, :3], force_1)[2]
                    + (1 / s) * np.cross(y[1, :3], acceleration_2)[2])
        assert abs(weighted) < 5e-16
