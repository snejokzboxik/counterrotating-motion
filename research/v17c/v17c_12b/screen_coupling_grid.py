"""Discovery scan including physical charge/C variation around the V10 shape."""
from __future__ import annotations

import screen_parameter_grid as engine

engine.RESULTS = engine.HERE / "results" / "symmetric_coupling_grid_corrected"
engine.RESULTS.mkdir(parents=True, exist_ok=True)
engine.N_START = 32
engine.SEED = 20261221


def coupling_configs() -> list[dict]:
    base = [0.03492413282400757, 0.15631667332941065,
            3.2196773638017477, 0.015388559140477825,
            1.0, 9.747265456858355e-6,
            0.24575362874758605, 0.24575362874758605]
    amplitude_pairs = (
        (0.0175, 0.07816),
        (0.03492413282400757, 0.15631667332941065),
        (0.0700, 0.07816),
        (0.0175, 0.23447),
    )
    nus = (2.5, 3.2196773638017477, 4.0)
    couplings = (9.747265456858355e-6, 2.0e-5, 3.5e-5)
    rows = []
    for A, B in amplitude_pairs:
        for nu in nus:
            for C in couplings:
                p = [A, B, nu, base[3], 1.0, C, base[6], base[6]]
                cid = f"A{A:.4f}_B{B:.4f}_nu{nu:.3f}_C{C:.3g}"
                rows.append({"id": cid, "p": p, "qratio": 1.0})
    return rows


engine.make_configs = coupling_configs


if __name__ == "__main__":
    engine.main()
