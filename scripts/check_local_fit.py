#!/usr/bin/env python3
"""Self-contained checks for the local measured-line registration solver.

Run with: PYTHONPATH=. python3 scripts/check_local_fit.py
The checks assert the physical promises of the local fit:
  * the four-corner frame never moves,
  * interior measured lines converge to their ruler readings,
  * frame-edge lines never drive the deformation,
  * the inverse warp round-trips,
  * blank and frame-only inputs are rejected.
"""
import sys

import numpy as np

from vitrail.contracts import Problem
from vitrail.localfit import LocalLine, fit_displacement

WIDTH_MM, HEIGHT_MM = 200.0, 150.0


def approx(actual, expected, tolerance):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected:.3f} +/- {tolerance}, got {actual:.3f}")


def main():
    frame_line = LocalLine("top edge", (.1, .02), (.9, .02), 160.0, True)
    interior = LocalLine("squeezed", (.34, .47), (.66, .47), 60.16, False)
    warp, inverse, ux, uy, _ = fit_displacement(
        [interior, frame_line], WIDTH_MM, HEIGHT_MM, .35)

    # 1. Boundary nodes stay exactly on the four-corner registration.
    boundary = max(abs(ux[0]).max(), abs(ux[-1]).max(), abs(ux[:, 0]).max(), abs(ux[:, -1]).max(),
                   abs(uy[0]).max(), abs(uy[-1]).max(), abs(uy[:, 0]).max(), abs(uy[:, -1]).max())
    if boundary > 1e-9:
        raise AssertionError(f"frame moved by {boundary}")
    corners = warp(((0, 0), (1, 0), (1, 1), (0, 1)))
    if not np.allclose(corners, ((0, 0), (1, 0), (1, 1), (0, 1))):
        raise AssertionError("panel corners were displaced")

    # 2. The interior line approaches its ruler reading (sub-millimetre residual).
    corrected_a, corrected_b = warp((interior.a, interior.b))
    corrected_len = float(np.linalg.norm((corrected_b - corrected_a) * [WIDTH_MM, HEIGHT_MM]))
    approx(corrected_len, interior.measured_mm, .6)

    # 3. The line centre is held still (symmetric squeeze, no translation drift).
    centre = (np.asarray(interior.a) + np.asarray(interior.b)) / 2
    if np.linalg.norm(warp((centre,))[0] - centre) > 1e-6:
        raise AssertionError("the measured-line centre drifted")

    # 4. Inverse warp round-trips to within a fraction of a millimetre.
    probe = np.array([(.35, .5), (.65, .5), (.2, .3), (.8, .7)])
    roundtrip_error = np.abs(inverse(warp(probe)) - probe).max()
    if roundtrip_error * WIDTH_MM > 0.2:
        raise AssertionError(f"inverse warp error {roundtrip_error} too large")

    # 5. A frame-only request is rejected.
    try:
        fit_displacement([frame_line], WIDTH_MM, HEIGHT_MM, .35)
    except Problem:
        pass
    else:
        raise AssertionError("frame-only fit should be rejected")

    # 6. A blank ruler reading is rejected upstream through the measured value guard.
    blank = LocalLine("blank", (.34, .47), (.66, .47), None, False)
    try:
        fit_displacement([blank], WIDTH_MM, HEIGHT_MM, .35)
    except TypeError:
        pass
    else:
        raise AssertionError("blank reading should not reach the solver")

    print("local-fit solver checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
