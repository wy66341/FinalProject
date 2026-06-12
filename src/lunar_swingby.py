"""M4: 月球借力 — 解析公式 + 数值仿真双套实现与对比验证."""

import numpy as np
from numpy.linalg import norm

MU_MOON = 4.9048695e3
R_MOON = 1737.4
R_MOON_SOI = 6.6e4

from conic_patch import lunar_swingby


def numerical_swingby_trajectory(r0, v0, dt=60.0, max_steps=20000):
    """Numerically integrate a Moon-centered hyperbolic trajectory.

    Uses Velocity-Verlet in the Moon-centered 2-body frame.
    Stops when the spacecraft exits the Moon's SOI.

    Parameters
    ----------
    r0 : (3,) ndarray — initial Moon-centered position (km)
    v0 : (3,) ndarray — initial Moon-centered velocity (km/s)
    dt : float — step size (s)
    max_steps : int

    Returns
    -------
    (r_out, v_out) or None if impact
    """
    r = r0.copy().astype(float)
    v = v0.copy().astype(float)

    for _ in range(max_steps):
        r_norm = norm(r)
        if r_norm < R_MOON:
            return None  # impact

        # Velocity-Verlet
        a0 = -MU_MOON * r / r_norm**3
        r_new = r + v * dt + 0.5 * a0 * dt**2
        a1 = -MU_MOON * r_new / norm(r_new)**3
        v_new = v + 0.5 * (a0 + a1) * dt

        r, v = r_new, v_new

        if norm(r) > R_MOON_SOI:
            return r, v

    return r, v  # didn't exit SOI within max_steps


def swingby_numerical(v_inf, r_p, side='trailing', b_offset=R_MOON_SOI):
    """Full numerical lunar swingby simulation with SOI boundary setup.

    Constructs initial conditions at the Moon's SOI boundary and
    numerically integrates the hyperbolic passage.

    Parameters
    ----------
    v_inf : float — v_infinity magnitude (km/s)
    r_p : float — pericynthion distance (km)
    side : str
    b_offset : float — distance from Moon center to start integration (km)

    Returns
    -------
    dict with delta, v_out, delta_v, impact
    """
    if r_p < R_MOON + 100:
        return {'delta': None, 'v_out': None, 'delta_v': None, 'impact': True,
                'error': f'r_p < {R_MOON+100:.0f} km'}

    # Impact parameter from conservation of angular momentum
    # r_p * v_p = b * v_inf, and v_p = sqrt(v_inf^2 + 2*mu/r_p)
    v_p = np.sqrt(v_inf**2 + 2 * MU_MOON / r_p)
    b = r_p * v_p / v_inf

    if b > b_offset:
        # Adjust starting distance
        b_offset = max(b_offset, b * 1.5)

    # Setup at SOI boundary: r = (-sqrt(b_offset^2 - b^2), b, 0)
    x_start = -np.sqrt(max(b_offset**2 - b**2, 0))
    y_start = b if side == 'trailing' else -b

    r0 = np.array([x_start, y_start, 0.0])
    v0 = np.array([v_inf, 0.0, 0.0])

    result = numerical_swingby_trajectory(r0, v0)

    if result is None:
        return {'delta': None, 'v_out': None, 'delta_v': None, 'impact': True}

    r_out, v_out = result
    v_out_mag = norm(v_out)

    # Deflection angle
    cos_delta = np.dot(v0, v_out) / (norm(v0) * v_out_mag)
    cos_delta = np.clip(cos_delta, -1, 1)
    delta = np.arccos(cos_delta)

    return {
        'delta': delta,
        'delta_deg': np.degrees(delta),
        'v_out': v_out_mag,
        'v_conservation': abs(v_out_mag - v_inf) / v_inf,
        'delta_v': 0.0,
        'impact': False,
    }


def compare_analytic_numerical(v_inf, r_p, side='trailing'):
    """Side-by-side comparison of analytic vs numerical swingby."""
    analytic = lunar_swingby(v_inf, r_p, side)
    numeric = swingby_numerical(v_inf, r_p, side)

    print(f'v_inf={v_inf:.1f} km/s  r_p={r_p:.0f} km  side={side}')
    print(f'  Analytic:  delta = {analytic["delta_deg"]:.3f} deg  '
          f'e = {analytic["e"]:.3f}')
    print(f'  Numerical: delta = {numeric["delta_deg"]:.3f} deg  '
          f'v_out/v_inf = {numeric["v_conservation"]:.6f}')

    if numeric['impact']:
        print(f'  IMPACT!')
        return False

    delta_err = abs(numeric['delta'] - analytic['delta'])
    ok = delta_err < 0.005  # ~0.3 deg
    print(f'  Delta mismatch = {np.degrees(delta_err):.4f} deg  '
          f'{"PASS" if ok else "FAIL"}')
    print()
    return ok


if __name__ == '__main__':
    print('=== 月球借力 解析 vs 数值 ===\n')
    all_ok = True
    for v_inf in [1.0, 2.0, 3.0]:
        for r_p in [2000.0, 5000.0, 15000.0]:
            for side in ['trailing', 'leading']:
                if not compare_analytic_numerical(v_inf, r_p, side):
                    all_ok = False
    print(f'Overall: {"ALL PASS" if all_ok else "SOME FAILED"}')
