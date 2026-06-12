"""M4: 月球借力 — 解析公式 + 数值仿真双套实现与对比验证."""

import numpy as np
from numpy.linalg import norm

MU_MOON = 4.9048695e3
R_MOON = 1737.4
R_MOON_SOI = 6.6e4

from conic_patch import lunar_swingby


def _kepler_hyperbola(a, e, r_target):
    """Compute true anomaly nu on a hyperbola at distance r_target.

    r = a*(e^2-1)/(1+e*cos(nu)) → cos(nu) = (a*(e^2-1)/r - 1)/e

    Returns nu in radians (negative for incoming leg).
    """
    cos_nu = (a * (e**2 - 1) / r_target - 1) / e
    cos_nu = np.clip(cos_nu, -1.0 + 1e-14, 1.0 - 1e-14)
    nu = np.arccos(cos_nu)
    return -nu  # negative = approaching (incoming leg)


def _state_from_elements(a, e, nu, mu=MU_MOON):
    """Compute position and velocity in perifocal frame from orbital elements.

    Parameters
    ----------
    a : float — semi-major axis (> 0 for hyperbola)
    e : float — eccentricity (> 1 for hyperbola)
    nu : float — true anomaly (rad)

    Returns
    -------
    (r, v) : (3,) ndarray each, in perifocal frame
    """
    p = a * (e**2 - 1)        # semi-latus rectum
    r_mag = p / (1 + e * np.cos(nu))

    # Position in perifocal frame
    r_pf = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0])

    # Velocity in perifocal frame
    factor = np.sqrt(mu / p)
    v_pf = np.array([-factor * np.sin(nu),
                      factor * (e + np.cos(nu)),
                      0.0])

    return r_pf, v_pf


def numerical_swingby_trajectory(r0, v0, dt=5.0, max_steps=200000):
    """Numerically integrate a Moon-centered hyperbolic trajectory.

    Velocity-Verlet in the Moon-centered 2-body frame.
    Stops when spacecraft exits the Moon SOI.

    Returns (r_out, v_out) or None if impact.
    """
    r = r0.copy().astype(float)
    v = v0.copy().astype(float)

    for step in range(max_steps):
        r_norm = norm(r)
        if r_norm < R_MOON:
            return None

        # Velocity-Verlet
        a0 = -MU_MOON * r / r_norm**3
        r_new = r + v * dt + 0.5 * a0 * dt**2
        r_norm_new = norm(r_new)
        if r_norm_new < R_MOON:
            return None
        a1 = -MU_MOON * r_new / r_norm_new**3
        v_new = v + 0.5 * (a0 + a1) * dt

        r, v = r_new, v_new

        if norm(r) > R_MOON_SOI:
            return r, v

    return r, v


def swingby_numerical(v_inf, r_p, side='trailing'):
    """Full numerical lunar swingby with proper Keplerian initial conditions.

    Constructs initial state on the hyperbola at r = R_MOON_SOI and
    integrates through periapsis to SOI exit.
    """
    if r_p < R_MOON + 100:
        return {'delta': None, 'v_out': None, 'delta_v': None,
                'impact': True, 'error': f'r_p < {R_MOON+100:.0f} km'}

    # Orbital elements
    a = MU_MOON / v_inf**2
    e = 1 + r_p / a

    # True anomaly at SOI boundary (incoming)
    nu_in = _kepler_hyperbola(a, e, R_MOON_SOI)

    # State in perifocal frame
    r_pf, v_pf = _state_from_elements(a, e, nu_in)

    # Flip y for leading vs trailing
    if side == 'leading':
        r_pf[1] *= -1
        v_pf[1] *= -1

    result = numerical_swingby_trajectory(r_pf, v_pf)

    if result is None:
        return {'delta': None, 'v_out': None, 'delta_v': None, 'impact': True}

    r_out, v_out = result
    v_in = v_pf  # initial velocity in perifocal frame
    v_out_mag = norm(v_out)
    v_in_mag = norm(v_in)

    cos_delta = np.dot(v_in, v_out) / (v_in_mag * v_out_mag)
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    delta = np.arccos(cos_delta)

    return {
        'delta': delta,
        'delta_deg': np.degrees(delta),
        'v_out': v_out_mag,
        'v_conservation': abs(v_out_mag - v_in_mag) / v_in_mag,
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

    if numeric['impact']:
        print(f'  Numerical: IMPACT!')
        return False

    print(f'  Numerical: delta = {numeric["delta_deg"]:.3f} deg  '
          f'v_out/v_inf = {numeric["v_conservation"]:.6f}')

    delta_err = abs(numeric['delta'] - analytic['delta'])
    tol = max(np.radians(1.0), 0.05 * analytic['delta'])  # 1° or 5% relative
    ok = delta_err < tol and numeric['v_conservation'] < 0.01

    print(f'  Delta mismatch = {np.degrees(delta_err):.4f} deg  '
          f'(tol {np.degrees(tol):.2f} deg)  '
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
