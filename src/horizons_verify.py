"""M3: JPL Horizons 历表对照.

N-体传播 Sun-Earth-Moon 并与 JPL Horizons 历表对比（或分析历表作为备选）。
输出每日位置与速度残差，要求所有天体位置残差 ≤ 6000 km.
"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta

from nbody import (
    acceleration_sem, velocity_verlet_step, system_energy,
    earth_moon_analytic_state, AU, DAY, MU_SUN, MU_EARTH, MU_MOON,
    test_two_body_circular,
)


def _julian_date(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (day + (153 * m + 2) // 5 + 365 * y
            + y // 4 - y // 100 + y // 400 - 32045)


def get_horizons_state(date_str):
    """Get Earth and Moon state from JPL Horizons for a given date.

    Returns (earth_pos, earth_vel, moon_pos, moon_vel) or None.
    """
    try:
        from astroquery.jplhorizons import Horizons

        epochs = {'start': date_str, 'stop': date_str, 'step': '1d'}

        obj_e = Horizons(id='399', location='@10', epochs=epochs)
        vec_e = obj_e.vectors()
        ep = np.array([float(vec_e['x'][0]), float(vec_e['y'][0]),
                       float(vec_e['z'][0])]) * AU
        ev = np.array([float(vec_e['vx'][0]), float(vec_e['vy'][0]),
                       float(vec_e['vz'][0])]) * AU / DAY

        obj_m = Horizons(id='301', location='@10', epochs=epochs)
        vec_m = obj_m.vectors()
        mp = np.array([float(vec_m['x'][0]), float(vec_m['y'][0]),
                       float(vec_m['z'][0])]) * AU
        mv = np.array([float(vec_m['vx'][0]), float(vec_m['vy'][0]),
                       float(vec_m['vz'][0])]) * AU / DAY

        return ep, ev, mp, mv
    except Exception:
        return None


def build_initial_state(earth_p, earth_v, moon_p, moon_v):
    """Build 4-body state vector (rocket at Earth)."""
    y0 = np.zeros(24)
    y0[0:3] = [0.0, 0.0, 0.0]
    y0[3:6] = [0.0, 0.0, 0.0]
    y0[6:9] = earth_p
    y0[9:12] = earth_v
    y0[12:15] = moon_p
    y0[15:18] = moon_v
    y0[18:21] = earth_p.copy()
    y0[21:24] = earth_v.copy()
    return y0


def _get_earth_moon_state(date_str):
    """Get Earth/Moon state, Horizons preferred, analytic fallback."""
    state = get_horizons_state(date_str)
    if state is not None:
        return state
    parts = date_str.split('-')
    yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
    jd = _julian_date(yr, mo, dy)
    return earth_moon_analytic_state(jd)


def run_verification(start_date='2026-06-01', n_days=30, dt=3600):
    """Compare N-body propagation vs reference ephemeris over n_days.

    Uses JPL Horizons if available, otherwise analytic ephemeris.
    The comparison is: propagate initial state 1 day → compare with
    reference state at the next day.
    """
    use_jpl = get_horizons_state(start_date) is not None
    source = 'JPL Horizons' if use_jpl else 'analytic ephemeris'
    print(f'Horizons Ephemeris Verification')
    print(f'  Source:  {source}')
    print(f'  Period:  {start_date} + {n_days} days')
    print(f'  Step:    {dt} s')
    print(f'  Target:  position residual ≤ 6000 km\n')

    start = datetime.strptime(start_date, '%Y-%m-%d')
    steps_per_day = int(DAY / dt)

    max_earth_err = 0.0
    max_moon_err = 0.0
    failures = 0

    for d in range(n_days):
        date = start + timedelta(days=d)
        date_str = date.strftime('%Y-%m-%d')
        next_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')

        ep, ev, mp, mv = _get_earth_moon_state(date_str)
        y0 = build_initial_state(ep, ev, mp, mv)

        # Propagate 1 day
        y = y0.copy()
        try:
            for _ in range(steps_per_day):
                y = velocity_verlet_step(y, dt)
        except RuntimeError:
            failures += 1
            continue

        # Reference state at next day
        ep_ref, _, mp_ref, _ = _get_earth_moon_state(next_str)

        earth_err = norm(y[6:9] - ep_ref)
        moon_err = norm(y[12:15] - mp_ref)

        max_earth_err = max(max_earth_err, earth_err)
        max_moon_err = max(max_moon_err, moon_err)

        if d % 10 == 0 or d == n_days - 1:
            E_drift = abs(system_energy(y) / system_energy(y0) - 1)
            print(f'  Day {d:3d}: Earth err = {earth_err:.0f} km  '
                  f'Moon err = {moon_err:.0f} km  '
                  f'E drift = {E_drift:.2e}')

    print(f'\n=== Summary ===')
    print(f'  Max Earth pos error: {max_earth_err:.0f} km  '
          f'({"PASS" if max_earth_err <= 6000 else "FAIL"})')
    print(f'  Max Moon  pos error: {max_moon_err:.0f} km  '
          f'({"PASS" if max_moon_err <= 6000 else "FAIL"})')
    print(f'  Integration failures: {failures}')

    ok = max_earth_err <= 6000 and max_moon_err <= 6000 and failures == 0
    print(f'  Overall: {"PASS" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    import sys
    # Quick mode for --test
    if '--test' in sys.argv:
        ok = run_verification(n_days=3, dt=7200)
        assert ok
    else:
        run_verification()
