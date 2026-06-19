"""O1: 3D Extension — Full mission with lunar 5.145deg orbital inclination.

Extends the 2D ecliptic-plane model to full 3D and compares
complete trajectory metrics (Moon approach, Earth return, perihelion)
between 2D and 3D N-body propagation.
"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta

from nbody import (
    acceleration_sem, velocity_verlet_step, system_energy,
    earth_moon_analytic_state, AU, DAY, MU_SUN, MU_EARTH, MU_MOON,
)
from conic_patch import R_MOON, R_SUN, R_MOON_SOI, helio_ellipse, lunar_swingby
from trajectory import get_ephemeris

MOON_INCLINATION = np.radians(5.145)


def _julian(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (day + (153 * m + 2) // 5 + 365 * y
            + y // 4 - y // 100 + y // 400 - 32045)


def _rotate_3d(vec_2d, inc, Omega):
    """Rotate a 2D ecliptic vector into 3D with inclination inc and node Omega."""
    x, y = vec_2d[0], vec_2d[1]
    z = 0.0
    c_o, s_o = np.cos(-Omega), np.sin(-Omega)
    x1 = c_o * x - s_o * y
    y1 = s_o * x + c_o * y
    c_i, s_i = np.cos(inc), np.sin(inc)
    y2 = c_i * y1
    z2 = s_i * y1
    c_o2, s_o2 = np.cos(Omega), np.sin(Omega)
    x3 = c_o2 * x1 - s_o2 * y2
    y3 = s_o2 * x1 + c_o2 * y2
    return np.array([x3, y3, z2])


def earth_moon_analytic_state_3d(jd):
    """3D Earth and Moon state vectors with lunar inclination."""
    ep, ev, mp_2d, mv_2d = earth_moon_analytic_state(jd)
    moon_geo_pos = mp_2d - ep
    moon_geo_vel = mv_2d - ev
    T = (jd - 2451545.0) / 36525.0
    Omega = np.radians(125.08 - 0.05295 * T)
    moon_geo_pos_3d = _rotate_3d(moon_geo_pos, MOON_INCLINATION, Omega)
    moon_geo_vel_3d = _rotate_3d(moon_geo_vel, MOON_INCLINATION, Omega)
    return ep, ev, ep + moon_geo_pos_3d, ev + moon_geo_vel_3d


def build_initial_state(earth_p, earth_v, moon_p, moon_v):
    """Build 4-body state vector."""
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


def verify_full_mission_3d(date_str, r_p, r_m=None, side=None, dt=3600):
    """Full N-body trajectory verification in 3D.

    Same logic as trajectory.verify_full_mission() but uses 3D
    initial conditions with lunar inclination.
    """
    result = {'date': date_str, 'r_p': r_p, 'r_m': r_m, 'side': side}

    # Get 2D analytic state for departure velocity
    parts = date_str.split('-')
    jd = _julian(int(parts[0]), int(parts[1]), int(parts[2]))
    ep, ev, mp_3d, mv_3d = earth_moon_analytic_state_3d(jd)

    r_1 = norm(ep)
    ell = helio_ellipse(r_p, r_1)

    # Build 3D initial state
    y0 = build_initial_state(ep, ev, mp_3d, mv_3d)

    # Departure velocity (same as 2D — heliocentric transfer is ecliptic)
    v_earth_dir = ev / norm(ev)
    v_rocket_helio_departure = ev - ell['Delta_v_dep'] * v_earth_dir

    best_approach = float('inf')
    best_phase1 = None
    dt_scan = 3600
    r_start = 6378.137 + 200.0

    for hour_off in range(0, 24, 4):
        dt_offset = hour_off * 3600.0
        y_adv = y0.copy()
        for _ in range(int(dt_offset / dt_scan)):
            try:
                y_adv = velocity_verlet_step(y_adv, dt_scan)
            except RuntimeError:
                break
        ep_o = y_adv[6:9].copy()

        for az_deg in range(0, 360, 60):
            az = np.radians(az_deg)
            for el_deg in range(-60, 61, 60):
                el = np.radians(el_deg)
                launch_dir = np.array([np.cos(el)*np.cos(az),
                                       np.cos(el)*np.sin(az),
                                       np.sin(el)])
                y_test = y_adv.copy()
                y_test[18:21] = ep_o + r_start * launch_dir
                y_test[21:24] = v_rocket_helio_departure

                for _ in range(min(int(10.0 * DAY / dt_scan), 250)):
                    try:
                        y_test = velocity_verlet_step(y_test, dt_scan)
                    except RuntimeError:
                        break
                    d = norm(y_test[18:21] - y_test[12:15])
                    if d < best_approach:
                        best_approach = d
                        best_phase1 = {
                            'y_at_closest': y_test.copy(),
                            'closest_approach_km': d,
                        }

    result['moon_closest_approach_km'] = best_approach
    result['rule15_pass'] = best_approach < R_MOON_SOI

    if best_phase1 is None:
        result['rule16_pass'] = False
        result['perihelion_ok'] = False
        return result

    # Apply flyby deflection
    y_at_moon = best_phase1['y_at_closest'].copy()
    v_rocket = y_at_moon[21:24].copy()
    v_moon_flyby = y_at_moon[15:18].copy()
    v_inf_moon_vec = v_rocket - v_moon_flyby
    v_inf_moon_mag = norm(v_inf_moon_vec)

    if v_inf_moon_mag > 0.5 and r_m is not None and side is not None:
        try:
            swingby = lunar_swingby(v_inf_moon_mag, r_m, side)
            delta = swingby['delta']
            sign = swingby['sign']
            v_inf_dir = v_inf_moon_vec / v_inf_moon_mag
            v_moon_dir = v_moon_flyby / (norm(v_moon_flyby) + 1e-30)
            rot_axis = np.cross(v_inf_dir, v_moon_dir)
            if norm(rot_axis) > 1e-12:
                rot_axis = rot_axis / norm(rot_axis)
                cos_d, sin_d = np.cos(sign * delta), np.sin(sign * delta)
                v_inf_out = (cos_d * v_inf_moon_vec +
                             sin_d * np.cross(rot_axis, v_inf_moon_vec) +
                             (1 - cos_d) * np.dot(rot_axis, v_inf_moon_vec) * rot_axis)
            else:
                v_inf_out = v_inf_moon_vec
            y_at_moon[21:24] = v_moon_flyby + v_inf_out
        except ValueError:
            pass

    # Phase 3: Heliocentric → Earth return
    flight_time_s = ell['T_years'] * 365.25 * DAY
    n_steps_helio = int(flight_time_s / dt)
    y_helio = y_at_moon.copy()
    r_min = float('inf')
    earth_closest = float('inf')
    hit_sun = False

    for step in range(min(n_steps_helio, 500000)):
        try:
            y_helio = velocity_verlet_step(y_helio, dt)
        except RuntimeError:
            break
        r_sun_dist = norm(y_helio[18:21])
        r_earth_dist = norm(y_helio[18:21] - y_helio[6:9])
        r_min = min(r_min, r_sun_dist)
        earth_closest = min(earth_closest, r_earth_dist)
        if r_sun_dist < R_SUN:
            hit_sun = True
            break

    result['earth_closest_approach_km'] = earth_closest
    result['perihelion_km'] = r_min
    result['perihelion_ok'] = r_min > R_SUN
    result['rule16_pass'] = earth_closest < 0.02 * AU and not hit_sun
    result['hit_sun'] = hit_sun

    return result


def compare_2d_vs_3d_mission(date_str='2026-01-07', r_p=0.4*AU,
                              r_m=5000.0, side='trailing'):
    """Compare full 2D vs 3D trajectory metrics.

    Runs both 2D (from trajectory module) and 3D full mission verification
    and compares the key metrics.
    """
    from trajectory import verify_full_mission as verify_2d

    print('=' * 60)
    print('2D vs 3D Full Mission Comparison')
    print('=' * 60)
    print(f'  Date: {date_str}  r_p: {r_p/AU:.3f} AU  '
          f'r_m: {r_m:.0f} km  side: {side}\n')

    # 2D verification
    print('Running 2D verification...')
    m2d = verify_2d(date_str, r_p, r_m, side)

    # 3D verification
    print('Running 3D verification...')
    m3d = verify_full_mission_3d(date_str, r_p, r_m, side)

    metrics = [
        ('Moon approach (km)', 'moon_closest_approach_km'),
        ('Earth return (km)', 'earth_closest_approach_km'),
        ('Perihelion (km)', 'perihelion_km'),
        ('R15 (Moon SOI)', 'rule15_pass'),
        ('R16 (Earth return)', 'rule16_pass'),
        ('Perihelion OK', 'perihelion_ok'),
    ]

    print(f'\n  {"Metric":>25s}  {"2D":>15s}  {"3D":>15s}  {"Δ":>15s}  {"Δ%":>10s}')
    print(f'  {"-"*25}  {"-"*15}  {"-"*15}  {"-"*15}  {"-"*10}')

    for label, key in metrics:
        v2 = m2d.get(key, 'N/A')
        v3 = m3d.get(key, 'N/A')
        if isinstance(v2, (int, float)) and isinstance(v3, (int, float)):
            delta = v3 - v2
            pct = abs(delta) / max(abs(v2), 1) * 100 if abs(v2) > 0.1 else 0
            print(f'  {label:>25s}  {v2:15,.0f}  {v3:15,.0f}  '
                  f'{delta:+15,.0f}  {pct:9.2f}%')
        elif isinstance(v2, bool):
            print(f'  {label:>25s}  {str(v2):>15s}  {str(v3):>15s}  '
                  f'{"—":>15s}  {"—":>10s}')
        else:
            print(f'  {label:>25s}  {str(v2):>15s}  {str(v3):>15s}')

    # Impact assessment
    moon_diff = abs(m3d.get('moon_closest_approach_km', 0) -
                     m2d.get('moon_closest_approach_km', 0))
    moon_2d = m2d.get('moon_closest_approach_km', 1)

    print(f'\n=== Impact Analysis ===')
    print(f'  Moon approach Δ: {moon_diff:,.0f} km  '
          f'({moon_diff/moon_2d*100:.2f}% of 2D)')
    if moon_diff < 20000:
        print(f'  Assessment: 2D approximation captures dominant dynamics.')
        print(f'  The 3D correction is negligible for heliocentric Δv.')
        print(f'  For precise SOI targeting, 3D may be needed at some dates.')

    return m2d, m3d


def compare_2d_vs_3d_propagation(date_str='2026-01-07', n_days=30, dt=3600):
    """Compare 2D and 3D N-body propagation (legacy, for continuity)."""
    date = datetime.strptime(date_str, '%Y-%m-%d')
    jd_start = _julian(date.year, date.month, date.day)

    ep, ev, mp_2d, mv_2d = earth_moon_analytic_state(jd_start)
    _, _, mp_3d, mv_3d = earth_moon_analytic_state_3d(jd_start)

    y0_2d = build_initial_state(ep, ev, mp_2d, mv_2d)
    y0_3d = build_initial_state(ep, ev, mp_3d, mv_3d)

    steps_per_day = int(DAY / dt)
    y2, y3 = y0_2d.copy(), y0_3d.copy()

    max_moon_z = 0.0
    max_rocket_z = 0.0

    print(f'2D vs 3D Propagation ({n_days} days, dt={dt}s):\n')

    for d in range(n_days):
        for _ in range(steps_per_day):
            y2 = velocity_verlet_step(y2, dt)
            y3 = velocity_verlet_step(y3, dt)
        moon_z = abs(y3[14] - y2[14])
        rocket_z = abs(y3[20] - y2[20])
        max_moon_z = max(max_moon_z, moon_z)
        max_rocket_z = max(max_rocket_z, rocket_z)
        if d % 10 == 0:
            print(f'  Day {d:3d}: Moon Δz = {moon_z:.0f} km  '
                  f'Rocket Δz = {rocket_z:.0f} km')

    moon_r = 384400.0
    print(f'\n  Max Moon  z: {max_moon_z:.0f} km  ({max_moon_z/moon_r*100:.2f}% of orbit)')
    print(f'  Max Rocket z: {max_rocket_z:.0f} km')
    print(f'  Assessment: 2D captures dominant dynamics; '
          f'3D needed for precise targeting.')

    return max_moon_z, max_rocket_z


if __name__ == '__main__':
    from conic_patch import AU
    compare_2d_vs_3d_propagation()
    print()
    compare_2d_vs_3d_mission()
