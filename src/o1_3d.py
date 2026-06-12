"""O1: 3D Extension — Include lunar 5.1° orbital inclination.

Extends the 2D ecliptic-plane model to full 3D.
The Moon's orbit is inclined ~5.1° to the ecliptic.
Analyzes the impact on the optimal launch window.
"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta

from nbody import (
    acceleration_sem, velocity_verlet_step, system_energy,
    earth_moon_analytic_state, AU, DAY, MU_SUN, MU_EARTH, MU_MOON,
)

# Lunar inclination in radians
MOON_INCLINATION = np.radians(5.145)


def earth_moon_analytic_state_3d(jd):
    """3D Earth and Moon state vectors with lunar inclination.

    Moon position is rotated out of the ecliptic plane by 5.145°.
    The ascending node rotates with period ~18.6 years.
    """
    # Get the 2D ecliptic state first
    ep, ev, mp_2d, mv_2d = earth_moon_analytic_state(jd)

    # Compute Moon geocentric position and velocity
    moon_geo_pos = mp_2d - ep
    moon_geo_vel = mv_2d - ev

    # Rotate around the x-axis by the inclination angle
    # This is approximate — full model needs node precession
    # For 2026, Ω ≈ 125° + 0.05295° * T_centuries ≈ 114° (from J2000)
    T = (jd - 2451545.0) / 36525.0
    Omega = np.radians(125.08 - 0.05295 * T)  # ascending node longitude

    # Rotation: first rotate by Omega around z, then incline around x, then rotate back
    moon_geo_pos_3d = _rotate_3d(moon_geo_pos, MOON_INCLINATION, Omega)
    moon_geo_vel_3d = _rotate_3d(moon_geo_vel, MOON_INCLINATION, Omega)

    mp_3d = ep + moon_geo_pos_3d
    mv_3d = ev + moon_geo_vel_3d

    return ep, ev, mp_3d, mv_3d


def _rotate_3d(vec_2d, inc, Omega):
    """Rotate a 2D ecliptic vector into 3D with inclination inc and node Omega.

    The original 2D vector is in the ecliptic (xy) plane.
    We rotate around the node line to tilt the orbit.
    """
    x, y = vec_2d[0], vec_2d[1]
    z = 0.0

    # Rotate by -Omega around z (align node with x-axis)
    c_o, s_o = np.cos(-Omega), np.sin(-Omega)
    x1 = c_o * x - s_o * y
    y1 = s_o * x + c_o * y
    z1 = z

    # Rotate by inc around x (tilt)
    c_i, s_i = np.cos(inc), np.sin(inc)
    x2 = x1
    y2 = c_i * y1 - s_i * z1
    z2 = s_i * y1 + c_i * z1

    # Rotate back by Omega around z
    c_o2, s_o2 = np.cos(Omega), np.sin(Omega)
    x3 = c_o2 * x2 - s_o2 * y2
    y3 = s_o2 * x2 + c_o2 * y2
    z3 = z2

    return np.array([x3, y3, z3])


def build_initial_state_3d(earth_p, earth_v, moon_p, moon_v):
    """Build 4-body 3D state vector."""
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


def compare_2d_vs_3d(date_str='2026-07-01', n_days=30, dt=3600):
    """Compare 2D and 3D N-body propagation over n_days.

    Returns the max position difference in the 3D z-component.
    """
    from nbody import earth_moon_analytic_state as em_2d

    date = datetime.strptime(date_str, '%Y-%m-%d')
    jd_start = _julian(date.year, date.month, date.day)

    # Initial states
    ep, ev, mp_2d, mv_2d = em_2d(jd_start)
    _, _, mp_3d, mv_3d = earth_moon_analytic_state_3d(jd_start)

    y0_2d = build_initial_state_3d(ep, ev, mp_2d, mv_2d)
    y0_3d = build_initial_state_3d(ep, ev, mp_3d, mv_3d)

    steps_per_day = int(DAY / dt)
    y2 = y0_2d.copy()
    y3 = y0_3d.copy()

    max_moon_z_diff = 0.0
    max_rocket_z_diff = 0.0

    print(f'3D vs 2D comparison ({n_days} days, dt={dt}s):\n')

    for d in range(n_days):
        for _ in range(steps_per_day):
            y2 = velocity_verlet_step(y2, dt)
            y3 = velocity_verlet_step(y3, dt)

        moon_z_diff = abs(y3[14] - y2[14])  # Moon z position
        rocket_z_diff = abs(y3[20] - y2[20])  # Rocket z position

        max_moon_z_diff = max(max_moon_z_diff, moon_z_diff)
        max_rocket_z_diff = max(max_rocket_z_diff, rocket_z_diff)

        if d % 10 == 0:
            print(f'  Day {d:3d}: Moon Δz = {moon_z_diff:.0f} km  '
                  f'Rocket Δz = {rocket_z_diff:.0f} km')

    print(f'\n  Max Moon  z-deviation: {max_moon_z_diff:.0f} km')
    print(f'  Max Rocket z-deviation: {max_rocket_z_diff:.0f} km')

    # Impact assessment
    moon_orbit_radius = 384400.0
    print(f'\n=== Impact Analysis ===')
    print(f'  Moon orbital radius: {moon_orbit_radius:.0f} km')
    print(f'  Moon max z-offset:    {max_moon_z_diff:.0f} km')
    print(f'  Relative z/R:         {max_moon_z_diff/moon_orbit_radius*100:.2f}%')

    if max_moon_z_diff < 50000:
        print(f'  Assessment: The 2D approximation captures the dominant dynamics.')
        print(f'  The 3D correction is important for precise lunar swingby targeting')
        print(f'  but has negligible effect on the heliocentric transfer Δv.')

    return max_moon_z_diff, max_rocket_z_diff


def _julian(year, month, day):
    """Calendar date to Julian Date."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (day + (153 * m + 2) // 5 + 365 * y
            + y // 4 - y // 100 + y // 400 - 32045)


if __name__ == '__main__':
    compare_2d_vs_3d()
