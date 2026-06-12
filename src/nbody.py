"""M2: N-体数值积分器 — Sun-Earth-Moon-Rocket 四体.

- 积分器：Velocity-Verlet（2 阶辛积分器），主步长 h = 3600 s
- 基准验证：二体圆轨道 (mu=4pi², 无量纲), 1 年内位置相对误差 ≤ 1e-4
- 状态向量 y[24]: sun(pos,vel), earth(pos,vel), moon(pos,vel), rocket(pos,vel)
"""

import numpy as np
from numpy.linalg import norm

DAY = 86400.0
MU_SUN = 1.32712440018e11
MU_EARTH = 3.986004418e5
MU_MOON = 4.9048695e3
AU = 1.495978707e8


def _accel_on(r_body, r_others, mus, r_sun=None):
    """Compute gravitational acceleration on a body.

    Parameters
    ----------
    r_body : (3,) ndarray — position of the body being accelerated
    r_others : list of (3,) ndarray — positions of gravitating bodies
    mus : list of float — GM values for each gravitating body
    r_sun : (3,) ndarray or None — sun position, for indirect term correction

    Returns
    -------
    a : (3,) ndarray — total acceleration
    """
    a = np.zeros(3)
    for r_other, mu in zip(r_others, mus):
        dr = r_other - r_body
        r = norm(dr)
        if r > 1.0:  # avoid self-division
            a += mu * dr / r**3
    return a


def acceleration_sem(y):
    """Compute dydt for Sun-Earth-Moon-Rocket 4-body system.

    y layout (24,):
      [sun_p(3), sun_v(3), earth_p(3), earth_v(3),
       moon_p(3), moon_v(3), rocket_p(3), rocket_v(3)]
    Returns dydt same layout — velocities copied, accelerations computed.
    """
    dydt = np.zeros(24)

    r_sun = y[0:3]
    r_earth = y[6:9]
    r_moon = y[12:15]
    r_rocket = y[18:21]

    # Velocities: dr/dt = v
    for i in range(4):
        dydt[i * 6 : i * 6 + 3] = y[i * 6 + 3 : i * 6 + 6]

    # Sun acceleration (from Earth, Moon; rocket mass negligible)
    dydt[3:6] = _accel_on(r_sun, [r_earth, r_moon], [MU_EARTH, MU_MOON])

    # Earth acceleration (from Sun, Moon)
    dydt[9:12] = _accel_on(r_earth, [r_sun, r_moon], [MU_SUN, MU_MOON])

    # Moon acceleration (from Sun, Earth)
    dydt[15:18] = _accel_on(r_moon, [r_sun, r_earth], [MU_SUN, MU_EARTH])

    # Rocket acceleration — test particle under Sun + Earth + Moon gravity
    dydt[21:24] = _accel_on(r_rocket, [r_sun, r_earth, r_moon],
                            [MU_SUN, MU_EARTH, MU_MOON])

    return dydt


def velocity_verlet_step(y, dt, accel_func=acceleration_sem):
    """Standard Velocity-Verlet step.

    r_{n+1} = r_n + v_n * dt + 0.5 * a_n * dt^2
    v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    """
    a0 = accel_func(y)
    y_new = y.copy()

    # Position update for all 4 bodies
    for i in range(4):
        p_s = i * 6
        v_s = p_s + 3
        y_new[p_s:p_s + 3] = (y[p_s:p_s + 3]
                              + y[v_s:v_s + 3] * dt
                              + 0.5 * a0[p_s:p_s + 3] * dt**2)

    # Acceleration at new positions
    a1 = accel_func(y_new)

    # Velocity update
    for i in range(4):
        v_s = i * 6 + 3
        y_new[v_s:v_s + 3] = (y[v_s:v_s + 3]
                              + 0.5 * (a0[v_s:v_s + 3] + a1[v_s:v_s + 3]) * dt)

    return y_new


def system_energy(y):
    """Total mechanical energy (kinetic + potential) of Sun-Earth-Moon system.

    E = T_sun + T_earth + T_moon + U_{se} + U_{sm} + U_{em}
    """
    bodies = []
    for i in range(3):  # skip rocket (test particle)
        m = [MU_SUN, MU_EARTH, MU_MOON][i]
        p = y[i * 6 : i * 6 + 3]
        v = y[i * 6 + 3 : i * 6 + 6]
        bodies.append((m, p, v))

    T = sum(0.5 * m * np.dot(v, v) for m, p, v in bodies)

    U = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            dr = bodies[j][1] - bodies[i][1]
            r = norm(dr)
            if r > 0:
                U -= bodies[i][0] * bodies[j][0] / r

    return T + U


def propagate(y0, dt, n_steps, accel_func=acceleration_sem, monitor_interval=1000):
    """整段传播，含能量守恒监测.

    Returns
    -------
    traj : list of (t, y, E_err)
    """
    y = y0.copy()
    E0 = system_energy(y)
    traj = [(0.0, y0.copy(), 0.0)]

    for step in range(1, n_steps + 1):
        y = velocity_verlet_step(y, dt, accel_func)

        if step % monitor_interval == 0:
            E = system_energy(y)
            E_err = abs((E - E0) / E0) if abs(E0) > 1e-30 else 0.0
            traj.append((step * dt, y.copy(), E_err))
            if E_err > 1e-6:
                raise RuntimeError(
                    f'Energy drift {E_err:.2e} exceeds 1e-6 at step {step}'
                )

    return traj


def test_two_body_circular():
    """二体圆轨道基准验证.

    mu = 4*pi^2, r0 = 1 AU, v0 = 2*pi AU/yr → 圆轨道周期 1 yr.
    积分 1 年后位置相对误差应 ≤ 1e-4.
    """
    mu = 4.0 * np.pi**2
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 2.0 * np.pi, 0.0])

    y0 = np.zeros(12)  # 只积分一个天体绕中心
    y0[0:3] = r0
    y0[3:6] = v0

    def accel_2b(y):
        dydt = np.zeros(12)
        dydt[0:3] = y[3:6]
        r = norm(y[0:3])
        dydt[3:6] = -mu * y[0:3] / r**3
        return dydt

    def verlet_step_2b(y, dt):
        a0 = accel_2b(y)
        y_new = y.copy()
        # a0[0:3]=velocity, a0[3:6]=acceleration
        y_new[0:3] = y[0:3] + y[3:6] * dt + 0.5 * a0[3:6] * dt**2
        a1 = accel_2b(y_new)
        y_new[3:6] = y[3:6] + 0.5 * (a0[3:6] + a1[3:6]) * dt
        return y_new

    # Integrate 1 year
    dt = 0.0001  # ~0.0001 yr ≈ 0.87 h
    n_steps = int(1.0 / dt)

    y = y0.copy()
    for _ in range(n_steps):
        y = verlet_step_2b(y, dt)

    pos_err = norm(y[0:3] - r0) / norm(r0)
    v_err = norm(y[3:6] - v0) / norm(v0)

    print(f'二体圆轨道 1 年基准验证:')
    print(f'  位置相对误差 = {pos_err:.2e}  (要求 ≤ 1e-4)')
    print(f'  速度相对误差 = {v_err:.2e}')
    ok = pos_err <= 1e-4
    print(f'  {"PASS" if ok else "FAIL"}')
    return ok


def earth_moon_analytic_state(jd, mu_sun=MU_SUN, mu_earth=MU_EARTH):
    """Analytic Earth/Moon state from mean orbital elements (J2000 ecliptic).

    Used as fallback when Horizons proxy is unavailable.
    Returns (earth_pos, earth_vel, moon_pos, moon_vel) in km, km/s.
    """
    # Julian centuries from J2000.0
    T = (jd - 2451545.0) / 36525.0

    # Earth mean elements (Standish 1998, low-precision)
    a_e = 1.00000261 * AU
    e_e = 0.01671123 - 0.00004392 * T
    i_e = np.radians(-0.00001531 - 0.01294668 * T)
    L_e = np.radians(100.466915 + 35999.373063 * T)
    varpi_e = np.radians(102.930058 + 0.317953 * T)
    Omega_e = np.radians(-5.112603 - 0.241238 * T)

    M_e = L_e - varpi_e
    E_e = _solve_kepler(M_e, e_e, tol=1e-12)
    nu_e = 2 * np.arctan2(np.sqrt(1 + e_e) * np.sin(E_e / 2),
                          np.sqrt(1 - e_e) * np.cos(E_e / 2))
    r_e_mag = a_e * (1 - e_e * np.cos(E_e))

    # Position in orbital plane
    x_orb = r_e_mag * np.cos(nu_e)
    y_orb = r_e_mag * np.sin(nu_e)

    # Velocity in orbital plane
    h_e = np.sqrt(mu_sun * a_e * (1 - e_e**2))
    vx_orb = -mu_sun / h_e * np.sin(nu_e)
    vy_orb = mu_sun / h_e * (e_e + np.cos(nu_e))

    # Rotate to ecliptic
    earth_pos = _rot_z(_rot_x(x_orb, y_orb, 0, i_e), varpi_e)
    earth_vel = _rot_z(_rot_x(vx_orb, vy_orb, 0, i_e), varpi_e)

    # Moon mean elements (low-precision analytic)
    a_m = 384400.0
    e_m = 0.0554
    i_m = np.radians(5.16)
    L_m = np.radians(218.316 + 481267.881 * T)
    varpi_m = np.radians(318.15 + 0.05295 * T)
    Omega_m = np.radians(125.08 - 0.05295 * T)

    M_m = L_m - varpi_m
    E_m = _solve_kepler(M_m, e_m)
    nu_m = 2 * np.arctan2(np.sqrt(1 + e_m) * np.sin(E_m / 2),
                          np.sqrt(1 - e_m) * np.cos(E_m / 2))
    r_m_mag = a_m * (1 - e_m * np.cos(E_m))

    x_m_orb = r_m_mag * np.cos(nu_m)
    y_m_orb = r_m_mag * np.sin(nu_m)
    h_m = np.sqrt(mu_earth * a_m * (1 - e_m**2))
    vx_m_orb = -mu_earth / h_m * np.sin(nu_m)
    vy_m_orb = mu_earth / h_m * (e_m + np.cos(nu_m))

    moon_pos_geo = _rot_z(_rot_x(x_m_orb, y_m_orb, 0, i_m),
                          varpi_m - Omega_m)
    moon_vel_geo = _rot_z(_rot_x(vx_m_orb, vy_m_orb, 0, i_m),
                          varpi_m - Omega_m)

    moon_pos = earth_pos + moon_pos_geo
    moon_vel = earth_vel + moon_vel_geo

    return earth_pos, earth_vel, moon_pos, moon_vel


def _solve_kepler(M, e, tol=1e-12, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson."""
    E = M + e * np.sin(M) / (1 - np.sin(M + e) + np.sin(M))
    for _ in range(max_iter):
        dE = (M - E + e * np.sin(E)) / (1 - e * np.cos(E))
        E += dE
        if abs(dE) < tol:
            break
    return E


def _rot_x(x, y, z, theta):
    """Rotate around x-axis."""
    c, s = np.cos(theta), np.sin(theta)
    y_new = c * y - s * z
    z_new = s * y + c * z
    return np.array([x, y_new, z_new])


def _rot_z(vec, theta):
    """Rotate vector around z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * vec[0] - s * vec[1],
                     s * vec[0] + c * vec[1],
                     vec[2]])


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        ok = test_two_body_circular()
        import sys as _sys
        _sys.exit(0 if ok else 1)
    else:
        test_two_body_circular()
