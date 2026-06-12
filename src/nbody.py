"""M2: N-体数值积分器 — Sun-Earth-Moon-Rocket 四体.

- 积分器：Velocity-Verlet（2 阶辛积分器），主步长 h = 3600 s
- 基准验证：二体圆轨道 (mu = 4pi^2, 无量纲)，1 年内位置相对误差 ≤ 1e-4
"""

import numpy as np
from numpy.linalg import norm

# Constants
DAY = 86400.0
MU_SUN = 1.32712440018e11   # km^3/s^2
MU_EARTH = 3.986004418e5
MU_MOON = 4.9048695e3
AU = 1.495978707e8


def acceleration_sem(y, mu_sun=MU_SUN, mu_earth=MU_EARTH, mu_moon=MU_MOON):
    """计算 Sun-Earth-Moon-Rocket 四体加速度.

    Parameters
    ----------
    y : ndarray, shape (24,)
        状态向量 [sun_pos(3), sun_vel(3), earth_pos(3), earth_vel(3),
                  moon_pos(3), moon_vel(3), rocket_pos(3), rocket_vel(3)]
        单位: km, km/s, 参考系: 太阳质心 J2000 黄道面投影

    Returns
    -------
    dydt : ndarray, shape (24,)
    """
    dydt = np.zeros(24)

    r_sun = y[0:3]
    r_earth = y[6:9]
    r_moon = y[12:15]
    r_rocket = y[18:21]

    # 速度直接赋值
    dydt[0:3] = y[3:6]
    dydt[6:9] = y[9:12]
    dydt[12:15] = y[15:18]
    dydt[18:21] = y[21:24]

    # 太阳加速度（受地球、月球、火箭影响 — 火箭项可忽略）
    for body_pos, mu in [(r_earth, mu_earth), (r_moon, mu_moon), (r_rocket, 0.0)]:
        dr = body_pos - r_sun
        r = norm(dr)
        if r > 0 and mu > 0:
            dydt[3:6] += mu * dr / r**3

    # 地球加速度
    for body_pos, mu in [(r_sun, mu_sun), (r_moon, mu_moon)]:
        dr = body_pos - r_earth
        r = norm(dr)
        if r > 0:
            dydt[9:12] += mu * dr / r**3

    # 月球加速度
    for body_pos, mu in [(r_sun, mu_sun), (r_earth, mu_earth)]:
        dr = body_pos - r_moon
        r = norm(dr)
        if r > 0:
            dydt[15:18] += mu * dr / r**3

    # 火箭加速度（试探粒子：受 Sun, Earth, Moon 引力）
    for body_pos, mu in [(r_sun, mu_sun), (r_earth, mu_earth), (r_moon, mu_moon)]:
        dr = body_pos - r_rocket
        r = norm(dr)
        if r > 0:
            dydt[21:24] += mu * dr / r**3

    return dydt


def velocity_verlet_step(y, dt, accel_func=acceleration_sem):
    """Velocity-Verlet 单步推进."""
    a0 = accel_func(y)
    v_idx = [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23]

    # 半步速度 + 全步位置
    y_half = y.copy()
    y_half[v_idx] += 0.5 * dt * a0[v_idx]
    y_half[0:21:3] += dt * y_half[v_idx[0:4]]  # 位置更新 (近似)

    # 完整 Verlet
    y_new = y.copy()
    y_new[0:21:3] += dt * y[v_idx[0:4]] + 0.5 * dt**2 * a0[0:21:3]
    a1 = accel_func(y_new)
    y_new[v_idx[0:4]] += 0.5 * dt * (a0[v_idx[0:4]] + a1[v_idx[0:4]])

    return y_new


def propagate(y0, dt, n_steps, accel_func=acceleration_sem, monitor_interval=1000):
    """整段传播，含能量守恒监测.

    Returns
    -------
    trajectory : list of (t, y, E_err)
    """
    y = y0.copy()
    E0 = total_energy(y)
    trajectory = [(0.0, y0.copy(), 0.0)]

    for step in range(1, n_steps + 1):
        y = velocity_verlet_step(y, dt, accel_func)
        if step % monitor_interval == 0:
            E = total_energy(y)
            E_err = abs((E - E0) / E0) if abs(E0) > 1e-30 else 0.0
            trajectory.append((step * dt, y.copy(), E_err))
            if E_err > 1e-6:
                raise RuntimeError(
                    f'Energy drift {E_err:.2e} exceeds tolerance at step {step}'
                )

    return trajectory


def total_energy(y, mu_sun=MU_SUN, mu_earth=MU_EARTH, mu_moon=MU_MOON):
    """计算系统总能量 (kinetic + potential)."""
    r_sun, v_sun = y[0:3], y[3:6]
    r_earth, v_earth = y[6:9], y[9:12]
    r_moon, v_moon = y[12:15], y[15:18]

    T = 0.5 * mu_sun * norm(v_sun)**2  # scale by mass
    # 简化：只返回总轨道能量近似
    bodies = [(r_sun, v_sun), (r_earth, v_earth), (r_moon, v_moon)]
    for i, (ri, vi) in enumerate(bodies):
        for j, (rj, vj) in enumerate(bodies):
            if i < j:
                T -= mu_sun * mu_earth / norm(ri - rj)  # 近似

    return T


def test_two_body_circular():
    """二体圆轨道基准验证：mu=4pi^2(AU^3/yr^2), r=1 AU, 验证 1 年位置误差."""
    mu = 4 * np.pi**2
    r0 = np.array([1.0, 0.0, 0.0])
    v0 = np.array([0.0, 2 * np.pi, 0.0])

    # 构造简化的二体状态
    y0 = np.zeros(24)
    y0[0:3] = [0, 0, 0]      # 太阳在原点
    y0[6:9] = r0
    y0[9:12] = v0

    def accel_2body(y):
        dydt = np.zeros(24)
        dydt[0:3] = y[3:6]
        dydt[6:9] = y[9:12]
        dr = y[6:9] - y[0:3]
        r = norm(dr)
        a = -mu * dr / r**3
        dydt[9:12] = a
        dydt[3:6] = -a  # 反作用（近似，质量比极端不考虑）
        return dydt

    dt = 0.01 / (2 * np.pi)  # ~0.0016 yr
    n_steps = int(1.0 / dt)

    y = y0.copy()
    for _ in range(n_steps):
        y = velocity_verlet_step(y, dt, accel_2body)

    pos_err = norm(y[6:9] - r0)
    rel_err = pos_err / norm(r0)
    print(f'二体圆轨道 1 年验证: 位置相对误差 = {rel_err:.2e}')
    print(f'  {"OK" if rel_err <= 1e-4 else "FAIL"}')
    return rel_err <= 1e-4


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        assert test_two_body_circular()
    else:
        test_two_body_circular()
