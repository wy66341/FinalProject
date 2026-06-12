"""M4: 月球借力 — 解析公式 + 数值仿真双套实现.

- 解析：双曲线偏折角公式 delta = 2*arcsin(1/e)
- 数值：在月球 SOI 内用二体传播火箭轨迹，进出 SOI 做坐标变换
- 两套对比验证
"""

import numpy as np
from numpy.linalg import norm

MU_MOON = 4.9048695e3   # km^3/s^2
MU_EARTH = 3.986004418e5
R_MOON = 1737.4          # km
R_MOON_SOI = 6.6e4       # km
DAY = 86400.0


def analytical_deflection(v_inf, r_p, side='trailing'):
    """解析计算月球借力偏转角.

    Parameters
    ----------
    v_inf : float
        进入月球 SOI 时的 v_inf 大小 (km/s)
    r_p : float
        近月距 (km), ≥ R_MOON + 100
    side : str
        'leading' 或 'trailing'

    Returns
    -------
    dict with delta (rad), delta_v (km/s), e
    """
    if r_p < R_MOON + 100:
        raise ValueError(f'r_p = {r_p} km < {R_MOON + 100} km')

    a = MU_MOON / v_inf**2
    e = 1 + r_p / a
    delta = 2 * np.arcsin(1 / e)
    sign = -1 if side == 'leading' else 1

    return {'delta': delta, 'sign': sign, 'e': e, 'a': a, 'r_p': r_p, 'v_inf': v_inf}


def numerical_swingby(r_in, v_in_moon, t_span, dt=60.0):
    """在月球 SOI 内数值积分火箭轨迹.

    Parameters
    ----------
    r_in : ndarray (3,)
        进入月球 SOI 时的月心位置 (km)
    v_in_moon : ndarray (3,)
        进入月球 SOI 时的月心速度 (km/s)
    t_span : float
        在 SOI 内的时间 (s)
    dt : float
        积分步长 (s)

    Returns
    -------
    (r_out, v_out): 出 SOI 时的月心位置和速度
    """
    n_steps = int(t_span / dt)
    r = r_in.copy().astype(float)
    v = v_in_moon.copy().astype(float)

    for _ in range(n_steps):
        r_norm = norm(r)
        if r_norm < R_MOON:
            return None  # 撞月
        a = -MU_MOON * r / r_norm**3
        v += 0.5 * dt * a
        r += dt * v
        a_new = -MU_MOON * r / norm(r)**3
        v += 0.5 * dt * a_new

        if norm(r) > R_MOON_SOI:
            break

    return r, v


def compare_analytic_numerical(v_inf, r_p, side='trailing'):
    """对比解析和数值结果."""
    analytic = analytical_deflection(v_inf, r_p, side)

    # 构造进入 SOI 的初始条件
    # 月球 SOI 边界入射
    b = R_MOON_SOI  # 瞄准距
    r_in = np.array([-np.sqrt(b**2 - r_p**2), r_p, 0.0])
    v_in = np.array([v_inf, 0.0, 0.0])

    t_soi = 2 * R_MOON_SOI / v_inf  # 近似穿越时间
    numeric = numerical_swingby(r_in, v_in, t_soi)

    if numeric is None:
        print('数值仿真：火箭撞月！')
        return False

    r_out, v_out = numeric
    v_out_norm = norm(v_out)
    delta_numeric = np.arccos(np.dot(v_in, v_out) / (norm(v_in) * v_out_norm))

    delta_err = abs(delta_numeric - analytic['delta'])
    print(f'v_inf = {v_inf:.3f} km/s, r_p = {r_p:.0f} km, side = {side}')
    print(f'  解析偏转角: {np.degrees(analytic["delta"]):.3f} deg')
    print(f'  数值偏转角: {np.degrees(delta_numeric):.3f} deg')
    print(f'  偏差: {np.degrees(delta_err):.4f} deg')
    print(f'  v_out 守恒: |v_out|/|v_in| = {v_out_norm / v_inf:.6f}')

    return delta_err < 1e-3  # < 0.001 rad ~ 0.06 deg


if __name__ == '__main__':
    print('=== 月球借力 解析 vs 数值 ===')
    for v_inf in [1.0, 2.0, 3.0]:
        for r_p in [2000.0, 5000.0, 10000.0]:
            for side in ['trailing', 'leading']:
                compare_analytic_numerical(v_inf, r_p, side)
                print()
