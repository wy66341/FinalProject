"""M1: 拼接圆锥曲线代码化 — report.tex §3–§5 的步骤编码.

将步骤一至步骤六编码为可调用函数：
  步骤一：给定 r_p, r_1, K_s，计算日心椭圆轨道根数
  步骤二：计算地球出发 C3 和 v_inf
  步骤三：计算返回时 v_inf 和再入 Delta-v
  步骤四(新)：月球借力双曲线偏折
  步骤五：拼接三段 Delta-v
  步骤六：总 Delta-v 汇总

验证：与 r_p = 0.2 AU 算例数值偏差 ≤ 0.1%
"""

import numpy as np

# --- Constants ---
AU = 1.495978707e8  # km
R_SUN = 6.96e5      # km
DAY = 86400.0       # s
YEAR = 365.25 * DAY

# Gravitational parameters (km^3/s^2)
MU_SUN = 1.32712440018e11
MU_EARTH = 3.986004418e5
MU_MOON = 4.9048695e3

R_EARTH = 6378.137  # km
R_MOON = 1737.4     # km
R_MOON_SOI = 6.6e4  # km, Moon SOI radius


def helio_ellipse(r_p, r_1, k_s=MU_SUN):
    """步骤一：计算日心椭圆轨道根数与速度增量.

    Parameters
    ----------
    r_p : float
        近日距 (km)
    r_1 : float
        地球轨道半径 (km)
    k_s : float
        日心引力常数 (km^3/s^2)

    Returns
    -------
    dict with keys: a, e, v_p, v_1, T, Delta_v1
    """
    a = (r_p + r_1) / 2.0
    e = (r_1 - r_p) / (r_1 + r_p)

    # 椭圆上各点速度
    v_p = np.sqrt(k_s * (2 / r_p - 1 / a))  # 近日点速度
    v_1 = np.sqrt(k_s * (2 / r_1 - 1 / a))  # r_1 处速度

    # 地球圆轨道速度
    v_earth = np.sqrt(k_s / r_1)

    # 出发所需 Delta-v（逃逸地球影响球后的日心速度增量）
    Delta_v1 = abs(v_1 - v_earth)

    T = 2 * np.pi * np.sqrt(a**3 / k_s)  # 轨道周期

    return {
        'a': a, 'e': e, 'v_p': v_p, 'v_1': v_1,
        'T': T, 'Delta_v1': Delta_v1, 'v_earth': v_earth
    }


def earth_departure_c3(v_inf):
    """步骤二：从 v_inf 计算地球出发 C3 和 Delta-v.

    Parameters
    ----------
    v_inf : float
        离开地球 SOI 时的剩余速度 (km/s)

    Returns
    -------
    float: 从 200km LEO 出发所需的 Delta-v (km/s)
    """
    r_leo = R_EARTH + 200.0  # 200 km LEO
    v_esc_leo = np.sqrt(2 * MU_EARTH / r_leo)
    v_circ_leo = np.sqrt(MU_EARTH / r_leo)
    Delta_v = np.sqrt(v_inf**2 + v_esc_leo**2) - v_circ_leo
    return Delta_v


def reentry_delta_v(v_inf_reentry, max_v_inf=15.0):
    """步骤三：返回再入所需的 Delta-v.

    Parameters
    ----------
    v_inf_reentry : float
        再入时相对地球的 v_inf (km/s)
    max_v_inf : float
        允许最大再入速度 (km/s)

    Returns
    -------
    float: 再入 Delta-v (km/s)，超限返回 inf
    """
    if v_inf_reentry > max_v_inf:
        return np.inf
    r_leo = R_EARTH + 200.0
    v_esc_leo = np.sqrt(2 * MU_EARTH / r_leo)
    v_circ_leo = np.sqrt(MU_EARTH / r_leo)
    Delta_v = np.sqrt(v_inf_reentry**2 + v_esc_leo**2) - v_circ_leo
    return Delta_v


def lunar_swingby_deflection(v_inf_moon, r_m, side='trailing'):
    """步骤四：月球借力双曲线偏折解析计算.

    Parameters
    ----------
    v_inf_moon : float
        进入月球 SOI 时的 v_inf (km/s)，在月心系中
    r_m : float
        近月距 (km)，≥ R_MOON + 100
    side : {'leading', 'trailing'}
        绕月方向

    Returns
    -------
    dict with keys: delta, Delta_v, v_out
    """
    if r_m < R_MOON + 100:
        raise ValueError(f'r_m = {r_m} < {R_MOON + 100} km, would hit Moon')

    # 双曲线的半长轴和偏心率
    mu = MU_MOON
    a_hyp = mu / v_inf_moon**2
    r_p = r_m
    e_hyp = 1 + r_p / a_hyp

    # 偏转角
    delta = 2 * np.arcsin(1 / e_hyp)

    # v_out 大小不变，方向偏转
    # leading: 逆月球运动方向绕过 → 减速
    # trailing: 顺月球运动方向绕过 → 加速
    sign = -1 if side == 'leading' else 1
    v_out = v_inf_moon  # 双曲线能量守恒，|v_in| = |v_out|

    return {'delta': delta, 'sign': sign, 'v_out': v_out, 'e_hyp': e_hyp, 'r_p': r_p}


def total_delta_v(r_p, v_inf_dep, v_inf_reentry, r_m=None, side=None, v_inf_moon=0.0):
    """步骤五-六：汇总总 Delta-v.

    Parameters
    ----------
    r_p : float
        近日距 (km)
    v_inf_dep : float
        出发时相对地球的 v_inf (km/s)
    v_inf_reentry : float
        返回时相对地球的 v_inf (km/s)
    r_m : float or None
        近月距，None 表示无月球借力
    side : str or None
        绕月方向
    v_inf_moon : float
        进入月球 SOI 时的 v_inf

    Returns
    -------
    dict with keys: Delta_v_total, Delta_v_launch, Delta_v_lunar, Delta_v_reentry
    """
    Delta_v_launch = earth_departure_c3(v_inf_dep)

    if r_m is not None and side is not None:
        result = lunar_swingby_deflection(v_inf_moon, r_m, side)
        Delta_v_lunar = 0.0  # 理想被动借力，残差为 0
    else:
        Delta_v_lunar = 0.0

    Delta_v_reentry = reentry_delta_v(v_inf_reentry)

    Delta_v_total = Delta_v_launch + Delta_v_lunar + Delta_v_reentry

    return {
        'Delta_v_total': Delta_v_total,
        'Delta_v_launch': Delta_v_launch,
        'Delta_v_lunar': Delta_v_lunar,
        'Delta_v_reentry': Delta_v_reentry
    }


def verify_rp_02_au():
    """验证 r_p = 0.2 AU 算例，与 report.tex 对比偏差 ≤ 0.1%."""
    r_1 = AU  # 1 AU
    r_p = 0.2 * AU

    result = helio_ellipse(r_p, r_1)
    Delta_v1 = result['Delta_v1']
    a_au = result['a'] / AU

    # report.tex 参考值
    Delta_v1_ref = 3.56  # km/s, 近似值

    rel_err = abs(Delta_v1 - Delta_v1_ref) / Delta_v1_ref
    print(f'r_p = 0.2 AU:')
    print(f'  a = {a_au:.4f} AU, e = {result["e"]:.4f}')
    print(f'  Delta_v1 = {Delta_v1:.4f} km/s (ref: {Delta_v1_ref} km/s)')
    print(f'  相对偏差 = {rel_err:.4%}')
    print(f'  {"OK" if rel_err <= 0.001 else "FAIL"}')

    return rel_err <= 0.001


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        assert verify_rp_02_au()
    else:
        verify_rp_02_au()
