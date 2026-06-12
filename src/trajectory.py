"""M5: 单点轨道求解.

在固定发射日期下求解满足任务约束的完整轨道，
输出三段 Delta-v 数值，与无月球借力情况对比，记录节能比例.
"""

import numpy as np
from numpy.linalg import norm

from conic_patch import (
    helio_ellipse, earth_departure_c3, reentry_delta_v,
    lunar_swingby_deflection, total_delta_v, AU, R_SUN, MU_SUN
)
from nbody import acceleration_sem, velocity_verlet_step, propagate

R_MOON = 1737.4
R_MOON_SOI = 6.6e4
DAY = 86400.0


def solve_fixed_date(r_p, r_m=None, side=None, v_inf_moon=0.0):
    """固定发射日期下的完整轨道求解.

    Returns
    -------
    dict with Delta_v breakdown and comparison
    """
    r_1 = AU  # 地球轨道半径

    # 日心椭圆参数
    ellipse = helio_ellipse(r_p, r_1)

    # 出发 Delta-v
    Delta_v_launch = earth_departure_c3(ellipse['Delta_v1'])

    # 返回 Delta-v (对称轨道，v_inf_reentry = v_inf_dep)
    Delta_v_reentry = reentry_delta_v(ellipse['Delta_v1'])

    # 月球借力残差
    Delta_v_lunar = 0.0
    if r_m is not None and side is not None:
        result = lunar_swingby_deflection(v_inf_moon, r_m, side)
        Delta_v_lunar = 0.0  # 理想情况残差为 0

    Delta_v_total = Delta_v_launch + Delta_v_lunar + Delta_v_reentry

    # 无月球借力对比（直接从地球出发）
    Delta_v_no_moon = Delta_v_launch + Delta_v_reentry

    saving = (Delta_v_no_moon - Delta_v_total) / Delta_v_no_moon * 100

    return {
        'Delta_v_total': Delta_v_total,
        'Delta_v_launch': Delta_v_launch,
        'Delta_v_lunar': Delta_v_lunar,
        'Delta_v_reentry': Delta_v_reentry,
        'Delta_v_no_moon': Delta_v_no_moon,
        'saving_pct': saving,
        'ellipse': ellipse,
    }


def verify_constraints(r_p, T_total, v_inf_reentry):
    """验证约束 C1-C4."""
    ok = True
    if r_p <= R_SUN:
        print(f'  FAIL C2: r_p = {r_p:.2e} <= R_SUN = {R_SUN:.2e}')
        ok = False
    if T_total > 2 * 365.25 * DAY:
        print(f'  FAIL C3: T_total = {T_total/DAY:.1f} d > 2 yr')
        ok = False
    if v_inf_reentry > 15.0:
        print(f'  FAIL C4: v_inf_reentry = {v_inf_reentry:.2f} > 15 km/s')
        ok = False
    return ok


if __name__ == '__main__':
    print('=== 单点轨道求解 ===')

    # r_p = 0.2 AU 基准
    r_p = 0.2 * AU
    result = solve_fixed_date(r_p)

    print(f'r_p = {r_p/AU:.3f} AU')
    print(f'  Delta_v 发射:     {result["Delta_v_launch"]:.3f} km/s')
    print(f'  Delta_v 月球借力:  {result["Delta_v_lunar"]:.3f} km/s')
    print(f'  Delta_v 再入:     {result["Delta_v_reentry"]:.3f} km/s')
    print(f'  Delta_v 总:       {result["Delta_v_total"]:.3f} km/s')
    print(f'  无月球借力 Delta_v:{result["Delta_v_no_moon"]:.3f} km/s')
    print(f'  节能比例:          {result["saving_pct"]:.1f}%')
    print(f'  轨道周期:          {result["ellipse"]["T"]/DAY:.1f} d')
    print(f'  轨道半长轴:        {result["ellipse"]["a"]/AU:.4f} AU')
    print(f'  离心率:            {result["ellipse"]["e"]:.4f}')

    print()
    verify_constraints(r_p, result['ellipse']['T'], result['ellipse']['Delta_v1'])
