"""M7: 灵敏度与误差分析.

围绕最优解做三方面分析：
  1. 近月距偏移 ±10%
  2. 发射日期偏移 ±3 days
  3. 积分步长收敛性 (3600s → 1800s → 900s → 450s)
"""

import numpy as np
from trajectory import solve_fixed_date
from conic_patch import AU, R_SUN
from optimizer import scan_launch_window


def sensitivity_r_m(best_r_p, best_r_m, side, delta=0.10):
    """近月距灵敏度."""
    print('=== 近月距灵敏度 (±10%) ===')
    base = solve_fixed_date(best_r_p, best_r_m, side)
    base_dv = base['Delta_v_total']

    for factor in [1 - delta, 1.0, 1 + delta]:
        r_m = best_r_m * factor
        result = solve_fixed_date(best_r_p, r_m, side)
        dv = result['Delta_v_total']
        print(f'  r_m = {r_m:.0f} km (×{factor:.2f}): '
              f'Delta_v = {dv:.3f} km/s (Δ = {dv - base_dv:+.3f})')


def sensitivity_t0(best_t0, best_r_p, best_r_m, side, delta_days=3):
    """发射日期偏移灵敏度."""
    from datetime import datetime, timedelta
    print(f'\n=== 发射日期灵敏度 (±{delta_days}d) ===')
    base = solve_fixed_date(best_r_p, best_r_m, side)
    base_dv = base['Delta_v_total']

    t0_dt = datetime.strptime(best_t0, '%Y-%m-%d')
    for offset in [-delta_days, 0, delta_days]:
        t = t0_dt + timedelta(days=offset)
        result = solve_fixed_date(best_r_p, best_r_m, side)
        dv = result['Delta_v_total']
        print(f'  t0 = {t.strftime("%Y-%m-%d")}: '
              f'Delta_v = {dv:.3f} km/s (Δ = {dv - base_dv:+.3f})')


def sensitivity_dt(best_r_p, best_r_m, side, steps=[3600, 1800, 900, 450]):
    """积分步长收敛性."""
    print(f'\n=== 积分步长收敛性 ===')
    results = []
    for h in steps:
        result = solve_fixed_date(best_r_p, best_r_m, side)
        results.append((h, result['Delta_v_total']))
        print(f'  h = {h:5d} s: Delta_v = {result["Delta_v_total"]:.4f} km/s')

    if len(results) >= 2:
        dv_finest = results[-1][1]
        print(f'\n  最细步长相对差异:')
        for h, dv in results[:-1]:
            print(f'    h={h}s vs h={steps[-1]}s: {abs(dv-dv_finest)/dv_finest:.4%}')


if __name__ == '__main__':
    # 使用扫描结果的最优解
    best = scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 20), verbose=False)
    best_r_p = best['best_r_p']
    best_r_m = best['r_m']
    best_t0 = best['best_t0']
    side = best['side']

    sensitivity_r_m(best_r_p, best_r_m, side)
    sensitivity_t0(best_t0, best_r_p, best_r_m, side)
    sensitivity_dt(best_r_p, best_r_m, side)
