"""M7: 灵敏度与误差分析.

围绕最优解做三方面灵敏度分析：
  1. 近月距 r_m 偏移 ±10%
  2. 发射日期偏移 ±3 天
  3. 积分步长收敛性
"""

import numpy as np
from datetime import datetime, timedelta

from conic_patch import AU
from trajectory import solve_single_date
from optimizer import scan_launch_window


def sensitivity_r_m(r_p_best, r_m_best, side, date_str, delta=0.10):
    """Sensitivity of Delta_v_total to pericynthion variation."""
    print('=' * 60)
    print('SENSITIVITY: Pericynthion Distance (±10%)')
    print('=' * 60)

    base = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
    base_dv = base['Delta_v_total']
    print(f'  Baseline: r_m = {r_m_best:.0f} km  →  Δv = {base_dv:.3f} km/s\n')

    for factor in [1 - delta, 1.0, 1 + delta]:
        r_m = r_m_best * factor
        try:
            result = solve_single_date(date_str, r_p_best, r_m, side, verbose=False)
            dv = result['Delta_v_total']
            diff = dv - base_dv
            print(f'  r_m = {r_m:.0f} km  (×{factor:.2f}):  '
                  f'Δv = {dv:.3f} km/s  (Δ = {diff:+.3f})')
        except ValueError as e:
            print(f'  r_m = {r_m:.0f} km  (×{factor:.2f}):  INFEASIBLE ({e})')

    print()


def sensitivity_t0(best_t0, r_p_best, r_m_best, side, delta_days=3):
    """Sensitivity of Delta_v_total to launch date offset."""
    print('=' * 60)
    print(f'SENSITIVITY: Launch Date (±{delta_days} days)')
    print('=' * 60)

    base = solve_single_date(best_t0, r_p_best, r_m_best, side, verbose=False)
    base_dv = base['Delta_v_total']
    print(f'  Baseline: t0 = {best_t0}  →  Δv = {base_dv:.3f} km/s\n')

    t0_dt = datetime.strptime(best_t0, '%Y-%m-%d')

    for offset in range(-delta_days, delta_days + 1):
        t = t0_dt + timedelta(days=offset)
        date_str = t.strftime('%Y-%m-%d')
        result = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
        dv = result['Delta_v_total']
        diff = dv - base_dv
        marker = ' ← baseline' if offset == 0 else ''
        print(f'  t0 = {date_str}  (offset {offset:+d}d):  '
              f'Δv = {dv:.3f} km/s  (Δ = {diff:+.3f}){marker}')

    print()


def sensitivity_step_size(r_p_best, r_m_best, side, date_str,
                          steps=None):
    """Integration step size convergence study."""
    if steps is None:
        steps = [3600, 1800, 900, 450]

    print('=' * 60)
    print('SENSITIVITY: Integration Step Size')
    print('=' * 60)

    results = []
    for h in steps:
        result = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
        results.append((h, result['Delta_v_total']))
        print(f'  h = {h:5d} s  →  Δv = {result["Delta_v_total"]:.4f} km/s')

    if len(results) >= 2:
        dv_finest = results[-1][1]
        print(f'\n  Relative differences vs finest (h = {steps[-1]} s):')
        for h, dv in results[:-1]:
            rel = abs(dv - dv_finest) / dv_finest
            print(f'    h = {h:5d} s:  {rel:.4%}')

    print()


def run_all_sensitivity():
    """Run full sensitivity analysis using optimizer's best result."""
    print('Running launch window scan to find optimal solution...\n')
    opt = scan_launch_window(
        r_p_range=(2 * 6.96e5, 0.4 * AU, 40),
        verbose=False,
    )

    best = opt['best']
    if best is None:
        print('No feasible solution found. Using reference parameters.')
        best = {
            'date': '2026-06-15',
            'r_p_au': 0.25,
            'Delta_v_total': 0.0,
        }
        r_p_best = 0.25 * AU
        r_m_best = opt['r_m']
        side = opt['side']
        best_t0 = best['date']
    else:
        r_p_best = best['r_p_au'] * AU
        r_m_best = opt['r_m']
        side = opt['side']
        best_t0 = best['date']

    sensitivity_r_m(r_p_best, r_m_best, side, best_t0)
    sensitivity_t0(best_t0, r_p_best, r_m_best, side)
    sensitivity_step_size(r_p_best, r_m_best, side, best_t0)


if __name__ == '__main__':
    run_all_sensitivity()
