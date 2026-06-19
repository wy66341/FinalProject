"""M7: 灵敏度与误差分析.

围绕最优解做三方面灵敏度分析（含 N-体验证）：
  1. 近月距 r_m 偏移 — 展示真实借力效果（Δv 随 r_m 变化）
  2. 发射日期偏移 ±3 天 — 展示最优窗口稳定性
  3. N-体积分步长收敛性 — 验证 N-体轨迹对步长的敏感性
"""

import numpy as np
from datetime import datetime, timedelta

from conic_patch import AU
from trajectory import solve_single_date, verify_full_mission


def sensitivity_r_m(r_p_best, r_m_best, side, date_str, delta=0.10):
    """Sensitivity of Delta_v_total to pericynthion variation.

    Now that the trajectory module computes real dv_lunar savings from
    the flyby geometry, r_m variation produces measurable Δv changes:
    closer approach → larger deflection → more savings.
    """
    print('=' * 60)
    print('SENSITIVITY: Pericynthion Distance')
    print('=' * 60)

    base = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
    base_dv = base['Delta_v_total']
    base_launch = base['Delta_v_launch']
    base_nomoon = base['Delta_v_total_nomoon']
    print(f'  Baseline: r_m = {r_m_best:.0f} km')
    print(f'    Δv_launch:  {base_launch:.4f} km/s')
    print(f'    Δv_reentry: {base["Delta_v_reentry"]:.4f} km/s')
    print(f'    Δv_total:   {base_dv:.4f} km/s  '
          f'(saving {base["saving_pct"]:.2f}% vs no-moon {base_nomoon:.4f})\n')

    print(f'  {"r_m (km)":>10s}  {"×factor":>8s}  {"Δv_launch":>10s}  '
          f'{"Δv_total":>10s}  {"ΔΔv":>10s}  {"saving%":>8s}')
    print(f'  {"-"*10}  {"-"*8}  {"-"*10}  {"-"*10}  {"-"*10}  {"-"*8}')

    for factor in [1 - delta, 1.0, 1 + delta]:
        r_m = max(1838.0, r_m_best * factor)
        try:
            result = solve_single_date(date_str, r_p_best, r_m, side, verbose=False)
            dv = result['Delta_v_total']
            diff = dv - base_dv
            marker = ' ← base' if factor == 1.0 else ''
            print(f'  {r_m:10.0f}  {factor:8.2f}  {result["Delta_v_launch"]:10.4f}  '
                  f'{dv:10.4f}  {diff:+10.4f}  {result["saving_pct"]:7.2f}%{marker}')
        except ValueError as e:
            print(f'  {r_m:10.0f}  {factor:8.2f}  INFEASIBLE ({e})')

    print()


def sensitivity_t0(best_t0, r_p_best, r_m_best, side, delta_days=3):
    """Sensitivity of Delta_v_total to launch date offset."""
    print('=' * 60)
    print(f'SENSITIVITY: Launch Date (±{delta_days} days)')
    print('=' * 60)

    base = solve_single_date(best_t0, r_p_best, r_m_best, side, verbose=False)
    base_dv = base['Delta_v_total']
    base_launch = base['Delta_v_launch']
    print(f'  Baseline: t0 = {best_t0}')
    print(f'    Δv_launch:  {base_launch:.4f} km/s')
    print(f'    Δv_reentry: {base["Delta_v_reentry"]:.4f} km/s')
    print(f'    Δv_total:   {base_dv:.4f} km/s  '
          f'(saving {base["saving_pct"]:.2f}%)\n')

    print(f'  {"t0":>12s}  {"offset":>6s}  {"Δv_total":>10s}  '
          f'{"ΔΔv":>10s}  {"moon_phase":>10s}')
    print(f'  {"-"*12}  {"-"*6}  {"-"*10}  {"-"*10}  {"-"*10}')

    t0_dt = datetime.strptime(best_t0, '%Y-%m-%d')

    for offset in range(-delta_days, delta_days + 1):
        t = t0_dt + timedelta(days=offset)
        date_str = t.strftime('%Y-%m-%d')
        result = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
        dv = result['Delta_v_total']
        diff = dv - base_dv
        phase = result.get('moon_phase_angle_deg', 'N/A')
        marker = ' ← baseline' if offset == 0 else ''
        print(f'  {date_str}  {offset:+5d}d  {dv:10.4f}  '
              f'{diff:+10.4f}  {phase!s:>10s}{marker}')

    print()


def sensitivity_step_size(r_p_best, r_m_best, side, date_str,
                          dt_steps=None):
    """N-body integration step size convergence study.

    Tests the sensitivity of the N-body trajectory (Moon approach, Earth
    return) to integration step size. The analytic Δv is step-independent;
    the N-body verification accuracy depends on step size.
    """
    if dt_steps is None:
        dt_steps = [3600, 1800, 900]

    print('=' * 60)
    print('SENSITIVITY: N-Body Integration Step Size')
    print('=' * 60)
    print(f'  Reference: date={date_str}, r_p={r_p_best/AU:.3f} AU, '
          f'r_m={r_m_best:.0f} km\n')

    # Analytic solution is step-independent
    base = solve_single_date(date_str, r_p_best, r_m_best, side, verbose=False)
    print(f'  Analytic Δv_total = {base["Delta_v_total"]:.4f} km/s  '
          f'(step-independent)\n')

    print(f'  {"dt (s)":>8s}  {"Moon appr (km)":>15s}  '
          f'{"Earth appr (km)":>15s}  {"r_min (km)":>15s}')
    print(f'  {"-"*8}  {"-"*15}  {"-"*15}  {"-"*15}')

    results = []
    for dt in dt_steps:
        # Re-run N-body verification with this step size
        # verify_full_mission uses internal dt=3600; to test sensitivity,
        # we temporarily modify the function's dt parameter
        mission = verify_full_mission(date_str, r_p_best, r_m_best, side, dt=dt)
        moon_app = mission.get('moon_closest_approach_km', float('inf'))
        earth_app = mission.get('earth_closest_approach_km', float('inf'))
        peri = mission.get('perihelion_km', 0)
        results.append((dt, moon_app, earth_app, peri))
        print(f'  {dt:8d}  {moon_app:15,.0f}  {earth_app:15,.0f}  {peri:15,.0f}')

    if len(results) >= 2:
        ref = results[-1]  # finest step
        print(f'\n  Convergence vs finest (dt={ref[0]} s):')
        for dt, moon, earth, peri in results[:-1]:
            moon_rel = abs(moon - ref[1]) / ref[1] if ref[1] > 0 else 0
            earth_rel = abs(earth - ref[2]) / ref[2] if ref[2] > 0 else 0
            print(f'    dt={dt:5d}: moon_err={moon_rel:.4%}  earth_err={earth_rel:.4%}')

    print()


def run_all_sensitivity():
    """Run full sensitivity analysis around the optimal solution."""
    best_t0 = '2026-01-07'
    r_p_best = 0.4 * AU
    r_m_best = 5000.0
    side = 'trailing'

    print(f'Reference: t0={best_t0}, r_p={r_p_best/AU:.3f} AU, '
          f'r_m={r_m_best:.0f} km, side={side}\n')

    sensitivity_r_m(r_p_best, r_m_best, side, best_t0)
    sensitivity_t0(best_t0, r_p_best, r_m_best, side)
    sensitivity_step_size(r_p_best, r_m_best, side, best_t0)


if __name__ == '__main__':
    run_all_sensitivity()
