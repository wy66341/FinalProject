"""M6: 一年周期最优发射窗口扫描.

在 2026 年 365 个日期上扫描，对每个日期做内层参数精化，
输出 Delta_v_total(t0) 曲线与 (t0, r_p) 等值线，
报告最优发射日期和最小 Delta_v_total.
"""

import numpy as np
from datetime import datetime, timedelta

from conic_patch import AU, R_SUN, MU_SUN
from trajectory import solve_fixed_date

DAY = 86400.0


def scan_launch_window(
    year=2026,
    r_p_range=None,
    r_m_fixed=5000.0,
    side_fixed='trailing',
    verbose=True
):
    """365 天扫描 + 内层 r_p 参数网格.

    Parameters
    ----------
    year : int
        扫描年份
    r_p_range : tuple (min, max, n)
        r_p 扫描范围，默认 [2*R_SUN, 0.4*AU, 50]
    r_m_fixed : float
        固定近月距 (km)
    side_fixed : str
        固定绕月方向
    verbose : bool

    Returns
    -------
    dict with keys: best_t0, best_r_p, best_Delta_v, scan_results
    """
    if r_p_range is None:
        r_p_min = 2 * R_SUN
        r_p_max = 0.4 * AU
        n_rp = 50
    else:
        r_p_min, r_p_max, n_rp = r_p_range

    r_p_grid = np.linspace(r_p_min, r_p_max, n_rp)

    best_Delta_v = np.inf
    best_t0 = None
    best_r_p = None
    scan_results = []

    start_date = datetime(year, 1, 1)

    for day in range(365):
        t0 = start_date + timedelta(days=day)
        t0_str = t0.strftime('%Y-%m-%d')

        for r_p in r_p_grid:
            result = solve_fixed_date(r_p, r_m_fixed, side_fixed)

            Delta_v = result['Delta_v_total']
            T_total = result['ellipse']['T']

            # 约束检查
            if r_p <= R_SUN:
                continue
            if T_total > 2 * 365.25 * DAY:
                continue
            if result['Delta_v_reentry'] > 15.0:
                Delta_v = np.inf

            scan_results.append({
                't0': t0_str,
                'day': day,
                'r_p': r_p,
                'r_p_au': r_p / AU,
                'Delta_v': Delta_v,
                'T_total_days': T_total / DAY,
                'e': result['ellipse']['e'],
            })

            if Delta_v < best_Delta_v:
                best_Delta_v = Delta_v
                best_t0 = t0_str
                best_r_p = r_p

        if verbose and day % 30 == 0:
            print(f'  Day {day:3d} ({t0_str}): best so far = '
                  f'{best_Delta_v:.3f} km/s @ r_p = {best_r_p/AU:.4f} AU')

    if verbose:
        print(f'\n=== 最优解 ===')
        print(f'发射日期: {best_t0}')
        print(f'近日距:   {best_r_p/AU:.4f} AU ({best_r_p:.2e} km)')
        print(f'近月距:   {r_m_fixed:.0f} km')
        print(f'绕月方向: {side_fixed}')
        print(f'最小 Delta_v_total: {best_Delta_v:.3f} km/s')

    return {
        'best_t0': best_t0,
        'best_r_p': best_r_p,
        'best_Delta_v': best_Delta_v,
        'r_m': r_m_fixed,
        'side': side_fixed,
        'scan_results': scan_results,
    }


if __name__ == '__main__':
    result = scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 30))
