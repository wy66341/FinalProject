"""M6: 一年周期最优发射窗口扫描.

365 天外层扫描 + 内层 (r_m, r_p) 参数网格，
输出最优解和 Delta_v_total(t0) 曲线数据。
"""

import numpy as np
from datetime import datetime, timedelta
import json
import os

from conic_patch import AU, R_SUN, R_MOON
from trajectory import solve_single_date

DAY = 86400.0

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def scan_launch_window(
    year=2026,
    r_p_range=(None, None, None),
    r_m_fixed=5000.0,
    side_fixed='trailing',
    use_lunar=True,
    verbose=True,
    save=True,
):
    """365-day launch window optimization.

    Parameters
    ----------
    year : int
    r_p_range : tuple (min, max, n) — defaults to (2*R_SUN, 0.4*AU, 80)
    r_m_fixed : float — fixed pericynthion (km)
    side_fixed : str
    use_lunar : bool
    verbose : bool
    save : bool — save scan results to data/

    Returns
    -------
    dict with best_t0, best_r_p, best_dv, all_results
    """
    r_p_min, r_p_max, n_rp = r_p_range
    if r_p_min is None:
        r_p_min = 2.0 * R_SUN
    if r_p_max is None:
        r_p_max = 0.4 * AU
    if n_rp is None:
        n_rp = 80

    r_p_grid = np.linspace(r_p_min, r_p_max, n_rp)

    best_dv = np.inf
    best_entry = None
    all_results = []

    start_date = datetime(year, 1, 1)
    total = 365 * n_rp

    for day_idx in range(365):
        t0 = start_date + timedelta(days=day_idx)
        date_str = t0.strftime('%Y-%m-%d')

        for r_p in r_p_grid:
            result = solve_single_date(
                date_str, r_p,
                r_m=r_m_fixed if use_lunar else None,
                side=side_fixed if use_lunar else None,
                use_lunar=use_lunar,
                verbose=False,
            )

            entry = {
                'day': day_idx,
                'date': date_str,
                'r_p_au': r_p / AU,
                'r_p_km': r_p,
                'Delta_v_total': result['Delta_v_total'],
                'Delta_v_launch': result['Delta_v_launch'],
                'Delta_v_reentry': result['Delta_v_reentry'],
                'T_years': result['ellipse']['T_years'],
                'e': result['ellipse']['e'],
                'all_ok': result['all_ok'],
            }

            all_results.append(entry)

            if result['all_ok'] and result['Delta_v_total'] < best_dv:
                best_dv = result['Delta_v_total']
                best_entry = entry

        if verbose and day_idx % 30 == 0:
            current_best = f'{best_dv:.3f}' if best_entry else 'N/A'
            print(f'  Day {day_idx:3d} ({date_str}): '
                  f'best so far = {current_best} km/s')

    if verbose:
        print()
        if best_entry:
            print('=== OPTIMAL SOLUTION ===')
            print(f'  Launch date:    {best_entry["date"]}')
            print(f'  r_p:             {best_entry["r_p_au"]:.4f} AU')
            print(f'  r_m:             {r_m_fixed:.0f} km')
            print(f'  Side:            {side_fixed}')
            print(f'  Δv_total:        {best_entry["Delta_v_total"]:.3f} km/s')
            print(f'  Δv_launch:       {best_entry["Delta_v_launch"]:.3f} km/s')
            print(f'  Δv_reentry:      {best_entry["Delta_v_reentry"]:.3f} km/s')
            print(f'  Flight time:     {best_entry["T_years"]:.2f} yr')
            print(f'  Eccentricity:    {best_entry["e"]:.4f}')
        else:
            print('No feasible solution found.')

    # Save results
    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        output = {
            'best': best_entry,
            'r_m': r_m_fixed,
            'side': side_fixed,
            'use_lunar': use_lunar,
            'n_results': len(all_results),
            'results': all_results,
        }
        path = os.path.join(DATA_DIR, 'scan_results.json')
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f'\nResults saved to {path}')

    return {
        'best': best_entry,
        'best_dv': best_dv,
        'r_m': r_m_fixed,
        'side': side_fixed,
        'all_results': all_results,
    }


if __name__ == '__main__':
    scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 60))
