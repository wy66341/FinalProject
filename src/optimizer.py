"""M6: 一年周期最优发射窗口扫描.

365 天外层扫描 + 内层 (r_m, r_p) 参数网格，
输出最优解和 Delta_v_total(t0) 曲线数据。

每个日期使用真实月球相位（通过 solve_single_date 的实时历表查询），
扫描结束后对最优候选进行 N-体全轨道验证（规则 15/16/18）。
"""

import numpy as np
from datetime import datetime, timedelta
import json
import os

from conic_patch import AU, R_SUN, R_MOON, DAY as _DAY
from trajectory import solve_single_date, verify_full_mission


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def scan_launch_window(
    year=2026,
    r_p_range=(None, None, None),
    r_m_fixed=5000.0,
    side_fixed='trailing',
    use_lunar=True,
    verbose=True,
    save=True,
    validate_top_n=3,
):
    """365-day launch window optimization with N-body validation.

    Parameters
    ----------
    year : int
    r_p_range : tuple (min, max, n) — defaults to (2*R_SUN, 0.4*AU, 80)
    r_m_fixed : float — fixed pericynthion (km)
    side_fixed : str
    use_lunar : bool
    verbose : bool
    save : bool — save scan results to data/
    validate_top_n : int — number of top candidates to validate with N-body

    Returns
    -------
    dict with best, best_dv, r_m, side, all_results, validation
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

    if verbose:
        print(f'Scanning {365} days × {n_rp} r_p grid points = {total} total')
        print(f'  r_p range: {r_p_min/AU:.4f} – {r_p_max/AU:.4f} AU')
        print(f'  r_m: {r_m_fixed:.0f} km  side: {side_fixed}  '
              f'lunar: {use_lunar}')
        print()

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

            # Compute Earth return date and phase
            flight_days = result['ellipse']['T_years'] * 365.25
            return_date = t0 + timedelta(days=flight_days)
            return_date_str = return_date.strftime('%Y-%m-%d')

            # Get Earth position at return date for phase check
            try:
                from trajectory import get_ephemeris
                ep_ret, _, _, _ = get_ephemeris(return_date_str)
                r_1_return = np.linalg.norm(ep_ret)
            except Exception:
                r_1_return = np.linalg.norm(result.get('ellipse', {}).get('r_1', AU))

            entry = {
                'day': day_idx,
                'date': date_str,
                'r_p_au': r_p / AU,
                'r_p_km': r_p,
                'Delta_v_total': result['Delta_v_total'],
                'Delta_v_launch': result['Delta_v_launch'],
                'Delta_v_lunar_residual': result['Delta_v_lunar_residual'],
                'Delta_v_reentry': result['Delta_v_reentry'],
                'Delta_v_total_nomoon': result.get('Delta_v_total_nomoon',
                                                    result['Delta_v_launch'] + result['Delta_v_reentry']),
                'em_closure': result.get('em_closure', 0.0),
                'flyby_deficit': result.get('flyby_deficit', 0.0),
                'helio_splice': result.get('helio_splice', 0.0),
                'return_rendezvous': result.get('return_rendezvous', 0.0),
                'T_years': result['ellipse']['T_years'],
                'e': result['ellipse']['e'],
                'all_ok': result['all_ok'],
                # Moon phase data (from real ephemeris)
                'moon_phase_angle_deg': result.get('moon_phase_angle_deg'),
                'moon_phase_ok': result.get('moon_phase_ok', False),
                # Return Earth data
                'return_date': return_date_str,
                'r_1_return_km': r_1_return,
                'r_1_departure_km': result['ellipse']['r_1'],
                # Flight dynamics
                'v_inf_moon_kms': result.get('v_inf_moon_kms', 0.0),
                'can_reach_moon': result.get('can_reach_moon', False),
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
            print('=== ANALYTIC OPTIMAL SOLUTION ===')
            print(f'  Launch date:    {best_entry["date"]}')
            print(f'  r_p:             {best_entry["r_p_au"]:.4f} AU')
            print(f'  r_m:             {r_m_fixed:.0f} km')
            print(f'  Side:            {side_fixed}')
            print(f'  Moon phase:      {best_entry.get("moon_phase_angle_deg", "N/A")}')
            print(f'  Δv_total:        {best_entry["Delta_v_total"]:.3f} km/s')
            print(f'  Δv_launch:       {best_entry["Delta_v_launch"]:.3f} km/s')
            if best_entry.get('Delta_v_lunar_residual', 0) > 0:
                print(f'  Δv_lunar_residual: {best_entry["Delta_v_lunar_residual"]:.3f} km/s')
            print(f'  Δv_reentry:      {best_entry["Delta_v_reentry"]:.3f} km/s')
            print(f'  Flight time:     {best_entry["T_years"]:.2f} yr')
            print(f'  Eccentricity:    {best_entry["e"]:.4f}')
            print(f'  Return date:     {best_entry["return_date"]}')
        else:
            print('No feasible solution found.')

    # ── N-body validation of top candidates ──
    validation = None
    if validate_top_n > 0 and use_lunar:
        # Get top N feasible candidates sorted by Δv
        feasible = sorted(
            [e for e in all_results if e['all_ok']],
            key=lambda x: x['Delta_v_total']
        )[:validate_top_n]

        if feasible and verbose:
            print(f'\n=== N-BODY VALIDATION (top {len(feasible)} candidates) ===')

        validation_results = []
        for rank, entry in enumerate(feasible):
            r_p_km = entry['r_p_km']
            date = entry['date']

            if verbose:
                print(f'\n  Candidate #{rank+1}: {date}  r_p={r_p_km/AU:.4f} AU  '
                      f'Δv={entry["Delta_v_total"]:.3f} km/s')

            mission = verify_full_mission(
                date, r_p_km, r_m_fixed, side_fixed
            )

            v_result = {
                'rank': rank + 1,
                'date': date,
                'r_p_au': r_p_km / AU,
                'Delta_v_total': entry['Delta_v_total'],
                'rule15_pass': mission.get('rule15_pass', False),
                'rule16_pass': mission.get('rule16_pass', False),
                'rule18_pass': mission.get('perihelion_ok', False),
                'moon_approach_km': mission.get('moon_closest_approach_km', None),
                'earth_approach_km': mission.get('earth_closest_approach_km', None),
                'perihelion_km': mission.get('perihelion_km', None),
                'all_rules_pass': (
                    mission.get('rule15_pass', False) and
                    mission.get('rule16_pass', False) and
                    mission.get('perihelion_ok', False)
                ),
            }
            validation_results.append(v_result)

            if verbose:
                print(f'    R15 (Moon SOI):    {"PASS" if v_result["rule15_pass"] else "FAIL"}  '
                      f'(closest={v_result["moon_approach_km"]:,.0f} km)')
                print(f'    R16 (Earth return): {"PASS" if v_result["rule16_pass"] else "FAIL"}  '
                      f'(closest={v_result["earth_approach_km"]:,.0f} km)')
                print(f'    R18 (perihelion):   {"PASS" if v_result["rule18_pass"] else "FAIL"}  '
                      f'(r_min={v_result["perihelion_km"]:,.0f} km)')
                print(f'    Overall: {"PASS" if v_result["all_rules_pass"] else "FAIL"}')

        # Find the best validated candidate
        validated = [v for v in validation_results if v['all_rules_pass']]
        if validated:
            best_validated = validated[0]  # already sorted by Δv
            validation = {
                'validated_optimal': best_validated,
                'all_validations': validation_results,
                'n_validated_passing': len(validated),
                'n_tested': len(validation_results),
            }
            if verbose:
                print(f'\n  Best validated: {best_validated["date"]}  '
                      f'r_p={best_validated["r_p_au"]:.4f} AU  '
                      f'Δv={best_validated["Delta_v_total"]:.3f} km/s')
        else:
            validation = {
                'validated_optimal': None,
                'all_validations': validation_results,
                'n_validated_passing': 0,
                'n_tested': len(validation_results),
            }
            if verbose:
                print(f'\n  No candidate passed all N-body checks.')

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
        if validation:
            output['validation'] = validation
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
        'validation': validation,
    }


if __name__ == '__main__':
    scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 40), validate_top_n=5)
