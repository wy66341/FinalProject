"""O4: Multi-flyby trajectory — Earth→Moon→Venus→Sun→Earth.

Patched-conic chain: uses Moon and Venus gravity assists to
reach smaller perihelia with lower total Δv.

Segments:
  1. Earth→Moon (lunar gravity assist)
  2. Moon→Venus (heliocentric transfer inward)
  3. Venus flyby (gravity assist for deeper penetration)
  4. Venus→Sun (perihelion)
  5. Sun→Earth (return to Earth)
"""

import numpy as np
from numpy.linalg import norm

from conic_patch import (
    AU, MU_SUN, MU_EARTH, MU_MOON, DAY,
    helio_ellipse, earth_departure, earth_arrival, lunar_swingby,
)

# Venus orbital parameters
R_VENUS = 0.723332 * AU       # semi-major axis (km)
MU_VENUS = 3.24859e5           # km^3/s^2 (GM)
R_VENUS_RADIUS = 6051.8        # km
R_VENUS_SOI = 6.16e5           # km (approximate)


def _hohmann_v_inf(r_depart, r_arrive, mu=MU_SUN):
    """Compute v_inf for Hohmann transfer between two circular orbits.

    Returns departure and arrival v_inf (km/s).
    Positive = need to speed up; negative = need to slow down.
    """
    a_trans = (r_depart + r_arrive) / 2.0
    v_circ_dep = np.sqrt(mu / r_depart)
    v_circ_arr = np.sqrt(mu / r_arrive)
    v_trans_dep = np.sqrt(mu * (2.0 / r_depart - 1.0 / a_trans))
    v_trans_arr = np.sqrt(mu * (2.0 / r_arrive - 1.0 / a_trans))
    return v_trans_dep - v_circ_dep, v_trans_arr - v_circ_arr


def solve_multi_flyby(r_p_sun, r_m_moon=5000.0, r_p_venus=10000.0,
                       date_str='2026-01-07', verbose=True):
    """Compute Δv for Earth→Moon→Venus→Sun→Earth trajectory.

    The lunar flyby provides initial boost. Venus flyby enables
    deeper Sun penetration.

    Returns dict with Δv breakdown and trajectory parameters.
    """
    from trajectory import get_ephemeris

    ep, ev, mp, mv = get_ephemeris(date_str)
    r_1 = norm(ep)
    v_earth = norm(ev)

    # ── Leg 1: Earth→Moon (lunar flyby) ──
    # Rocket must go to Moon for gravity assist
    # Moon-relative v_inf based on escape from Earth toward Moon
    v_moon_orbit = np.sqrt(MU_EARTH / norm(mp - ep))  # actual Moon orbital speed
    # Approximate: v_inf needed to reach Moon from LEO
    r_leo = 6378.137 + 200.0
    a_trans_em = (r_leo + norm(mp - ep)) / 2.0
    v_peri_em = np.sqrt(2.0 * MU_EARTH / r_leo - MU_EARTH / a_trans_em)
    v_circ_leo = np.sqrt(MU_EARTH / r_leo)
    dv_tli = v_peri_em - v_circ_leo  # trans-lunar injection

    # v_inf at Moon: rocket Earth-relative speed at arrival
    v_rocket_earth_at_moon = np.sqrt(v_peri_em**2 -
                                      2 * MU_EARTH * (1/r_leo - 1/norm(mp-ep)))
    v_inf_moon_in = abs(v_rocket_earth_at_moon - v_moon_orbit)
    v_inf_moon_in = max(v_inf_moon_in, 0.5)

    # Lunar flyby
    swingby_moon = lunar_swingby(v_inf_moon_in, r_m_moon, 'trailing')

    # After lunar flyby: rocket's heliocentric velocity
    v_rocket_helio_after_moon = norm(mv) + v_inf_moon_in  # approximate max gain

    # ── Leg 2: Moon→Venus transfer ──
    v_inf_venus_dep, v_inf_venus_arr = _hohmann_v_inf(r_1, R_VENUS)

    # The rocket needs v_inf_venus_dep relative to Earth
    # After Moon flyby, it has some v_inf relative to Earth
    v_inf_from_moon = v_rocket_helio_after_moon - v_earth
    # Additional Δv needed (if any)
    dv_venus_transfer = max(0.0, v_inf_venus_dep - v_inf_from_moon)

    # ── Leg 3: Venus flyby ──
    # Arrive at Venus with relative velocity
    v_venus_orbit = np.sqrt(MU_SUN / R_VENUS)
    v_inf_venus = abs(v_inf_venus_arr) + v_inf_from_moon  # approximate
    v_inf_venus = max(v_inf_venus, 0.5)

    # Venus flyby parameters
    a_h_venus = MU_VENUS / v_inf_venus**2
    e_h_venus = 1.0 + r_p_venus / a_h_venus
    if e_h_venus > 1.0:
        delta_venus = 2.0 * np.arcsin(1.0 / e_h_venus)
    else:
        delta_venus = 0.0

    # Gain from Venus flyby
    v_gain_venus = 2.0 * v_inf_venus * np.sin(delta_venus / 2.0)
    v_rocket_helio_after_venus = v_venus_orbit + v_gain_venus

    # ── Leg 4: Venus→Sun (perihelion) ──
    # After Venus flyby, rocket has enough velocity to reach r_p_sun
    ell_venus_sun = helio_ellipse(r_p_sun, R_VENUS)
    v_needed_at_venus = np.sqrt(MU_SUN * (2.0 / R_VENUS - 1.0 / ell_venus_sun['a']))
    dv_venus_deep = max(0.0, v_needed_at_venus - v_rocket_helio_after_venus)

    # ── Leg 5: Sun→Earth return ──
    ell_return = helio_ellipse(r_p_sun, R_VENUS)
    v_arrive_earth = ell_return['v_a']

    # Capture at Earth
    earth_dep_v_inf = abs(v_arrive_earth - v_earth)
    dv_earth_capture = earth_arrival(earth_dep_v_inf)

    # ── Δv Budget ──
    dv_total_multi = dv_tli + dv_venus_transfer + dv_venus_deep + dv_earth_capture

    # Compare with Moon-only baseline
    from trajectory import solve_single_date
    moon_only = solve_single_date(date_str, r_p_sun, r_m_moon, 'trailing',
                                   verbose=False)

    if verbose:
        print('=' * 60)
        print('MULTI-FLYBY: Earth→Moon→Venus→Sun→Earth')
        print('=' * 60)
        print(f'  Launch date: {date_str}')
        print(f'  Target r_p:   {r_p_sun/AU:.4f} AU '
              f'({r_p_sun/R_VENUS:.2f} × Venus orbit)')
        print()
        print(f'  Leg 1 (TLI to Moon):')
        print(f'    Δv = {dv_tli:.3f} km/s  (200km LEO → lunar transfer)')
        print(f'    v_inf at Moon: {v_inf_moon_in:.1f} km/s')
        print(f'    Flyby deflection: {swingby_moon["delta_deg"]:.1f} deg')
        print()
        print(f'  Leg 2 (Moon→Venus transfer):')
        print(f'    v_inf from Moon assist: {v_inf_from_moon:.1f} km/s')
        print(f'    Required v_inf (Hohmann): {v_inf_venus_dep:.1f} km/s')
        print(f'    Δv correction: {dv_venus_transfer:.3f} km/s')
        print()
        print(f'  Leg 3 (Venus flyby):')
        print(f'    v_inf at Venus: {v_inf_venus:.1f} km/s')
        print(f'    Deflection: {np.degrees(delta_venus):.1f} deg')
        print(f'    Velocity gain: {v_gain_venus:.1f} km/s')
        print()
        print(f'  Leg 4 (Venus→Sun):')
        print(f'    Δv for deep penetration: {dv_venus_deep:.3f} km/s')
        print()
        print(f'  Leg 5 (Sun→Earth return):')
        print(f'    Earth capture Δv: {dv_earth_capture:.3f} km/s')
        print()
        print(f'  Multi-flyby total Δv:  {dv_total_multi:.3f} km/s')
        print(f'  Moon-only total Δv:    {moon_only["Delta_v_total"]:.3f} km/s')
        saving = moon_only['Delta_v_total'] - dv_total_multi
        print(f'  Savings vs Moon-only:  {saving:+.3f} km/s '
              f'({saving/moon_only["Delta_v_total"]*100:.1f}%)')

    return {
        'Delta_v_total': dv_total_multi,
        'Delta_v_tli': dv_tli,
        'Delta_v_venus_transfer': dv_venus_transfer,
        'Delta_v_venus_flyby_gain': v_gain_venus,
        'Delta_v_venus_deep': dv_venus_deep,
        'Delta_v_earth_capture': dv_earth_capture,
        'v_inf_moon': v_inf_moon_in,
        'v_inf_venus': v_inf_venus,
        'delta_moon_deg': swingby_moon['delta_deg'],
        'delta_venus_deg': np.degrees(delta_venus),
        'moon_only_dv': moon_only['Delta_v_total'],
        'saving_vs_moon_only': moon_only['Delta_v_total'] - dv_total_multi,
        'can_reach_venus': v_inf_from_moon > v_inf_venus_dep * 0.5,
        'perihelion_reachable': dv_venus_deep < 10.0,
    }


def compare_moon_only_vs_multi(r_p_values=None):
    """Compare Moon-only vs multi-flyby Δv across a range of r_p."""
    if r_p_values is None:
        r_p_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

    print('=' * 60)
    print('Moon-only vs Multi-flyby Comparison')
    print('=' * 60)
    print(f'  {"r_p (AU)":>10s}  {"Moon-only":>12s}  {"Multi-flyby":>12s}  '
          f'{"Saving":>10s}')
    print(f'  {"-"*10}  {"-"*12}  {"-"*12}  {"-"*10}')

    for rp_au in r_p_values:
        rp = rp_au * AU
        moon = solve_multi_flyby(rp, verbose=False)
        if moon['can_reach_venus']:
            print(f'  {rp_au:10.3f}  {moon["moon_only_dv"]:12.3f}  '
                  f'{moon["Delta_v_total"]:12.3f}  '
                  f'{moon["saving_vs_moon_only"]:+10.3f}')
        else:
            print(f'  {rp_au:10.3f}  {moon["moon_only_dv"]:12.3f}  '
                  f'{"N/A":>12s}  {"—":>10s}')

    print()


if __name__ == '__main__':
    solve_multi_flyby(0.1 * AU)
    print()
    compare_moon_only_vs_multi()
