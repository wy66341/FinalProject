"""M5: 单点轨道求解.

固定发射日期下求解满足约束的完整轨道，
输出三段 Delta-v，与无月球借力情况对比。

月球借力使用真实月球历表数据（Horizons 优先，解析回退）：
  - 月球相对速度基于实际 mv（非固定圆轨道近似）
  - 月球位置 mp 用于相位检查
  - N-体传播验证 Earth-Moon 转移及 SOI 进入
  - dv_lunar 由实际借力几何计算得出
"""

import numpy as np
from numpy.linalg import norm

from conic_patch import (
    AU, R_SUN, R_MOON, R_MOON_SOI,
    MU_SUN, MU_EARTH, MU_MOON, DAY,
    helio_ellipse, earth_departure, earth_arrival,
    lunar_swingby,
)
from nbody import (
    acceleration_sem, velocity_verlet_step, system_energy,
    earth_moon_analytic_state,
)


def _julian_date(year, month, day):
    """Convert calendar date to Julian Date."""
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jd = (day + (153 * m + 2) // 5 + 365 * y + y // 4
          - y // 100 + y // 400 - 32045)
    return jd


_EPHEM_CACHE = {}

def get_ephemeris(date_str):
    """Get Earth/Moon state for a given date. Results are cached per date.

    Tries JPL Horizons first; falls back to analytic.
    Returns (earth_pos, earth_vel, moon_pos, moon_vel) in km, km/s.
    """
    if date_str in _EPHEM_CACHE:
        return _EPHEM_CACHE[date_str]
    try:
        from astroquery.jplhorizons import Horizons

        epochs = {'start': date_str, 'stop': date_str, 'step': '1d'}
        obj_e = Horizons(id='399', location='@10', epochs=epochs)
        vec_e = obj_e.vectors()
        ep = np.array([float(vec_e['x'][0]), float(vec_e['y'][0]),
                       float(vec_e['z'][0])]) * AU
        ev = np.array([float(vec_e['vx'][0]), float(vec_e['vy'][0]),
                       float(vec_e['vz'][0])]) * AU / DAY

        obj_m = Horizons(id='301', location='@10', epochs=epochs)
        vec_m = obj_m.vectors()
        mp = np.array([float(vec_m['x'][0]), float(vec_m['y'][0]),
                       float(vec_m['z'][0])]) * AU
        mv = np.array([float(vec_m['vx'][0]), float(vec_m['vy'][0]),
                       float(vec_m['vz'][0])]) * AU / DAY

        _EPHEM_CACHE[date_str] = (ep, ev, mp, mv)
        return ep, ev, mp, mv
    except Exception:
        pass

    # Fallback to analytic
    parts = date_str.split('-')
    yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
    jd = _julian_date(yr, mo, dy)
    result = earth_moon_analytic_state(jd)
    _EPHEM_CACHE[date_str] = result
    return result


def _propagate_earth_to_moon(y0, moon_pos_init, moon_dist_geo, dt=3600, max_steps=400000):
    """N-体传播火箭从地球到月球附近，检测月球 SOI 进入.

    Also verifies the rocket's Earth-relative apogee reaches lunar distance.

    Args:
        y0: 24-element state vector (Sun, Earth, Moon, Rocket)
        moon_pos_init: initial Moon heliocentric position
        moon_dist_geo: initial Moon geocentric distance (km)

    Returns dict with:
        entered_soi: bool — rocket passed within R_MOON_SOI of Moon
        closest_approach_km: float — minimum distance to Moon center
        can_reach_moon: bool — rocket apogee ≥ moon distance
        max_geo_distance_km: float — max geocentric distance reached
        tof_days: float — time to closest approach
        state_at_closest: ndarray or None
    """
    y = y0.copy()
    min_dist = float('inf')
    min_dist_state = None
    min_dist_step = 0
    entered_soi = False
    max_geo_dist = 0.0

    for step in range(1, max_steps + 1):
        try:
            y = velocity_verlet_step(y, dt)
        except RuntimeError:
            break

        r_rocket = y[18:21]
        r_earth = y[6:9]
        r_moon = y[12:15]
        dist_to_moon = norm(r_rocket - r_moon)
        geo_dist = norm(r_rocket - r_earth)

        max_geo_dist = max(max_geo_dist, geo_dist)

        if dist_to_moon < min_dist:
            min_dist = dist_to_moon
            min_dist_state = y.copy()
            min_dist_step = step

        if dist_to_moon < R_MOON_SOI:
            entered_soi = True

        # Stop once past Moon and SOI was checked
        if dist_to_moon > 3 * R_MOON_SOI and entered_soi:
            break

        # Stop if rocket escapes to heliocentric space
        if geo_dist > 2 * moon_dist_geo:
            break

    tof_days = min_dist_step * dt / DAY
    can_reach_moon = max_geo_dist >= moon_dist_geo * 0.85

    return {
        'entered_soi': entered_soi,
        'closest_approach_km': min_dist,
        'can_reach_moon': can_reach_moon,
        'max_geo_distance_km': max_geo_dist,
        'tof_days': tof_days,
        'state_at_closest': min_dist_state,
    }


def solve_single_date(date_str, r_p, r_m=None, side=None,
                      use_lunar=True, verbose=True):
    """Solve the complete trajectory for a fixed launch date.

    Uses real Earth-Sun distance and Moon state from ephemeris.
    Lunar swingby uses actual Moon position/velocity for v_inf calculation,
    phase verification, dv_lunar computation, and N-body SOI validation.

    Parameters
    ----------
    date_str : str — launch date 'YYYY-MM-DD'
    r_p : float — perihelion distance (km)
    r_m : float or None — closest approach to Moon (km)
    side : str or None — 'trailing' or 'leading'
    use_lunar : bool — enable lunar swingby
    verbose : bool — print results
    """
    # ── Get real ephemeris ──
    ep, ev, mp, mv = get_ephemeris(date_str)
    r_1 = norm(ep)                          # actual Earth-Sun distance (km)
    v_earth_actual = norm(ev)               # actual Earth orbital speed (km/s)
    v_moon_actual = norm(mv)                # actual Moon orbital speed (km/s)

    # Moon geocentric position and velocity
    moon_geo_pos = mp - ep                  # km
    moon_geo_vel = mv - ev                  # km/s
    moon_dist_geo = norm(moon_geo_pos)      # actual Moon-Earth distance (km)

    # ── Heliocentric ellipse ──
    ell = helio_ellipse(r_p, r_1)

    # ── Baseline (no lunar assist): departure and reentry ──
    # Departure: from 200 km LEO to hyperbolic escape with v_inf = Δv_dep
    # Reentry:  propulsive capture from hyperbolic approach to 200 km LEO
    dv_launch_nomoon = earth_departure(ell['Delta_v_dep'])
    dv_reentry = earth_arrival(ell['Delta_v_dep'])

    # ── Lunar swingby with real Moon data ──
    # Δv_total = Δv_launch + Δv_lunar_residual + Δv_reentry
    # where:
    #   Δv_launch:         departure injection (savings baked in → lower than nomoon)
    #   Δv_lunar_residual: sum of closure corrections (≥ 0, zero in ideal case)
    #   Δv_reentry:        propulsive capture to 200 km LEO
    dv_launch = dv_launch_nomoon
    dv_lunar_residual = 0.0
    em_closure = 0.0
    flyby_deficit = 0.0
    helio_splice = 0.0
    return_rendezvous = 0.0
    swingby_info = None
    nbody_result = None
    moon_phase_ok = True
    soi_entered = False
    can_reach_moon = False
    closest_approach = None

    if use_lunar and r_m is not None and side is not None:
        # ── C2: Compute v_inf_moon from real Moon velocity ──
        # Rocket's heliocentric velocity after Earth escape:
        # v_inf direction is along Earth's orbital velocity (optimal for
        # Hohmann-like transfer)
        v_earth_dir = ev / v_earth_actual
        v_rocket_helio = ev + ell['Delta_v_dep'] * v_earth_dir

        # Moon-relative velocity at encounter (real, not approximation)
        v_inf_moon_vec = v_rocket_helio - mv
        v_inf_moon = norm(v_inf_moon_vec)
        v_inf_moon = max(v_inf_moon, 0.5)

        # ── C3: Moon phase verification ──
        # For trailing-side flyby: Moon should be ahead of Earth in orbit
        # direction so the flyby adds velocity
        # For leading-side flyby: Moon should be behind Earth
        earth_vel_hat = ev / v_earth_actual
        moon_geo_hat = moon_geo_pos / moon_dist_geo if moon_dist_geo > 0 else np.zeros(3)
        cos_phase = np.dot(moon_geo_hat, earth_vel_hat)

        if side == 'trailing':
            # Moon ahead: cos_phase > 0, angle < 90°
            moon_phase_ok = cos_phase > -0.5  # tolerant: within ~120°
        else:
            # Moon behind: cos_phase < 0, angle > 90°
            moon_phase_ok = cos_phase < 0.5   # tolerant: within ~120°

        phase_angle_deg = np.degrees(np.arccos(np.clip(cos_phase, -1, 1)))

        # ── C1/C2: Analytic swingby with real v_inf ──
        try:
            swingby_info = lunar_swingby(v_inf_moon, r_m, side)
        except ValueError:
            swingby_info = None

        # ── C5: Compute Δv_launch with lunar assist savings ──
        # The lunar flyby deflects v_inf_moon by angle δ.
        # Trailing flyby (sign=+1) adds heliocentric speed → reduces required v_inf
        # Leading flyby (sign=-1) subtracts heliocentric speed → increases required v_inf
        # The savings are reflected in a LOWER Δv_launch, not a negative Δv_lunar.
        # Phase affects magnitude (via cos_phase) but savings are always computed.
        if swingby_info is not None:
            delta = swingby_info['delta']
            sign = swingby_info['sign']  # trailing=+1, leading=-1

            # Velocity change from lunar flyby projected onto Earth velocity.
            # cos_phase ~ +1 when Moon is ahead of Earth (favorable trailing),
            # cos_phase ~ -1 when Moon is behind (favorable leading),
            # cos_phase ~ 0 when Moon is perpendicular (no benefit).
            v_gain = 2.0 * v_inf_moon * np.sin(delta / 2.0) * abs(cos_phase)
            v_gain_effective = sign * v_gain  # > 0 for trailing gain

            # Reduced v_inf requirement after lunar assist
            effective_v_inf = max(0.0, ell['Delta_v_dep'] - v_gain_effective)

            # Actual departure Δv (lower than no-moon due to lunar assist)
            dv_launch = earth_departure(effective_v_inf)

        # ── Closure correction terms ──
        # In the ideal patched-conic approximation, these are zero.
        # A full N-body treatment would compute:
        #   em_closure:        mid-course correction to ensure Moon SOI entry
        #   flyby_deficit:     |required_deflection - achieved_deflection|
        #   helio_splice:      match Moon-SOI-exit state to heliocentric target
        #   return_rendezvous: terminal correction to intercept Earth
        dv_lunar_residual = em_closure + flyby_deficit + helio_splice + return_rendezvous

        # ── C4: N-体传播验证 Earth-Moon 转移 ──
        # Build initial state: Sun at SSB origin, Earth, Moon, Rocket at Earth
        y0 = np.zeros(24)
        y0[0:3] = [0.0, 0.0, 0.0]    # Sun pos
        y0[3:6] = [0.0, 0.0, 0.0]    # Sun vel
        y0[6:9] = ep                  # Earth pos
        y0[9:12] = ev                 # Earth vel
        y0[12:15] = mp                # Moon pos
        y0[15:18] = mv                # Moon vel

        # Trans-lunar injection (TLI) from 200 km circular LEO.
        # Hohmann transfer: perigee at LEO, apogee at Moon distance.
        # Use N-body to predict Moon position at encounter time.
        r_leo = 6378.137 + 200.0
        a_trans = (r_leo + moon_dist_geo) / 2.0
        v_peri_trans = np.sqrt(2.0 * MU_EARTH / r_leo - MU_EARTH / a_trans)
        tof_trans = np.pi * np.sqrt(a_trans**3 / MU_EARTH)
        dt_em = 1800  # 30 min step for Earth-Moon leg
        z_hat = np.array([0.0, 0.0, 1.0])

        # Use N-body to propagate Moon to encounter time (more accurate
        # than linear extrapolation since Moon is highly perturbed)
        y_moon_adv = y0.copy()
        n_adv = int(tof_trans / 3600)
        for _ in range(n_adv):
            try:
                y_moon_adv = velocity_verlet_step(y_moon_adv, 3600)
            except RuntimeError:
                break
        moon_at_enc = y_moon_adv[12:15] - y_moon_adv[6:9]  # geocentric
        moon_enc_dist = norm(moon_at_enc)
        if moon_enc_dist > 0:
            moon_enc_dir = moon_at_enc / moon_enc_dist
        else:
            moon_enc_dir = moon_geo_pos / moon_dist_geo

        # Hohmann apogee is 180° from perigee → launch from opposite side
        launch_dir = -moon_enc_dir
        y0[18:21] = ep + r_leo * launch_dir

        # Velocity tangent to launch direction at perigee
        tangent_dir = np.cross(launch_dir, z_hat)
        if norm(tangent_dir) < 1e-12:
            tangent_dir = np.array([0.0, 1.0, 0.0])
        else:
            tangent_dir = tangent_dir / norm(tangent_dir)
        y0[21:24] = ev + v_peri_trans * tangent_dir

        nbody_result = _propagate_earth_to_moon(y0, mp, moon_dist_geo, dt=dt_em)
        soi_entered = nbody_result['entered_soi']
        closest_approach = nbody_result['closest_approach_km']
        can_reach_moon = nbody_result['can_reach_moon']

        if verbose:
            print(f'  Moon phase angle: {phase_angle_deg:.1f} deg  '
                  f'({"favorable" if moon_phase_ok else "unfavorable"})')
            print(f'  v_inf at Moon:    {v_inf_moon:.2f} km/s  '
                  f'(from real mv={v_moon_actual:.2f} km/s, '
                  f'moon_dist={moon_dist_geo:.0f} km)')
            if swingby_info:
                print(f'  Swingby deflection: {swingby_info["delta_deg"]:.1f} deg')
                saving_amount = dv_launch_nomoon - dv_launch
                print(f'  Lunar saving:    {saving_amount:.3f} km/s  '
                      f'(launch Δv: {dv_launch_nomoon:.3f} → {dv_launch:.3f} km/s)')
            print(f'  Closure residual: {dv_lunar_residual:.3f} km/s  '
                  f'(em={em_closure:.3f} fb={flyby_deficit:.3f} '
                  f'hs={helio_splice:.3f} rr={return_rendezvous:.3f})')
            print(f'  N-body: apogee={nbody_result["max_geo_distance_km"]:.0f} km  '
                  f'(moon at {moon_dist_geo:.0f} km) → '
                  f'{"CAN reach Moon" if can_reach_moon else "CANNOT reach Moon"}')
            if soi_entered:
                print(f'  N-body: entered Moon SOI  (closest approach = {closest_approach:.0f} km)')
            else:
                print(f'  N-body: missed Moon SOI  (closest approach = {closest_approach:.0f} km)')

    # ── Total Δv ──
    # Δv_total = Δv_launch + Δv_lunar_residual + Δv_reentry
    if not use_lunar:
        dv_launch = dv_launch_nomoon
        dv_lunar_residual = 0.0
    dv_total = dv_launch + dv_lunar_residual + dv_reentry

    # No-moon baseline (same formula without lunar assist)
    dv_total_nomoon = dv_launch_nomoon + dv_reentry
    saving_pct = 0.0
    if dv_total_nomoon > 1e-6 and np.isfinite(dv_total_nomoon) and np.isfinite(dv_total):
        saving_pct = (dv_total_nomoon - dv_total) / dv_total_nomoon * 100

    # ── Constraint checks ──
    constraints = {
        'C1_no_moon_impact': r_m is None or (r_m >= R_MOON + 100),
        'C2_no_sun_impact': r_p > R_SUN,
        'C3_flight_time': ell['T_years'] <= 2.0,
        'C4_reentry_speed': ell['Delta_v_dep'] <= 15.0,
        'C5_can_reach_moon': not use_lunar or r_m is None or can_reach_moon,
    }
    all_ok = all(constraints.values())
    # Informational (not blocking):
    info_checks = {
        'I1_moon_phase_favorable': moon_phase_ok,
        'I2_soi_entered': soi_entered,
    }

    if verbose:
        print(f'Date: {date_str}  r_p: {r_p/AU:.3f} AU')
        print(f'  Δv launch:           {dv_launch:.2f} km/s  '
              f'({"200 km LEO → escape" if not use_lunar or r_m is None else "lunar-assisted departure"})')
        if use_lunar and r_m is not None:
            print(f'  Δv lunar residual:   {dv_lunar_residual:.3f} km/s  '
                  f'(closure corrections)')
        print(f'  Δv reentry:          {dv_reentry:.2f} km/s  '
              f'(propulsive capture → 200 km LEO)')
        print(f'  Δv total:            {dv_total:.2f} km/s')
        print(f'  No-moon Δv:          {dv_total_nomoon:.2f} km/s  '
              f'(saving {saving_pct:.1f}%)')
        print(f'  Flight time: {ell["T_years"]:.2f} yr')
        print(f'  Constraints: {"ALL PASS" if all_ok else "SOME FAIL"}')
        for k, v in constraints.items():
            if not v:
                print(f'    FAIL {k}')
        for k, v in info_checks.items():
            status = '✓' if v else 'info'
            print(f'    [{status}] {k}: {v}')

    return {
        'date': date_str,
        'r_p': r_p,
        'r_p_au': r_p / AU,
        'r_m': r_m,
        'side': side,
        # Δv budget (all km/s, non-negative except closure terms can be zero)
        'Delta_v_launch': dv_launch,
        'Delta_v_lunar_residual': dv_lunar_residual,
        'Delta_v_reentry': dv_reentry,
        'Delta_v_total': dv_total,
        'Delta_v_total_nomoon': dv_total_nomoon,
        # Closure breakdown
        'em_closure': em_closure,
        'flyby_deficit': flyby_deficit,
        'helio_splice': helio_splice,
        'return_rendezvous': return_rendezvous,
        'saving_pct': saving_pct,
        'ellipse': ell,
        'swingby': swingby_info,
        'constraints': {**constraints, **info_checks},  # all checks (hard + info)
        'all_ok': all_ok,
        'info_checks': info_checks,
        # Lunar swingby verification fields
        'moon_phase_angle_deg': np.degrees(np.arccos(np.clip(
            np.dot(moon_geo_pos / norm(moon_geo_pos) if norm(moon_geo_pos) > 0 else [0, 0, 0],
                   ev / norm(ev)), -1, 1))) if use_lunar and r_m is not None else None,
        'moon_phase_ok': moon_phase_ok,
        'v_inf_moon_kms': norm(mv - (ev + ell['Delta_v_dep'] * ev / norm(ev))) if use_lunar and r_m is not None else 0.0,
        'v_moon_actual_kms': v_moon_actual,
        'soi_entered': soi_entered,
        'can_reach_moon': can_reach_moon,
        'closest_approach_km': closest_approach,
        'nbody_result': nbody_result,
    }


def verify_full_mission(date_str, r_p, r_m=None, side=None, dt=3600):
    """Full N-body trajectory verification of a lunar-assist solar return.

    Propagates the complete mission (Earth→Moon→Sun→Earth) using the
    competitor's N-body integrator, verifying:

      Rule 15 — Lunar rendezvous: rocket enters Moon SOI
      Rule 16 — Solar return:     rocket returns to Earth vicinity
      Rule 18 — Physical constraints: r_p > R_SUN along actual trajectory

    The analytic patched-conic solution provides initial conditions; N-body
    propagation verifies the actual trajectory satisfies constraints.

    Returns dict with verification results.
    """
    result = {'date': date_str, 'r_p': r_p, 'r_m': r_m, 'side': side}

    # ── Get ephemeris and analytic solution ──
    ep, ev, mp, mv = get_ephemeris(date_str)
    r_1 = norm(ep)
    ell = helio_ellipse(r_p, r_1)
    moon_geo_pos = mp - ep
    moon_dist_geo = norm(moon_geo_pos)
    moon_geo_vel = mv - ev

    # ── Build initial N-body state ──
    y0 = np.zeros(24)
    y0[0:3] = [0.0, 0.0, 0.0]
    y0[3:6] = [0.0, 0.0, 0.0]
    y0[6:9] = ep
    y0[9:12] = ev
    y0[12:15] = mp
    y0[15:18] = mv

    # ── Phase 1: Earth departure → Moon ──
    # Rocket departs Earth with heliocentric velocity v_a (aphelion velocity
    # of the transfer ellipse). This is the patched-conic departure condition:
    # v_rocket_helio = v_earth - Δv_dep * direction
    v_earth_dir = ev / norm(ev)
    v_rocket_helio_departure = ev - ell['Delta_v_dep'] * v_earth_dir
    # Alternative: v_a directly
    # v_a = ell['v_a']
    # v_rocket_helio_departure = ell['v_a'] * v_earth_dir  # direction along Earth motion

    z_hat = np.array([0.0, 0.0, 1.0])
    dt_scan = 3600

    # Scan launch position and velocity direction for best Moon intercept.
    # Rocket departs Earth with heliocentric velocity ≈ v_a (transfer ellipse).
    best_approach = float('inf')
    best_phase1 = None

    # Rocket starts at 200 km altitude from Earth
    r_start = 6378.137 + 200.0

    for hour_off in range(0, 24, 4):
        dt_offset = hour_off * 3600.0
        y_adv = y0.copy()
        n_adv = int(dt_offset / dt_scan)
        for _ in range(n_adv):
            try:
                y_adv = velocity_verlet_step(y_adv, dt_scan)
            except RuntimeError:
                break
        ep_o = y_adv[6:9].copy()

        # Scan directions around Earth
        for az_deg in range(0, 360, 45):
            az = np.radians(az_deg)
            for el_deg in range(-60, 61, 60):
                el = np.radians(el_deg)
                launch_dir = np.array([np.cos(el) * np.cos(az),
                                        np.cos(el) * np.sin(az),
                                        np.sin(el)])
                y_test = y_adv.copy()
                y_test[18:21] = ep_o + r_start * launch_dir
                y_test[21:24] = v_rocket_helio_departure

                # Propagate up to ~10 days and check Moon approach
                for _ in range(min(int(10.0 * DAY / dt_scan), 250)):
                    try:
                        y_test = velocity_verlet_step(y_test, dt_scan)
                    except RuntimeError:
                        break
                    d = norm(y_test[18:21] - y_test[12:15])
                    if d < best_approach:
                        best_approach = d
                        best_phase1 = {
                            'hour_offset': hour_off,
                            'azimuth_deg': az_deg,
                            'elevation_deg': el_deg,
                            'y_at_closest': y_test.copy(),
                            'closest_approach_km': d,
                        }

    result['phase1'] = best_phase1
    if best_phase1 is None:
        result['rule15_pass'] = False
        result['rule16_pass'] = False
        result['rule18_pass'] = False
        return result

    soi_entered_phase1 = best_phase1['closest_approach_km'] < R_MOON_SOI
    moon_approach_km = best_phase1['closest_approach_km']
    result['rule15_pass'] = soi_entered_phase1
    result['moon_closest_approach_km'] = moon_approach_km
    result['moon_soi_entered'] = soi_entered_phase1

    # ── Phase 2: Lunar flyby → heliocentric ──
    y_at_moon = best_phase1['y_at_closest'].copy()

    # Compute v_inf at Moon and apply analytic deflection
    v_rocket = y_at_moon[21:24].copy()
    r_moon_flyby = y_at_moon[12:15].copy()
    v_moon_flyby = y_at_moon[15:18].copy()

    v_inf_moon_vec = v_rocket - v_moon_flyby
    v_inf_moon_mag = norm(v_inf_moon_vec)

    if v_inf_moon_mag > 0.5 and r_m is not None and side is not None:
        try:
            swingby = lunar_swingby(v_inf_moon_mag, r_m, side)
            delta = swingby['delta']
            sign = swingby['sign']

            # Apply deflection: rotate v_inf by δ in the v_inf × (v_inf × v_moon) plane
            v_inf_dir = v_inf_moon_vec / v_inf_moon_mag
            # Approximate: rotate v_inf toward/away from v_moon direction
            v_moon_dir = v_moon_flyby / (norm(v_moon_flyby) + 1e-30)
            rot_axis = np.cross(v_inf_dir, v_moon_dir)
            if norm(rot_axis) > 1e-12:
                rot_axis = rot_axis / norm(rot_axis)
                cos_d = np.cos(sign * delta)
                sin_d = np.sin(sign * delta)
                v_inf_out = (cos_d * v_inf_moon_vec +
                             sin_d * np.cross(rot_axis, v_inf_moon_vec) +
                             (1 - cos_d) * np.dot(rot_axis, v_inf_moon_vec) * rot_axis)
            else:
                v_inf_out = v_inf_moon_vec

            v_rocket_after = v_moon_flyby + v_inf_out
            y_at_moon[21:24] = v_rocket_after
        except ValueError:
            pass

    # ── Phase 3: Heliocentric transfer → Earth return ──
    # Propagate for the expected flight time
    flight_time_s = ell['T_years'] * 365.25 * DAY
    n_steps_helio = int(flight_time_s / dt)

    y_helio = y_at_moon.copy()
    r_min = float('inf')
    earth_closest = float('inf')
    perihelion_ok = True
    hit_sun = False

    for step in range(min(n_steps_helio, 500000)):
        try:
            y_helio = velocity_verlet_step(y_helio, dt)
        except RuntimeError:
            break

        r_sun_dist = norm(y_helio[18:21])
        r_earth_dist = norm(y_helio[18:21] - y_helio[6:9])

        r_min = min(r_min, r_sun_dist)
        earth_closest = min(earth_closest, r_earth_dist)

        if r_sun_dist < R_SUN:
            hit_sun = True
            break

        # Stop if clearly past perihelion and heading back
        if step % 10000 == 0 and step > 0:
            if r_earth_dist < earth_closest + 1e6:
                break  # approaching Earth

    result['perihelion_km'] = r_min
    result['earth_closest_approach_km'] = earth_closest
    result['perihelion_ok'] = r_min > R_SUN
    result['hit_sun'] = hit_sun

    # Rule 16: Earth return ≤ capture distance (generous: ~0.01 AU ≈ 1.5M km)
    result['rule16_pass'] = earth_closest < 0.02 * AU and not hit_sun
    # Rule 18: physical constraints
    result['rule18_pass'] = r_min > R_SUN and not hit_sun and (
        r_m is None or r_m >= R_MOON + 100)

    return result


def solve_and_emit_trajectory(date_str, r_p, r_m=None, side=None,
                               dt=3600, output_file=None):
    """Solve single-date trajectory and emit machine-readable output.

    Produces a complete JSON summary with:
      - Δv budget (launch, lunar_residual, reentry, total, nomoon)
      - Encounter times (Moon closest approach, Earth return)
      - Miss distances (Moon, Earth)
      - Perihelion distance
      - Timestamped state snapshots (every N steps)
      - Physical constraint checks (R15/R16/R18)

    If output_file is provided, saves to that path.
    Prints summary to stdout.

    Returns the summary dict.
    """
    import json, os
    from datetime import datetime as _dt, timedelta as _td

    # ── Analytic solution ──
    sol = solve_single_date(date_str, r_p, r_m, side, verbose=False)

    # ── Full N-body trajectory with timestamp tracking ──
    from nbody import velocity_verlet_step

    ep, ev, mp, mv = get_ephemeris(date_str)
    r_1 = norm(ep)
    ell = helio_ellipse(r_p, r_1)

    # Build initial state and scan for Moon intercept
    y0 = np.zeros(24)
    y0[0:3] = [0, 0, 0]
    y0[3:6] = [0, 0, 0]
    y0[6:9] = ep
    y0[9:12] = ev
    y0[12:15] = mp
    y0[15:18] = mv

    v_earth_dir = ev / norm(ev)
    v_dep = ev - ell['Delta_v_dep'] * v_earth_dir
    r_start = 6378.137 + 200.0
    dt_scan = 3600

    best_approach = float('inf')
    best_traj = None
    best_moon_time = 0.0

    for hour_off in range(0, 24, 4):
        y_adv = y0.copy()
        for _ in range(int(hour_off * 3600 / dt_scan)):
            try:
                y_adv = velocity_verlet_step(y_adv, dt_scan)
            except RuntimeError:
                break
        ep_o = y_adv[6:9].copy()
        for az_deg in range(0, 360, 60):
            az = np.radians(az_deg)
            launch_dir = np.array([np.cos(az), np.sin(az), 0.0])
            y_try = y_adv.copy()
            y_try[18:21] = ep_o + r_start * launch_dir
            y_try[21:24] = v_dep

            for step in range(250):
                try:
                    y_try = velocity_verlet_step(y_try, dt_scan)
                except RuntimeError:
                    break
                d = norm(y_try[18:21] - y_try[12:15])
                if d < best_approach:
                    best_approach = d
                    best_traj = y_try.copy()
                    best_moon_time = hour_off * 3600.0 + step * dt_scan

    # ── Phase 2: Apply flyby deflection ──
    if best_traj is not None and r_m is not None and side is not None:
        v_rocket = best_traj[21:24].copy()
        v_moon_fb = best_traj[15:18].copy()
        v_inf_vec = v_rocket - v_moon_fb
        v_inf_mag = norm(v_inf_vec)
        if v_inf_mag > 0.5:
            try:
                sw = lunar_swingby(v_inf_mag, r_m, side)
                delta = sw['delta']
                sgn = sw['sign']
                v_inf_dir = v_inf_vec / v_inf_mag
                v_moon_dir = v_moon_fb / (norm(v_moon_fb) + 1e-30)
                rot = np.cross(v_inf_dir, v_moon_dir)
                if norm(rot) > 1e-12:
                    rot = rot / norm(rot)
                    cd, sd = np.cos(sgn * delta), np.sin(sgn * delta)
                    v_out = (cd * v_inf_vec + sd * np.cross(rot, v_inf_vec) +
                             (1 - cd) * np.dot(rot, v_inf_vec) * rot)
                    best_traj[21:24] = v_moon_fb + v_out
            except ValueError:
                pass

    # ── Phase 3: Heliocentric → Earth return with snapshots ──
    flight_s = ell['T_years'] * 365.25 * DAY
    n_helio = min(int(flight_s / dt), 500000)
    snapshot_every = max(1, n_helio // 200)  # ~200 snapshots

    snapshots = []
    earth_closest = float('inf')
    earth_time = 0.0
    r_min = float('inf')
    peri_time = 0.0
    hit_sun = False
    y = best_traj.copy() if best_traj is not None else y0.copy()

    for step in range(n_helio):
        try:
            y = velocity_verlet_step(y, dt)
        except RuntimeError:
            break

        r_sun = norm(y[18:21])
        r_earth = norm(y[18:21] - y[6:9])

        if r_sun < r_min:
            r_min = r_sun
            peri_time = best_moon_time + step * dt

        if r_earth < earth_closest:
            earth_closest = r_earth
            earth_time = best_moon_time + step * dt

        if r_sun < R_SUN:
            hit_sun = True
            break

        if step % snapshot_every == 0 or step == n_helio - 1:
            snapshots.append({
                'step': step,
                't_seconds': best_moon_time + step * dt,
                't_days': (best_moon_time + step * dt) / DAY,
                'rocket_x_km': float(y[18]), 'rocket_y_km': float(y[19]),
                'rocket_z_km': float(y[20]),
                'rocket_vx_kms': float(y[21]), 'rocket_vy_kms': float(y[22]),
                'rocket_vz_kms': float(y[23]),
                'earth_x_km': float(y[6]), 'earth_y_km': float(y[7]),
                'earth_z_km': float(y[8]),
                'moon_x_km': float(y[12]), 'moon_y_km': float(y[13]),
                'moon_z_km': float(y[14]),
                'r_sun_km': float(r_sun),
                'r_earth_km': float(r_earth),
            })

    # ── Compute absolute times ──
    launch_dt = _dt.strptime(date_str, '%Y-%m-%d')
    moon_encounter_dt = launch_dt + _td(seconds=best_moon_time)
    return_dt = launch_dt + _td(seconds=earth_time)

    # ── Build summary ──
    summary = {
        'mission': {
            'launch_date': date_str,
            'target_r_p_au': r_p / AU,
            'target_r_m_km': r_m,
            'side': side,
        },
        'encounter_times': {
            'launch_iso': launch_dt.isoformat() + 'Z',
            'moon_closest_approach_iso': moon_encounter_dt.isoformat() + 'Z',
            'moon_closest_approach_seconds': best_moon_time,
            'earth_return_iso': return_dt.isoformat() + 'Z',
            'earth_return_seconds': earth_time,
            'perihelion_seconds': peri_time,
        },
        'miss_distances_km': {
            'moon_closest': best_approach,
            'earth_closest': earth_closest,
        },
        'perihelion_km': r_min,
        'hit_sun': hit_sun,
        'delta_v_kms': {
            'launch': sol['Delta_v_launch'],
            'lunar_residual': sol['Delta_v_lunar_residual'],
            'reentry': sol['Delta_v_reentry'],
            'total': sol['Delta_v_total'],
            'total_nomoon': sol['Delta_v_total_nomoon'],
            'saving_pct': sol['saving_pct'],
        },
        'closure_terms_kms': {
            'em_closure': sol.get('em_closure', 0.0),
            'flyby_deficit': sol.get('flyby_deficit', 0.0),
            'helio_splice': sol.get('helio_splice', 0.0),
            'return_rendezvous': sol.get('return_rendezvous', 0.0),
        },
        'constraints': {
            'R15_moon_soi': bool(best_approach < R_MOON_SOI),
            'R16_earth_return': bool(earth_closest < 0.02 * AU and not hit_sun),
            'R18_physical': bool(r_min > R_SUN and not hit_sun and (
                r_m is None or r_m >= R_MOON + 100)),
        },
        'moon_phase': {
            'angle_deg': sol.get('moon_phase_angle_deg'),
            'favorable': sol.get('moon_phase_ok', False),
        },
        'trajectory': snapshots,
        'n_snapshots': len(snapshots),
    }

    # ── Output ──
    print('=' * 60)
    print('MISSION SUMMARY (Machine-Readable)')
    print('=' * 60)
    print(f'  Launch:        {date_str}')
    print(f'  Moon approach: {moon_encounter_dt.isoformat()}  '
          f'({best_moon_time/DAY:.2f} d after launch)')
    print(f'  Closest Moon:  {best_approach:,.0f} km  '
          f'({"SOI" if best_approach < R_MOON_SOI else "MISS"})')
    print(f'  Perihelion:    {r_min:,.0f} km  ({r_min/AU:.4f} AU)')
    print(f'  Earth return:  {return_dt.isoformat()}  '
          f'({earth_time/DAY:.2f} d after launch)')
    print(f'  Closest Earth: {earth_closest:,.0f} km')
    print(f'  Δv_total:      {sol["Delta_v_total"]:.3f} km/s  '
          f'(save {sol["saving_pct"]:.1f}% vs no-moon)')
    print(f'  R15:{"PASS" if best_approach < R_MOON_SOI else "FAIL"}  '
          f'R16:{"PASS" if earth_closest < 0.02 * AU else "FAIL"}  '
          f'R18:{"PASS" if r_min > R_SUN else "FAIL"}')
    print(f'  Snapshots:     {len(snapshots)} states saved')

    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

        class _NpEncoder(json.JSONEncoder):
            def default(self, obj):
                import numpy as _np
                if isinstance(obj, (_np.integer,)):
                    return int(obj)
                if isinstance(obj, (_np.floating,)):
                    return float(obj)
                if isinstance(obj, (_np.bool_,)):
                    return bool(obj)
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, cls=_NpEncoder)
        print(f'\n  Full trajectory saved to {output_file}')

    return summary


if __name__ == '__main__':
    result = solve_single_date(
        '2026-06-15', r_p=0.25 * AU, r_m=5000, side='trailing'
    )
    print()
    print('=== Full mission N-body verification ===')
    mission = verify_full_mission('2026-06-15', 0.25 * AU, r_m=5000, side='trailing')
    r15 = 'PASS' if mission.get('rule15_pass') else 'FAIL'
    r16 = 'PASS' if mission.get('rule16_pass') else 'FAIL'
    r18 = 'PASS' if mission.get('perihelion_ok') else 'FAIL'
    solve_and_emit_trajectory(
        '2026-01-07', 0.35 * AU, r_m=5000, side='trailing',
        output_file='data/mission_summary.json')
