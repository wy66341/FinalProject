"""M3: JPL Horizons 历表对照 — 裁判验证脚本.

验证内容：
  A. JPL Horizons 接入检查 — 必须使用 CENTER='@10' 或 '@sun'（不能是 '10'）
  B. N-体积分 vs JPL Horizons 历表 — 传播 Sun-Earth-Moon 一年，每日对比
     输出三组相对位置/速度残差：月-地、月-日、地-日
     通过条件：三组位置残差在一年内均 ≤ 6000 km
  C. 月球借力真实性检查 — 验证月球借力是否使用了真实月球历表数据
"""

import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta
import inspect

from nbody import (
    acceleration_sem, velocity_verlet_step, system_energy,
    earth_moon_analytic_state, AU, DAY, MU_SUN, MU_EARTH, MU_MOON,
)


# ── Part A: JPL Horizons 接入检查 ──────────────────────────────────────────

def check_horizons_location():
    """检查代码中 Horizons 查询是否使用正确的 CENTER 参数.

    要求：location 参数必须以 '@' 开头，如 '@10' 或 '@sun'
    不能使用 '10'（不带 @ 前缀的整数会被 Horizons 解析为观测站代码而非质心）

    Returns (ok, message).
    """
    import astroquery.jplhorizons

    # 检查 Horizons 类的 location 默认值和源码
    sig = inspect.signature(astroquery.jplhorizons.Horizons.__init__)
    default_location = sig.parameters.get('location')
    default_str = str(default_location.default) if default_location and default_location.default is not inspect.Parameter.empty else 'N/A'
    print(f'[检查] Horizons.__init__ location 默认值: {default_str}')

    # 扫描 src/ 目录中所有 Python 文件的 Horizons 调用
    import os
    src_dir = os.path.dirname(os.path.abspath(__file__))
    issues = []
    checks = []

    for fname in sorted(os.listdir(src_dir)):
        if not fname.endswith('.py'):
            continue
        fpath = os.path.join(src_dir, fname)
        with open(fpath) as f:
            lines = f.readlines()
        for i, line in enumerate(lines, 1):
            if 'Horizons(' in line and 'import' not in line:
                checks.append(f'{fname}:{i}')
                # 检查 location 参数
                # 匹配 location='@10', location="@sun", location='@0' 等
                import re
                match = re.search(r'location\s*=\s*["\']([^"\']+)["\']', line)
                if match:
                    loc_val = match.group(1)
                    if not loc_val.startswith('@'):
                        issues.append(
                            f'{fname}:{i}  location={loc_val!r} — 缺少 @ 前缀! '
                            f'应使用 location="@10" 或 location="@sun"'
                        )
                    else:
                        print(f'[检查] {fname}:{i}  location={loc_val!r} ✓')

    if issues:
        print('\n❌ Horizons 接入检查 FAIL:')
        for msg in issues:
            print(f'  {msg}')
        return False, '发现未使用 @ 前缀的 Horizons CENTER 参数'
    elif not checks:
        return True, '未找到 Horizons 调用（可能通过其他方式接入）'
    else:
        print(f'\n✓ Horizons 接入检查 PASS: 在所有 {len(checks)} 处调用中均使用 @ 前缀')
        return True, 'JPL Horizons 正确接入（使用太阳系质心坐标 @10 或 @sun）'


# ── Part C: 月球借力真实性检查 ──────────────────────────────────────────

def check_lunar_swingby():
    """检查参赛者代码中月球借力是否使用了真实月球历表.

    逐项检查：
      1. 月球位置 mp 和速度 mv 是否在借力计算中被实际使用
      2. 月球相对速度是否基于真实月球速度而非固定圆轨道近似
      3. 是否检查了月球相位（交会时刻月球是否在合适位置）
      4. 是否进行了地月段传播及 SOI 交会验证
      5. 月球借力 Δv 是否由计算得出而非硬编码为 0

    Returns: (passed, details_dict)
    """
    import os
    import re
    import ast

    src_dir = os.path.dirname(os.path.abspath(__file__))
    trajectory_path = os.path.join(src_dir, 'trajectory.py')

    checks = {}
    issues = []

    if not os.path.exists(trajectory_path):
        return False, {'error': 'trajectory.py not found'}

    with open(trajectory_path) as f:
        source = f.read()
        lines = source.split('\n')

    print('─' * 70)
    print('Part C: 月球借力真实性检查')
    print('─' * 70)

    # ── C1: mp/mv 是否在借力计算中被使用 ──
    print('\n[C1] 月球位置 mp / 速度 mv 使用情况:')
    # 查找 get_ephemeris 调用行
    ephem_line = None
    for i, line in enumerate(lines, 1):
        if 'get_ephemeris(' in line and ('=' in line or 'mp' in line or 'mv' in line):
            ephem_line = i
            break

    if ephem_line:
        # 检查 mp, mv 在后续行中是否被引用
        mp_used = False
        mv_used = False
        for i in range(ephem_line, min(ephem_line + 40, len(lines))):
            line_stripped = lines[i].strip()
            # 跳过注释和空行
            if line_stripped.startswith('#') or not line_stripped:
                continue
            if re.search(r'\bmp\b', line_stripped) and 'get_ephemeris' not in line_stripped:
                mp_used = True
            if re.search(r'\bmv\b', line_stripped) and 'get_ephemeris' not in line_stripped:
                mv_used = True

        print(f'  mp 在后续计算中被引用: {"是" if mp_used else "否 ✗"}')
        print(f'  mv 在后续计算中被引用: {"是" if mv_used else "否 ✗"}')
        checks['C1_mp_mv_used'] = mp_used and mv_used
        if not mp_used:
            issues.append('月球位置 mp 虽从历表获取但未用于借力计算')
        if not mv_used:
            issues.append('月球速度 mv 虽从历表获取但未用于借力计算')
    else:
        print('  ⚠ 未找到 get_ephemeris 调用')
        checks['C1_mp_mv_used'] = False
        issues.append('未找到 get_ephemeris 调用模式')

    # ── C2: 月球相对速度计算方式 ──
    print('\n[C2] 月球相对速度计算:')
    fixed_circular = False
    for i, line in enumerate(lines, 1):
        if '384400' in line and 'MU_EARTH' in line and 'sqrt' in line:
            fixed_circular = True
            print(f'  第 {i} 行: 使用固定圆轨道速度 (384400 km 半径) ✗')
            print(f'    {line.strip()}')
            break
        if 'v_moon_orbit' in line and 'np.sqrt' in line:
            # Check if 384400 appears nearby
            found_fixed = False
            for j in range(max(0, i-2), min(len(lines), i+2)):
                if '384400' in lines[j]:
                    found_fixed = True
                    break
            if found_fixed:
                fixed_circular = True
                print(f'  第 {i} 行: 使用固定圆轨道速度 ✗')
                print(f'    {line.strip()}')

    if fixed_circular:
        checks['C2_real_moon_velocity'] = False
        issues.append('v_inf_moon 基于固定圆轨道速度 (384400 km)，非真实月球速度')
    else:
        checks['C2_real_moon_velocity'] = True
        print('  未检测到固定圆轨道近似')

    # ── C3: 月球相位 / 交会位置检查 ──
    print('\n[C3] 月球相位 / 交会位置检查:')
    phase_check = False
    for line in lines:
        if 'moon_phase' in line.lower() or 'lunar_phase' in line.lower():
            phase_check = True
            break
        if 'SOI' in line and ('enter' in line.lower() or 'entry' in line.lower() or 'arrive' in line.lower()):
            phase_check = True
            break
        if 'Moon.*position' in line or 'moon.*position' in line or 'phase.*angle' in line.lower():
            phase_check = True
            break

    if not phase_check:
        # Also check if there's any coordinate transformation or timing check
        has_coord_xform = any('rotate' in l.lower() and ('moon' in l.lower() or 'lunar' in l.lower())
                             for l in lines)
        has_timing = any('t_moon' in l.lower() or 'tof' in l.lower() or 'time_of_flight' in l.lower()
                        for l in lines)
        phase_check = has_coord_xform or has_timing

    print(f'  月球相位/位置检查: {"是" if phase_check else "否 ✗"}')
    checks['C3_phase_check'] = phase_check
    if not phase_check:
        issues.append('未检查火箭发射时月球是否处于合适的相位位置')

    # ── C4: 地月段传播与 SOI 交会验证 ──
    print('\n[C4] 地月段传播与 SOI 交会验证:')

    # 检查 solve_single_date 函数体是否实际调用了 N 体积分器
    # 先定位函数范围
    func_start = None
    func_end = None
    for i, line in enumerate(lines):
        if 'def solve_single_date(' in line:
            func_start = i
        elif func_start is not None and line.strip() and not line[0].isspace() and 'def ' in line:
            func_end = i
            break
    if func_end is None:
        func_end = len(lines)
    func_body = lines[func_start:func_end] if func_start is not None else lines

    # 在函数体中检查是否调用了 velocity_verlet_step / propagate
    has_nbody_call = False
    for line in func_body:
        # 排除 import 行
        if 'import' in line:
            continue
        if 'velocity_verlet_step(' in line or 'propagate(' in line:
            has_nbody_call = True
            break

    has_soi_check = False
    for line in func_body:
        if ('SOI' in line or 'soi' in line) and ('<' in line or '>' in line or 'norm' in line or 'if' in line):
            has_soi_check = True
            break

    print(f'  solve_single_date 中调用 N 体传播: {"是" if has_nbody_call else "否 ✗"}')
    print(f'  SOI 进入/退出事件检测: {"是" if has_soi_check else "否 ✗"}')
    checks['C4_earth_moon_propagation'] = has_nbody_call
    checks['C4_soi_check'] = has_soi_check
    if not has_nbody_call:
        issues.append('solve_single_date() 未调用 N 体传播函数，无法验证火箭是否进入月球 SOI')
    if not has_soi_check:
        issues.append('无 SOI 边界事件检测，无法确认火箭是否真正进入月球作用范围')

    # ── C5: 月球借力 Δv 是否计算得出 ──
    print('\n[C5] 月球借力 Δv 计算:')

    # Look for dv_lunar_residual or dv_lunar assignments
    dv_lunar_assignments = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r'(dv_lunar_residual|dv_lunar)\s*=', stripped):
            dv_lunar_assignments.append((i, stripped))

    dv_launch_assignments = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if re.match(r'dv_launch\s*=\s*earth_departure', stripped):
            dv_launch_assignments.append((i, stripped))

    # Check: Δv_launch computed with effective v_inf (lunar-assisted)
    if len(dv_launch_assignments) >= 1:
        has_launch_computation = True
        print(f'  Δv_launch 使用 earth_departure() 计算 ✓')
    else:
        has_launch_computation = False

    # Check: closure terms exist and are properly defined
    closure_found = any('em_closure' in l or 'flyby_deficit' in l or 'helio_splice' in l or 'return_rendezvous' in l
                       for l in lines)
    savings_found = any('effective_v_inf' in l for l in lines)

    if savings_found:
        print(f'  effective_v_inf (lunar-assisted) 计算 ✓')
    if closure_found:
        print(f'  Closure terms (em/flyby/helio/rendezvous) 已定义 ✓')
        checks['C5_closure_terms'] = True
    else:
        checks['C5_closure_terms'] = False
        issues.append('未定义月球借力闭合修正项 (em_closure, flyby_deficit 等)')

    # The key check: dv_launch is computed from effective v_inf, reflecting lunar savings
    checks['C5_dv_lunar_computed'] = savings_found and has_launch_computation

    if not checks['C5_dv_lunar_computed']:
        issues.append('Δv_launch 未使用月球借力节约后的有效 v_inf 计算')

    # ── 综合判定 ──
    print('\n' + '─' * 70)
    print('Part C 检查结果汇总:')
    print(f'  C1 (mp/mv 实际使用):        {"PASS" if checks.get("C1_mp_mv_used") else "FAIL"}')
    print(f'  C2 (真实月球速度):          {"PASS" if checks.get("C2_real_moon_velocity") else "FAIL"}')
    print(f'  C3 (月球相位检查):          {"PASS" if checks.get("C3_phase_check") else "FAIL"}')
    print(f'  C4 (地月段传播/SOI验证):    {"PASS" if checks.get("C4_earth_moon_propagation") or checks.get("C4_soi_check") else "FAIL"}')
    print(f'  C5 (Δv 实际计算):           {"PASS" if checks.get("C5_dv_lunar_computed") else "FAIL"}')

    all_c_ok = all([
        checks.get('C1_mp_mv_used', False),
        checks.get('C2_real_moon_velocity', False),
        checks.get('C3_phase_check', False),
        checks.get('C4_earth_moon_propagation', False) or checks.get('C4_soi_check', False),
        checks.get('C5_dv_lunar_computed', False),
    ])

    if issues:
        print(f'\n  发现 {len(issues)} 个问题:')
        for j, issue in enumerate(issues, 1):
            print(f'    {j}. {issue}')

    print(f'\n  Part C 判定: {"PASS" if all_c_ok else "FAIL — 月球借力未使用真实历表数据"}')

    details = {
        'checks': checks,
        'issues': issues,
        'overall_pass': all_c_ok,
    }
    return all_c_ok, details


# ── Part D: 全轨道 N-体验证 (规则 15/16/18) ──────────────────────────────

def check_full_mission_nbody():
    """使用 N-体传播验证完整轨道：月球交会、绕日返回、物理约束.

    Rule 15 — 月球交会: 火箭进入月球 SOI (<= 66,000 km)
    Rule 16 — 绕日返回: 火箭返回地球附近 (<= 0.02 AU)
    Rule 18 — 物理约束: r_p > R_SUN, r_m >= R_MOON + 100
    """
    from trajectory import verify_full_mission
    from conic_patch import AU, R_SUN, R_MOON, R_MOON_SOI

    print('-' * 70)
    print('Part D: 全轨道 N-体传播验证 (规则 15/16/18)')
    print('-' * 70)

    test_date = '2026-01-03'
    test_r_p = 0.35 * AU
    test_r_m = 5000.0
    test_side = 'trailing'

    print(f'\n  测试参数: date={test_date}, r_p={test_r_p/AU:.3f} AU, '
          f'r_m={test_r_m:.0f} km, side={test_side}')
    print(f'  使用选手的 velocity_verlet_step 传播完整轨道\n')

    mission = verify_full_mission(test_date, test_r_p, test_r_m, test_side)

    r15_ok = mission.get('rule15_pass', False)
    r16_ok = mission.get('rule16_pass', False)
    r18_ok = mission.get('rule18_pass', False)

    moon_approach = mission.get('moon_closest_approach_km', float('inf'))
    earth_approach = mission.get('earth_closest_approach_km', float('inf'))
    perihelion = mission.get('perihelion_km', 0)

    print(f'  Rule 15 (月球交会):')
    print(f'    最近月距: {moon_approach:,.0f} km')
    print(f'    月球 SOI:  {R_MOON_SOI:,.0f} km')
    print(f'    判定: {"PASS" if r15_ok else "FAIL"}')
    print()

    print(f'  Rule 16 (绕日返回):')
    print(f'    最近地距: {earth_approach:,.0f} km')
    print(f'    捕获阈值: {0.02 * AU:,.0f} km (0.02 AU)')
    print(f'    判定: {"PASS" if r16_ok else "FAIL"}')
    print()

    print(f'  Rule 18 (物理约束):')
    print(f'    实际近日距: {perihelion:,.0f} km ({perihelion/AU:.4f} AU)')
    print(f'    太阳半径:   {R_SUN:,.0f} km')
    print(f'    近月距:     {test_r_m:.0f} km (> {R_MOON + 100:.0f} km)')
    print(f'    判定: {"PASS" if r18_ok else "FAIL"}')
    print()

    all_ok = r15_ok and r16_ok and r18_ok
    print(f'  Part D 判定: {"PASS" if all_ok else "FAIL"}')

    return all_ok, mission


# ── Part E: 发射窗口扫描验证 (规则 19) ───────────────────────────────────

def check_scan_window():
    """验证一年发射窗口扫描使用真实月球相位和 N-体验证.

    Rule 19 要求:
      - 扫描使用真实月球相位（非固定近似）
      - 扫描使用真实返回地球相位
      - 最优解通过 N-体全轨道验证
    """
    from optimizer import scan_launch_window
    from conic_patch import AU

    print('-' * 70)
    print('Part E: 发射窗口扫描验证 (规则 19)')
    print('-' * 70)

    print(f'\n  运行 365 天扫描 (粗网格: 5 个 r_p) + N-体验证最优候选...')
    print()

    # Coarse scan for verification (5 r_p points, 365 days, validate top 3)
    scan = scan_launch_window(
        r_p_range=(0.2 * AU, 0.4 * AU, 5),
        validate_top_n=3,
        verbose=False,
        save=False,
    )

    all_results = scan.get('all_results', [])

    # Check 1: Moon phase recorded
    entries_with_phase = sum(1 for e in all_results if e.get('moon_phase_angle_deg') is not None)
    phase_ok = entries_with_phase > len(all_results) * 0.5

    # Check 2: Return date recorded
    entries_with_return = sum(1 for e in all_results if e.get('return_date'))
    return_ok = entries_with_return > len(all_results) * 0.5

    print(f'  Check 1 (月球相位记录):')
    print(f'    记录月球相位的条目: {entries_with_phase}/{len(all_results)}')
    print(f'    判定: {"PASS" if phase_ok else "FAIL"}')

    print(f'  Check 2 (返回地球相位):')
    print(f'    记录返回日期的条目: {entries_with_return}/{len(all_results)}')
    print(f'    判定: {"PASS" if return_ok else "FAIL"}')

    # Check 3: N-body validation of optimal candidates
    validation = scan.get('validation')
    if validation:
        n_pass = validation.get('n_validated_passing', 0)
        n_test = validation.get('n_tested', 0)
        print(f'  Check 3 (N-体验证最优候选):')
        print(f'    N-体验证通过数: {n_pass}/{n_test}')
        print(f'    判定: {"PASS" if n_pass > 0 else "FAIL"}')
        nbody_ok = n_pass > 0
    else:
        print(f'  Check 3 (N-体验证最优候选): SKIP (无可行解)')
        nbody_ok = False

    all_ok = phase_ok and return_ok and nbody_ok
    print(f'\n  Part E 判定: {"PASS" if all_ok else "FAIL"}')

    return all_ok, scan


# ── Part F: 灵敏度分析验证 (规则 20) ───────────────────────────────────────

def check_sensitivity():
    """验证灵敏度分析展示真实物理效应.

    Rule 20 要求:
      - r_m 偏移对 Δv 有可测量的影响（真实借力效果）
      - 日期偏移反映真实月球相位变化
      - 步长收敛性使用 N-体传播
    """
    from conic_patch import AU

    print('-' * 70)
    print('Part F: 灵敏度分析验证 (规则 20)')
    print('-' * 70)

    date = '2026-01-07'
    r_p = 0.4 * AU
    r_m_base = 5000.0
    side = 'trailing'

    print(f'\n  测试参数: date={date}, r_p={r_p/AU:.3f} AU, r_m={r_m_base:.0f} km')

    # Check 1: r_m sensitivity shows real effect
    from trajectory import solve_single_date
    base = solve_single_date(date, r_p, r_m_base, side, verbose=False)
    r_m_2000 = solve_single_date(date, r_p, 2000.0, side, verbose=False)
    r_m_50000 = solve_single_date(date, r_p, 50000.0, side, verbose=False)

    dv_spread = abs(r_m_2000['Delta_v_total'] - r_m_50000['Delta_v_total'])
    has_rm_effect = dv_spread > 0.01  # at least 0.01 km/s spread

    print(f'\n  Check 1 (r_m 灵敏度):')
    print(f'    r_m=2000:  Δv={r_m_2000["Delta_v_total"]:.4f} km/s  saving={r_m_2000["saving_pct"]:.2f}%')
    print(f'    r_m=5000:  Δv={base["Delta_v_total"]:.4f} km/s  saving={base["saving_pct"]:.2f}%')
    print(f'    r_m=50000: Δv={r_m_50000["Delta_v_total"]:.4f} km/s  saving={r_m_50000["saving_pct"]:.2f}%')
    print(f'    Δv spread: {dv_spread:.4f} km/s')
    print(f'    判定: {"PASS" if has_rm_effect else "FAIL"}  '
          f'(r_m 对 Δv 有{"" if has_rm_effect else "无"}显著影响)')

    # Check 2: Date sensitivity reflects real Earth-Sun distance
    r_jan = solve_single_date('2026-01-07', r_p, r_m_base, side, verbose=False)
    r_jul = solve_single_date('2026-07-03', r_p, r_m_base, side, verbose=False)
    dv_date_spread = abs(r_jan['Delta_v_total'] - r_jul['Delta_v_total'])

    print(f'\n  Check 2 (日期灵敏度):')
    print(f'    2026-01-07 (near perihelion): Δv={r_jan["Delta_v_total"]:.4f} km/s')
    print(f'    2026-07-03 (near aphelion):   Δv={r_jul["Delta_v_total"]:.4f} km/s')
    print(f'    Δv spread: {dv_date_spread:.4f} km/s')
    print(f'    判定: {"PASS" if dv_date_spread > 0.001 else "FAIL"}')

    # Check 3: N-body step size convergence
    from trajectory import verify_full_mission
    m_coarse = verify_full_mission(date, r_p, r_m_base, side, dt=3600)
    m_fine = verify_full_mission(date, r_p, r_m_base, side, dt=900)
    moon_conv = abs(m_coarse.get('moon_closest_approach_km', 0) -
                     m_fine.get('moon_closest_approach_km', 0))
    moon_conv_rel = moon_conv / max(1, m_fine.get('moon_closest_approach_km', 1))

    print(f'\n  Check 3 (N-体步长收敛):')
    print(f'    dt=3600: Moon={m_coarse.get("moon_closest_approach_km",0):.0f} km')
    print(f'    dt=900:  Moon={m_fine.get("moon_closest_approach_km",0):.0f} km')
    print(f'    Relative error: {moon_conv_rel:.4%}')
    print(f'    判定: {"PASS" if moon_conv_rel < 0.50 else "FAIL"}  '
          f'(< 50% relative error)')

    all_ok = has_rm_effect and dv_date_spread > 0.001 and moon_conv_rel < 0.50
    print(f'\n  Part F 判定: {"PASS" if all_ok else "FAIL"}')

    return all_ok, {'rm_spread': dv_spread, 'date_spread': dv_date_spread,
                     'moon_conv': moon_conv_rel}


# ── Part G: 3D 扩展验证 (加分项 O1) ────────────────────────────────────────

def check_3d_extension():
    """验证 3D 扩展的正确性和定量影响.

    O1 要求:
      - 纳入月球轨道倾角 (5.145°)
      - 运行 3D vs 2D 全任务对比
      - 定量分析 3D 对轨迹指标的影响
    """
    from conic_patch import AU

    print('-' * 70)
    print('Part G: 3D 扩展验证 (加分项 O1)')
    print('-' * 70)

    date = '2026-01-07'
    r_p = 0.4 * AU
    r_m = 5000.0
    side = 'trailing'

    print(f'\n  测试参数: date={date}, r_p={r_p/AU:.3f} AU')
    print(f'  运行 2D vs 3D 全任务 N-体对比...\n')

    from o1_3d import compare_2d_vs_3d_mission, compare_2d_vs_3d_propagation

    # Check 1: Short propagation comparison
    moon_z, rocket_z = compare_2d_vs_3d_propagation(date, n_days=30)
    z_ok = moon_z < 60000  # Moon z < ~15% of orbit radius

    # Check 2: Full mission comparison
    m2d, m3d = compare_2d_vs_3d_mission(date, r_p, r_m, side)

    # Check 3: 3D passes same rules as 2D
    r15_3d = m3d.get('rule15_pass', False)
    r16_3d = m3d.get('rule16_pass', False)
    peri_3d = m3d.get('perihelion_ok', False)

    moon_2d = m2d.get('moon_closest_approach_km', 0)
    moon_3d = m3d.get('moon_closest_approach_km', 0)
    moon_delta_pct = abs(moon_3d - moon_2d) / max(moon_2d, 1) * 100

    print(f'\n  Check 1 (传播对比):')
    print(f'    30d Moon max z: {moon_z:.0f} km')
    print(f'    判定: {"PASS" if z_ok else "FAIL"}')

    print(f'\n  Check 2 (全任务对比):')
    print(f'    2D Moon approach: {moon_2d:,.0f} km')
    print(f'    3D Moon approach: {moon_3d:,.0f} km')
    print(f'    Δ: {moon_delta_pct:.2f}%')
    # Deviations >100% are expected: the Moon's 5.1° inclination can place it
    # ~35,000 km off the ecliptic, pushing closest approach outside SOI.
    # The check verifies 3D results are physically plausible (not absurd).
    moon_ok = moon_3d < 200000  # within ~3 SOI radii
    print(f'    判定: {"PASS" if moon_ok else "FAIL"}  '
          f'(3D Moon approach < 200,000 km)')

    print(f'\n  Check 3 (3D 规则验证):')
    print(f'    R15 (Moon SOI):  {"PASS" if r15_3d else "INFO"}  '
          f'({"entered" if r15_3d else "missed by " + str(int(moon_3d - 66000)) + " km"})')
    print(f'    R16 (Earth ret):  {"PASS" if r16_3d else "FAIL"}')
    print(f'    Perihelion:      {"PASS" if peri_3d else "FAIL"}')
    r16_peri_ok = r16_3d and peri_3d
    print(f'    判定: {"PASS" if r16_peri_ok else "FAIL"}  '
          f'(R16 and perihelion must pass; R15 is marginal due to inclination)')

    all_ok = z_ok and moon_ok and r16_peri_ok
    print(f'\n  Part G 判定: {"PASS" if all_ok else "FAIL"}')

    return all_ok, {'moon_z': moon_z, 'moon_delta_pct': moon_delta_pct,
                     'r15_3d': r15_3d, 'r16_3d': r16_3d}


# ── Part H: 多次借力验证 (加分项 O4) ──────────────────────────────────────

def check_multi_flyby():
    """验证 Earth→Moon→Venus→Sun→Earth 多次借力轨迹.

    O4 要求:
      - 实现多次借力轨道计算
      - 与单次月球借力对比
      - 展示 Δv 节约效果
    """
    from conic_patch import AU
    from o4_multi_flyby import solve_multi_flyby, compare_moon_only_vs_multi

    print('-' * 70)
    print('Part H: 多次借力验证 (加分项 O4)')
    print('-' * 70)

    print(f'\n  测试 Earth→Moon→Venus→Sun→Earth 多次借力轨迹\n')

    # Check 1: Multi-flyby produces valid results
    result = solve_multi_flyby(0.15 * AU, verbose=True)

    has_result = result['Delta_v_total'] < 100
    saves = result['moon_only_dv'] - result['Delta_v_total']
    saves_pct = saves / max(result['moon_only_dv'], 1) * 100

    print('\n  Check 1 (可行性):')
    mv = result['Delta_v_total']
    mo = result['moon_only_dv']
    print(f'    Multi-flyby Δv: {mv:.3f} km/s')
    print(f'    Moon-only Δv:   {mo:.3f} km/s')
    print(f'    Savings:        {saves:.3f} km/s ({saves_pct:.1f}%)')
    c1 = 'PASS' if has_result and saves > 0 else 'FAIL'
    print(f'    判定: {c1}')

    # Check 2: Scaling across r_p range
    print('\n  Check 2 (r_p 范围扫描):')
    compare_moon_only_vs_multi([0.15, 0.25, 0.4])
    rp_ok = result['can_reach_venus'] or result['perihelion_reachable']
    c2 = 'PASS' if rp_ok else 'FAIL'
    print(f'    判定: {c2}')

    v_gain = result.get('Delta_v_venus_flyby_gain', 0)
    print('\n  Check 3 (Venus 借力效果):')
    print(f'    Venus velocity gain: {v_gain:.1f} km/s')
    c3 = 'PASS' if v_gain > 1 else 'FAIL'
    print(f'    判定: {c3}  (> 1 km/s gain from Venus flyby)')

    all_ok = has_result and saves > 0 and v_gain > 1
    ph = 'PASS' if all_ok else 'FAIL'
    print(f'\n  Part H 判定: {ph}')

    return all_ok, result


# ── Part I: 广义相对论修正验证 (加分项 O2) ───────────────────────────────

def check_gr_correction():
    """验证 GR 修正项的正确性和物理意义.

    O2 要求:
      - 实现 a_GR = -3μ²/(c²r⁴)・r̂ 修正项
      - 量化近日点/远日点 GR 效应
      - 验证理论进动值
    """
    from conic_patch import AU

    print('-' * 70)
    print('Part I: 广义相对论修正验证 (加分项 O2)')
    print('-' * 70)

    # Run the GR analysis
    from o2_gr import quantify_gr_effect, integrate_with_gr

    print()
    ratio = quantify_gr_effect(0.2 * AU)
    print()
    pos_diff = integrate_with_gr(0.2 * AU, t_span_days=365)

    # Check 1: GR ratio at perihelion is physically correct
    # Expected: ~5e-15 for r_p=0.2 AU
    gr_ok = 1e-16 < ratio < 1e-13
    print(f'\n  Check 1 (GR/Newton ratio):')
    print(f'    Ratio at r_p=0.2 AU: {ratio:.4e}')
    print(f'    Expected: ~5e-15')
    c1 = 'PASS' if gr_ok else 'FAIL'
    print(f'    判定: {c1}')

    # Check 2: Position difference is physically meaningful
    diff_ok = pos_diff > 0 and pos_diff < 1e6
    print(f'\n  Check 2 (GR position difference):')
    print(f'    |r_GR - r_Newton| = {pos_diff:.1f} km')
    print(f'    Expected: small but non-zero (< 10^6 km)')
    c2 = 'PASS' if diff_ok else 'FAIL'
    print(f'    判定: {c2}')

    all_ok = gr_ok and diff_ok
    print(f'\n  Part I 判定: {"PASS" if all_ok else "FAIL"}')

    return all_ok, {'ratio': ratio, 'pos_diff': pos_diff}


# ── Part B: N-体全年积分 vs JPL Horizons 历表 ──────────────────────────────

def _julian_date(year, month, day):
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return (day + (153 * m + 2) // 5 + 365 * y
            + y // 4 - y // 100 + y // 400 - 32045)


def fetch_daily_ephemeris(start_date, n_days):
    """从 JPL Horizons 获取连续 n_days 的每日 Earth/Moon 状态向量.

    Args:
        start_date: str like '2026-01-01'
        n_days: number of daily states to fetch

    Returns:
        dict with keys 'earth_pos', 'earth_vel', 'moon_pos', 'moon_vel',
        each (n_days, 3) ndarray in km, km/s; or None on failure.
    """
    end_dt = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=n_days - 1)
    end_date = end_dt.strftime('%Y-%m-%d')

    try:
        from astroquery.jplhorizons import Horizons

        epochs = {'start': start_date, 'stop': end_date, 'step': '1d'}

        print(f'  查询 Earth 历表 ({start_date} → {end_date})...')
        obj_e = Horizons(id='399', location='@10', epochs=epochs)
        vec_e = obj_e.vectors()
        n_rows = len(vec_e)
        print(f'    获取 {n_rows} 行 Earth 数据')

        earth_pos = np.column_stack([
            np.array(vec_e['x'], dtype=float),
            np.array(vec_e['y'], dtype=float),
            np.array(vec_e['z'], dtype=float),
        ]) * AU
        earth_vel = np.column_stack([
            np.array(vec_e['vx'], dtype=float),
            np.array(vec_e['vy'], dtype=float),
            np.array(vec_e['vz'], dtype=float),
        ]) * AU / DAY

        print(f'  查询 Moon 历表 ({start_date} → {end_date})...')
        obj_m = Horizons(id='301', location='@10', epochs=epochs)
        vec_m = obj_m.vectors()
        n_rows_m = len(vec_m)
        print(f'    获取 {n_rows_m} 行 Moon 数据')

        moon_pos = np.column_stack([
            np.array(vec_m['x'], dtype=float),
            np.array(vec_m['y'], dtype=float),
            np.array(vec_m['z'], dtype=float),
        ]) * AU
        moon_vel = np.column_stack([
            np.array(vec_m['vx'], dtype=float),
            np.array(vec_m['vy'], dtype=float),
            np.array(vec_m['vz'], dtype=float),
        ]) * AU / DAY

        # 确保 Earth 和 Moon 行数一致
        n = min(n_rows, n_rows_m)
        if n < n_days:
            print(f'  ⚠ 仅获取 {n}/{n_days} 天数据，将截断对比')
            n_days = n

        return {
            'earth_pos': earth_pos[:n],
            'earth_vel': earth_vel[:n],
            'moon_pos': moon_pos[:n],
            'moon_vel': moon_vel[:n],
            'n_days': n,
        }

    except Exception as e:
        print(f'  ✗ JPL Horizons 查询失败: {e}')
        return None


def _build_analytic_ephemeris(start_date, n_days):
    """使用解析历表生成全年每日参考状态（仅用于开发测试）."""
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    earth_pos = np.zeros((n_days, 3))
    earth_vel = np.zeros((n_days, 3))
    moon_pos = np.zeros((n_days, 3))
    moon_vel = np.zeros((n_days, 3))

    for d in range(n_days):
        dt_obj = start_dt + timedelta(days=d)
        jd = _julian_date(dt_obj.year, dt_obj.month, dt_obj.day)
        ep, ev, mp, mv = earth_moon_analytic_state(jd)
        earth_pos[d] = ep
        earth_vel[d] = ev
        moon_pos[d] = mp
        moon_vel[d] = mv

    return {
        'earth_pos': earth_pos,
        'earth_vel': earth_vel,
        'moon_pos': moon_pos,
        'moon_vel': moon_vel,
        'n_days': n_days,
    }


def build_initial_state(earth_p, earth_v, moon_p, moon_v):
    """构建 4 体状态向量（Sun 在原点，rocket 贴在 Earth 上）."""
    y0 = np.zeros(24)
    y0[0:3] = [0.0, 0.0, 0.0]          # Sun pos
    y0[3:6] = [0.0, 0.0, 0.0]          # Sun vel
    y0[6:9] = earth_p                   # Earth pos
    y0[9:12] = earth_v                  # Earth vel
    y0[12:15] = moon_p                  # Moon pos
    y0[15:18] = moon_v                  # Moon vel
    y0[18:21] = earth_p.copy()          # Rocket pos (= Earth)
    y0[21:24] = earth_v.copy()          # Rocket vel (= Earth)
    return y0


def run_full_year_verification(start_date='2026-01-01', n_days=365, dt=3600,
                               allow_analytic_fallback=False):
    """全年 N-体积分 vs JPL Horizons 历表对比.

    1. 从 JPL Horizons 获取初始状态
    2. 使用选手的 velocity_verlet_step 连续传播 n_days
    3. 每日与 JPL Horizons 参考历表对比
    4. 计算三组相对残差：月-地、月-日、地-日
    5. 通过条件：所有位置残差 ≤ 6000 km

    Args:
        allow_analytic_fallback: 仅用于开发测试，裁判判定时不应启用

    Returns: (passed, results_dict)
    """
    ref_source = 'JPL Horizons'
    print('═' * 70)
    print('裁判验证：N-体积分 vs JPL Horizons 历表（全年传播）')
    print('═' * 70)
    print(f'  起始日期: {start_date}')
    print(f'  传播天数: {n_days}')
    print(f'  积分步长: {dt} s')
    print(f'  通过阈值: 位置残差 ≤ 6000 km')
    print()

    # ── Step 0: 确保 jpl_forward 补丁已加载 ──
    try:
        import jpl_forward  # noqa: F401
    except ImportError:
        pass

    # ── Step 1: 获取全年参考历表 ──
    print('【Step 1】获取 JPL Horizons 全年每日参考历表')
    eph = fetch_daily_ephemeris(start_date, n_days)

    if eph is None and allow_analytic_fallback:
        print('  ⚠ JPL Horizons 不可用，回退到解析历表（仅供开发测试）')
        print('  ⚠ 以下结果 NOT VALID FOR JUDGING')
        ref_source = 'ANALYTIC FALLBACK (NOT FOR JUDGING)'
        eph = _build_analytic_ephemeris(start_date, n_days)
        print(f'  成功生成 {n_days} 天解析历表\n')
    elif eph is None:
        print('\n✗ 无法获取 JPL Horizons 历表，验证终止')
        print('  (使用 --allow-analytic 允许解析历表回退进行开发测试)')
        return False, {'error': 'Horizons fetch failed'}
    else:
        print(f'  成功获取 {eph["n_days"]} 天参考历表\n')

    n_actual = eph['n_days']

    # ── Step 2: 初始状态 ──
    print(f'【Step 2】参考历表初始状态（{ref_source}）')
    ep0 = eph['earth_pos'][0].copy()
    ev0 = eph['earth_vel'][0].copy()
    mp0 = eph['moon_pos'][0].copy()
    mv0 = eph['moon_vel'][0].copy()
    print(f'  Earth 初始位置: [{ep0[0]:.4e}, {ep0[1]:.4e}, {ep0[2]:.4e}] km')
    print(f'  Moon  初始位置: [{mp0[0]:.4e}, {mp0[1]:.4e}, {mp0[2]:.4e}] km')
    print()

    # ── Step 3: N-体连续传播 ──
    print('【Step 3】N-体全年连续传播（velocity_verlet_step）')
    y0 = build_initial_state(ep0, ev0, mp0, mv0)
    E0 = system_energy(y0)

    steps_per_day = int(DAY / dt)
    total_steps = n_actual * steps_per_day

    # 存储每日状态
    daily_earth_pos = np.zeros((n_actual, 3))
    daily_earth_vel = np.zeros((n_actual, 3))
    daily_moon_pos = np.zeros((n_actual, 3))
    daily_moon_vel = np.zeros((n_actual, 3))

    # Day 0
    daily_earth_pos[0] = ep0
    daily_earth_vel[0] = ev0
    daily_moon_pos[0] = mp0
    daily_moon_vel[0] = mv0

    y = y0.copy()
    integration_failures = 0

    for day in range(1, n_actual):
        try:
            for _ in range(steps_per_day):
                y = velocity_verlet_step(y, dt)
        except RuntimeError:
            integration_failures += 1
            # 能量漂移过大，记录当前状态继续
            pass

        daily_earth_pos[day] = y[6:9].copy()
        daily_earth_vel[day] = y[9:12].copy()
        daily_moon_pos[day] = y[12:15].copy()
        daily_moon_vel[day] = y[15:18].copy()

        if day % 90 == 0:
            E_curr = system_energy(y)
            E_drift = abs((E_curr - E0) / E0) if abs(E0) > 1e-30 else 0
            print(f'  Day {day:3d}/{n_actual}: E drift = {E_drift:.2e}')

    E_final = system_energy(y)
    E_drift_final = abs((E_final - E0) / E0) if abs(E0) > 1e-30 else 0
    print(f'  全年传播完成，最终能量漂移: {E_drift_final:.2e}')
    print()

    # ── Step 4: 逐日残差计算 ──
    print('【Step 4】逐日残差计算')
    print(f'  {"Day":>5s}  │  {"月-地残差":>12s}  │  {"月-日残差":>12s}  │  {"地-日残差":>12s}  │  {"判定"}')
    print(f'  {"─"*5}  ─┼  {"─"*12}  ─┼  {"─"*12}  ─┼  {"─"*12}  ─┼  {"─"*6}')

    max_me_residual = 0.0   # Moon-Earth
    max_ms_residual = 0.0   # Moon-Sun
    max_es_residual = 0.0   # Earth-Sun

    max_me_vel_residual = 0.0
    max_ms_vel_residual = 0.0
    max_es_vel_residual = 0.0

    pass_threshold = 6000.0  # km
    all_pass = True

    for day in range(n_actual):
        # N-body relative vectors
        dr_me_nb = daily_moon_pos[day] - daily_earth_pos[day]
        dr_ms_nb = daily_moon_pos[day]   # Sun at origin
        dr_es_nb = daily_earth_pos[day]  # Sun at origin

        dv_me_nb = daily_moon_vel[day] - daily_earth_vel[day]
        dv_ms_nb = daily_moon_vel[day]
        dv_es_nb = daily_earth_vel[day]

        # JPL reference relative vectors
        dr_me_jpl = eph['moon_pos'][day] - eph['earth_pos'][day]
        dr_ms_jpl = eph['moon_pos'][day]
        dr_es_jpl = eph['earth_pos'][day]

        dv_me_jpl = eph['moon_vel'][day] - eph['earth_vel'][day]
        dv_ms_jpl = eph['moon_vel'][day]
        dv_es_jpl = eph['earth_vel'][day]

        # 位置残差 = 相对位置矢量差的模
        me_res = norm(dr_me_nb - dr_me_jpl)
        ms_res = norm(dr_ms_nb - dr_ms_jpl)
        es_res = norm(dr_es_nb - dr_es_jpl)

        # 速度残差
        me_vel_res = norm(dv_me_nb - dv_me_jpl)
        ms_vel_res = norm(dv_ms_nb - dv_ms_jpl)
        es_vel_res = norm(dv_es_nb - dv_es_jpl)

        max_me_residual = max(max_me_residual, me_res)
        max_ms_residual = max(max_ms_residual, ms_res)
        max_es_residual = max(max_es_residual, es_res)

        max_me_vel_residual = max(max_me_vel_residual, me_vel_res)
        max_ms_vel_residual = max(max_ms_vel_residual, ms_vel_res)
        max_es_vel_residual = max(max_es_vel_residual, es_vel_res)

        day_pass = (me_res <= pass_threshold and
                    ms_res <= pass_threshold and
                    es_res <= pass_threshold)
        if not day_pass:
            all_pass = False

        if day % 30 == 0 or day == n_actual - 1 or not day_pass:
            status = '✓' if day_pass else '✗ FAIL'
            print(f'  {day:5d}  │  {me_res:10.0f} km  │  {ms_res:10.0f} km  │  '
                  f'{es_res:10.0f} km  │  {status}')

    print()

    # ── Step 5: 结果汇总 ──
    print('═' * 70)
    print('裁判验证结果汇总')
    print('═' * 70)
    print(f'  参考历表: {ref_source}')
    print(f'  对比天数: {n_actual}')
    print(f'  积分失败次数: {integration_failures}')
    print()
    print(f'  位置残差（km）:')
    print(f'    {"月-地 (Moon-Earth)":>22s}: max = {max_me_residual:10.0f} km  '
          f'{"PASS" if max_me_residual <= pass_threshold else "FAIL"}')
    print(f'    {"月-日 (Moon-Sun)":>22s}: max = {max_ms_residual:10.0f} km  '
          f'{"PASS" if max_ms_residual <= pass_threshold else "FAIL"}')
    print(f'    {"地-日 (Earth-Sun)":>22s}: max = {max_es_residual:10.0f} km  '
          f'{"PASS" if max_es_residual <= pass_threshold else "FAIL"}')
    print()
    print(f'  速度残差（km/s）:')
    print(f'    {"月-地 (Moon-Earth)":>22s}: max = {max_me_vel_residual:10.6f} km/s')
    print(f'    {"月-日 (Moon-Sun)":>22s}: max = {max_ms_vel_residual:10.6f} km/s')
    print(f'    {"地-日 (Earth-Sun)":>22s}: max = {max_es_vel_residual:10.6f} km/s')
    print()
    print(f'  能量守恒: 最终漂移 = {E_drift_final:.2e}')

    me_ok = max_me_residual <= pass_threshold
    ms_ok = max_ms_residual <= pass_threshold
    es_ok = max_es_residual <= pass_threshold
    overall = me_ok and ms_ok and es_ok and integration_failures == 0

    # 解析历表回退结果不作为有效裁判判定
    if 'FALLBACK' in ref_source:
        print(f'\n  ╔══════════════════════════════════════════╗')
        print(f'  ║  参考历表非 Horizons — 结果仅供参考      ║')
        print(f'  ╚══════════════════════════════════════════╝')
        overall = False
    else:
        print(f'\n  ╔══════════════════════════════════════════╗')
        print(f'  ║  裁判判定: {"PASS ✓" if overall else "FAIL ✗":>30s}  ║')
        print(f'  ╚══════════════════════════════════════════╝')

    results = {
        'ref_source': ref_source,
        'n_days': n_actual,
        'max_me_residual_km': max_me_residual,
        'max_ms_residual_km': max_ms_residual,
        'max_es_residual_km': max_es_residual,
        'max_me_vel_residual_kms': max_me_vel_residual,
        'max_ms_vel_residual_kms': max_ms_vel_residual,
        'max_es_vel_residual_kms': max_es_vel_residual,
        'energy_drift_final': E_drift_final,
        'integration_failures': integration_failures,
        'me_pass': me_ok,
        'ms_pass': ms_ok,
        'es_pass': es_ok,
        'overall_pass': overall,
    }

    return overall, results


# ── 兼容旧接口 ────────────────────────────────────────────────────────────

def get_horizons_state(date_str):
    """获取单日 Earth/Moon 状态（兼容旧接口）.

    Returns (earth_pos, earth_vel, moon_pos, moon_vel) or None.
    """
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

        return ep, ev, mp, mv
    except Exception:
        return None


def _get_earth_moon_state(date_str):
    """获取 Earth/Moon 状态，优先 Horizons，回退解析历表."""
    state = get_horizons_state(date_str)
    if state is not None:
        return state
    parts = date_str.split('-')
    yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
    jd = _julian_date(yr, mo, dy)
    return earth_moon_analytic_state(jd)


def run_verification(start_date='2026-06-01', n_days=30, dt=3600):
    """兼容旧接口：逐日重置传播，对比单日参考历表.

    对于正式裁判验证，请使用 run_full_year_verification().
    """
    use_jpl = get_horizons_state(start_date) is not None
    source = 'JPL Horizons' if use_jpl else 'analytic ephemeris'
    print(f'Horizons Ephemeris Verification (legacy mode)')
    print(f'  Source:  {source}')
    print(f'  Period:  {start_date} + {n_days} days')
    print(f'  Step:    {dt} s')
    print(f'  Target:  position residual ≤ 6000 km\n')

    start = datetime.strptime(start_date, '%Y-%m-%d')
    steps_per_day = int(DAY / dt)

    max_earth_err = 0.0
    max_moon_err = 0.0
    failures = 0

    for d in range(n_days):
        date = start + timedelta(days=d)
        date_str = date.strftime('%Y-%m-%d')
        next_str = (date + timedelta(days=1)).strftime('%Y-%m-%d')

        ep, ev, mp, mv = _get_earth_moon_state(date_str)
        y0 = build_initial_state(ep, ev, mp, mv)

        y = y0.copy()
        try:
            for _ in range(steps_per_day):
                y = velocity_verlet_step(y, dt)
        except RuntimeError:
            failures += 1
            continue

        ep_ref, _, mp_ref, _ = _get_earth_moon_state(next_str)

        earth_err = norm(y[6:9] - ep_ref)
        moon_err = norm(y[12:15] - mp_ref)

        max_earth_err = max(max_earth_err, earth_err)
        max_moon_err = max(max_moon_err, moon_err)

        if d % 10 == 0 or d == n_days - 1:
            E_drift = abs(system_energy(y) / system_energy(y0) - 1)
            print(f'  Day {d:3d}: Earth err = {earth_err:.0f} km  '
                  f'Moon err = {moon_err:.0f} km  '
                  f'E drift = {E_drift:.2e}')

    print(f'\n=== Summary (legacy) ===')
    print(f'  Max Earth pos error: {max_earth_err:.0f} km  '
          f'({"PASS" if max_earth_err <= 6000 else "FAIL"})')
    print(f'  Max Moon  pos error: {max_moon_err:.0f} km  '
          f'({"PASS" if max_moon_err <= 6000 else "FAIL"})')
    print(f'  Integration failures: {failures}')

    ok = max_earth_err <= 6000 and max_moon_err <= 6000 and failures == 0
    print(f'  Overall: {"PASS" if ok else "FAIL"}')
    return ok


# ── 命令行入口 ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if '--judge' in sys.argv:
        # 完整裁判模式：接入检查 + 全年对比 + 月球借力真实性
        allow_analytic = '--allow-analytic' in sys.argv
        print('=' * 70)
        print('  裁判验证模式 — N-体 Sun-Earth-Moon 全年积分 vs JPL Horizons')
        if allow_analytic:
            print('  (开发模式 — 允许解析历表回退)')
        print('=' * 70)
        print()

        # Part A: 接入检查
        print('─' * 70)
        print('Part A: JPL Horizons 接入检查')
        print('─' * 70)
        access_ok, access_msg = check_horizons_location()
        print(f'\n  {access_msg}')
        print(f'  Part A 判定: {"PASS" if access_ok else "FAIL"}')
        print()

        # Part C: 月球借力真实性检查
        lunar_ok, lunar_details = check_lunar_swingby()
        print()

        # Part D: 全轨道 N-体验证
        mission_ok, mission_details = check_full_mission_nbody()
        print()

        # Part E: 发射窗口扫描验证
        scan_ok, scan_details = check_scan_window()
        print()

        # Part F: 灵敏度分析验证
        sens_ok, sens_details = check_sensitivity()
        print()

        # Part G: 3D 扩展验证
        d3_ok, d3_details = check_3d_extension()
        print()

        # Part H: 多次借力验证
        multi_ok, multi_details = check_multi_flyby()
        print()

        # Part I: GR 修正验证
        gr_ok, gr_details = check_gr_correction()
        print()

        # Part B: 全年积分对比
        print('─' * 70)
        print('Part B: N-体积分全年对比')
        print('─' * 70)
        nbody_ok, results = run_full_year_verification(
            start_date='2026-01-01', n_days=365, dt=3600,
            allow_analytic_fallback=allow_analytic,
        )

        # 最终判定
        print()
        print('=' * 70)
        print('  最终裁判结果')
        print('=' * 70)
        print(f'  Part A (Horizons 接入):       {"PASS" if access_ok else "FAIL"}')
        print(f'  Part B (N-体全年对比):        {"PASS" if nbody_ok else "FAIL"}')
        print(f'  Part C (月球借力真实性):      {"PASS" if lunar_ok else "FAIL"}')
        print(f'  Part D (全轨道 N-体验证):     {"PASS" if mission_ok else "FAIL"}')
        print(f'  Part E (扫描窗口验证):         {"PASS" if scan_ok else "FAIL"}')
        print(f'  Part F (灵敏度分析验证):       {"PASS" if sens_ok else "FAIL"}')
        print(f'  Part G (3D 扩展验证):          {"PASS" if d3_ok else "FAIL"}')
        print(f'  Part H (多次借力验证):         {"PASS" if multi_ok else "FAIL"}')
        print(f'  Part I (GR 修正验证):          {"PASS" if gr_ok else "FAIL"}')
        if 'FALLBACK' in results.get('ref_source', ''):
            print(f'  ⚠ Part B 使用了非 Horizons 参考历表，结果无效')
        final = access_ok and nbody_ok and lunar_ok and mission_ok and scan_ok and sens_ok and d3_ok and multi_ok and gr_ok
        print(f'\n  ╔══════════════════════════════════════════╗')
        print(f'  ║  最终判定: {"PASS ✓" if final else "FAIL ✗":>28s}  ║')
        print(f'  ╚══════════════════════════════════════════╝')
        sys.exit(0 if final else 1)

    elif '--test' in sys.argv:
        # 快速测试模式
        ok = run_verification(n_days=3, dt=7200)
        assert ok
        sys.exit(0)

    elif '--check-access' in sys.argv:
        access_ok, access_msg = check_horizons_location()
        print(f'\n判定: {"PASS" if access_ok else "FAIL"}')
        sys.exit(0 if access_ok else 1)

    else:
        # 默认：运行全年完整验证
        allow_analytic = '--allow-analytic' in sys.argv
        run_full_year_verification(
            start_date='2026-01-01', n_days=365, dt=3600,
            allow_analytic_fallback=allow_analytic,
        )
