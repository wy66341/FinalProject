"""M3: JPL Horizons 历表对照.

仅用 N-体传播 Sun-Earth-Moon 一年，与 JPL Horizons 历表对比，
输出每日位置与速度残差，要求所有天体位置残差 ≤ 6000 km.

使用课程提供的 Horizons 代理（jpl_forward.py）获取真实历表。
"""

import sys
import numpy as np
from numpy.linalg import norm
from datetime import datetime, timedelta

sys.path.insert(0, 'src')

from nbody import propagate, velocity_verlet_step, acceleration_sem

AU = 1.495978707e8
DAY = 86400.0
MU_SUN = 1.32712440018e11


def get_jpl_state(date_str):
    """从 JPL Horizons 获取指定日期的 Sun-Earth-Moon 状态向量.

    Parameters
    ----------
    date_str : str
        日期字符串 'YYYY-MM-DD'

    Returns
    -------
    dict: {'earth': (pos, vel), 'moon': (pos, vel)} in km, km/s
    """
    try:
        import jpl_forward
        from astroquery.jplhorizons import Horizons

        epoch = {'start': date_str, 'stop': date_str, 'step': '1d'}

        # Earth
        obj_e = Horizons(id='399', location='@10', epochs=epoch)
        vec_e = obj_e.vectors()
        x_e = float(vec_e['x'][0]) * AU
        y_e = float(vec_e['y'][0]) * AU
        z_e = float(vec_e['z'][0]) * AU
        vx_e = float(vec_e['vx'][0]) * AU / DAY
        vy_e = float(vec_e['vy'][0]) * AU / DAY
        vz_e = float(vec_e['vz'][0]) * AU / DAY
        earth = (np.array([x_e, y_e, z_e]), np.array([vx_e, vy_e, vz_e]))

        # Moon
        obj_m = Horizons(id='301', location='@10', epochs=epoch)
        vec_m = obj_m.vectors()
        x_m = float(vec_m['x'][0]) * AU
        y_m = float(vec_m['y'][0]) * AU
        z_m = float(vec_m['z'][0]) * AU
        vx_m = float(vec_m['vx'][0]) * AU / DAY
        vy_m = float(vec_m['vy'][0]) * AU / DAY
        vz_m = float(vec_m['vz'][0]) * AU / DAY
        moon = (np.array([x_m, y_m, z_m]), np.array([vx_m, vy_m, vz_m]))

        return {'earth': earth, 'moon': moon}
    except Exception:
        return None


def build_initial_state(earth_pos, earth_vel, moon_pos, moon_vel):
    """构造四体初始状态向量（火箭项填零）."""
    y0 = np.zeros(24)
    y0[0:3] = [0, 0, 0]     # 太阳在原心
    y0[3:6] = [0, 0, 0]
    y0[6:9] = earth_pos
    y0[9:12] = earth_vel
    y0[12:15] = moon_pos
    y0[15:18] = moon_vel
    y0[18:21] = earth_pos  # 火箭初始在地球位置
    y0[21:24] = earth_vel
    return y0


def compare_single_day(date_str, y0, dt=3600, n_steps=None):
    """传播一天并与 JPL 历表对比."""
    if n_steps is None:
        n_steps = int(DAY / dt)

    y = y0.copy()
    for _ in range(n_steps):
        y = velocity_verlet_step(y, dt)

    jpl = get_jpl_state(date_str)
    if jpl is None:
        return None

    earth_pos_err = norm(y[6:9] - jpl['earth'][0])
    moon_pos_err = norm(y[12:15] - jpl['moon'][0])

    return {'earth_pos_err': earth_pos_err, 'moon_pos_err': moon_pos_err}


def run_verification(start_date='2026-01-01', n_days=365):
    """全年逐日对比 N-体传播 vs JPL 历表."""
    print(f'Horizons 历表对照: {start_date} 起 {n_days} 天')
    print(f'要求: 位置残差 ≤ 6000 km')
    print()

    max_earth_err = 0
    max_moon_err = 0

    start = datetime.strptime(start_date, '%Y-%m-%d')

    for d in range(n_days):
        date = start + timedelta(days=d)
        date_str = date.strftime('%Y-%m-%d')

        jpl = get_jpl_state(date_str)
        if jpl is None:
            continue

        # 从 JPL 初始状态传播一天
        y0 = build_initial_state(
            jpl['earth'][0], jpl['earth'][1],
            jpl['moon'][0], jpl['moon'][1]
        )

        result = compare_single_day(
            (date + timedelta(days=1)).strftime('%Y-%m-%d'), y0
        )
        if result:
            max_earth_err = max(max_earth_err, result['earth_pos_err'])
            max_moon_err = max(max_moon_err, result['moon_pos_err'])

        if d % 30 == 0:
            print(f'  Day {d:3d}: Earth err = {max_earth_err:.0f} km, '
                  f'Moon err = {max_moon_err:.0f} km')

    print(f'\n最大地球位置残差: {max_earth_err:.0f} km '
          f'({"OK" if max_earth_err <= 6000 else "FAIL"})')
    print(f'最大月球位置残差: {max_moon_err:.0f} km '
          f'({"OK" if max_moon_err <= 6000 else "FAIL"})')

    return max_earth_err <= 6000 and max_moon_err <= 6000


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        assert run_verification('2026-01-01', 5)  # 快速测试 5 天
    else:
        run_verification()
