"""M8: 可视化与展示 — 静态图生成.

生成图表：
  1. 轨道图（日心椭圆 + 月球借力 zoom-in）
  2. 能量守恒监测
  3. Delta_v 扫描曲线
  4. Horizons 残差
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conic_patch import AU, R_SUN
from optimizer import scan_launch_window


def plot_orbit(ax, ellipse_params, label='Transfer'):
    """绘制日心椭圆轨道."""
    a = ellipse_params['a'] / AU
    e = ellipse_params['e']

    theta = np.linspace(0, 2 * np.pi, 500)
    r = a * (1 - e**2) / (1 + e * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    ax.plot(x, y, label=label)
    ax.plot(0, 0, 'yo', markersize=8, label='Sun')
    ax.plot(1, 0, 'bo', markersize=4, label='Earth orbit')
    ax.plot(a * (1 - e), 0, 'ro', markersize=4, label='Perihelion')

    # 地球轨道圆
    earth_circle = plt.Circle((0, 0), 1.0, fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(earth_circle)

    ax.set_aspect('equal')
    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.legend()
    ax.set_title('Heliocentric Transfer Orbit')


def plot_delta_v_scan(scan_results):
    """绘制 Delta_v 扫描曲线."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    days = [r['day'] for r in scan_results]
    dvs = [r['Delta_v'] for r in scan_results]
    rps_au = [r['r_p_au'] for r in scan_results]

    # Delta_v vs day
    ax1 = axes[0]
    valid = [(d, dv) for d, dv in zip(days, dvs) if dv < np.inf]
    if valid:
        d_vals, dv_vals = zip(*valid)
        ax1.scatter(d_vals, dv_vals, s=2, alpha=0.5)
        ax1.set_xlabel('Day of 2026')
        ax1.set_ylabel('Δv_total (km/s)')
        ax1.set_title('Δv_total vs Launch Date')

    # (t0, r_p) contour
    ax2 = axes[1]
    valid2 = [(d, rp, dv) for d, rp, dv in zip(days, rps_au, dvs) if dv < np.inf]
    if valid2:
        d_vals2, rp_vals2, dv_vals2 = zip(*valid2)
        sc = ax2.scatter(d_vals2, rp_vals2, c=dv_vals2, s=2, cmap='viridis')
        plt.colorbar(sc, ax=ax2, label='Δv_total (km/s)')
        ax2.set_xlabel('Day of 2026')
        ax2.set_ylabel('r_p (AU)')
        ax2.set_title('Δv_total (t0, r_p)')

    plt.tight_layout()
    plt.savefig('delta_v_scan.png', dpi=150)
    plt.close()


def plot_energy_conservation(energy_log):
    """绘制能量守恒监测."""
    fig, ax = plt.subplots(figsize=(8, 4))
    times = [e[0] for e in energy_log]
    errors = [e[1] for e in energy_log]
    ax.semilogy(times, errors)
    ax.axhline(1e-6, color='r', linestyle='--', label='Tolerance 1e-6')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|ΔE/E|')
    ax.set_title('Energy Conservation')
    ax.legend()
    plt.tight_layout()
    plt.savefig('energy_conservation.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    print('Generating static figures...')

    # Orbit plot
    fig, ax = plt.subplots(figsize=(8, 8))
    from conic_patch import helio_ellipse
    ell = helio_ellipse(0.2 * AU, AU)
    plot_orbit(ax, ell)
    plt.savefig('orbit.png', dpi=150)
    plt.close()
    print('  -> orbit.png')

    # Delta-v scan
    result = scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 20), verbose=False)
    plot_delta_v_scan(result['scan_results'])
    print('  -> delta_v_scan.png')

    print('Done.')
