"""M8: 可视化与展示 — 静态图生成.

Output: orbit.png, delta_v_scan.png, energy_conservation.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json
import os

from conic_patch import AU, R_SUN, helio_ellipse
from optimizer import scan_launch_window

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def plot_orbit(r_p, output='orbit.png'):
    """Plot heliocentric transfer ellipse with Earth orbit."""
    ell = helio_ellipse(r_p)
    a_au = ell['a'] / AU
    e = ell['e']

    fig, ax = plt.subplots(figsize=(8, 8))

    # Transfer ellipse
    theta = np.linspace(0, 2 * np.pi, 500)
    r = a_au * (1 - e**2) / (1 + e * np.cos(theta))
    x, y = r * np.cos(theta), r * np.sin(theta)
    ax.plot(x, y, 'r-', linewidth=1.5, label='Transfer Orbit')

    # Sun
    ax.plot(0, 0, 'yo', markersize=12, markeredgecolor='orange')
    ax.text(0.05, 0.05, 'Sun', fontsize=9, color='orange')

    # Earth orbit
    earth_orbit = Circle((0, 0), 1.0, fill=False, linestyle='--',
                         color='blue', alpha=0.4, linewidth=1)
    ax.add_patch(earth_orbit)

    # Earth at departure
    ax.plot(1.0, 0, 'bo', markersize=6)
    ax.text(1.05, 0.08, 'Earth (departure)', fontsize=8, color='blue')

    # Perihelion
    peri_x = a_au * (1 - e)
    ax.plot(peri_x, 0, 'mo', markersize=6)
    ax.text(peri_x + 0.05, 0.1, 'Perihelion', fontsize=8, color='magenta')

    ax.set_xlabel('x (AU)')
    ax.set_ylabel('y (AU)')
    ax.set_title(f'Heliocentric Transfer Orbit  (r_p = {r_p/AU:.2f} AU)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)

    path = os.path.join(FIGS_DIR, output)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_delta_v_scan(scan_results, output='delta_v_scan.png'):
    """2-panel Delta-v scan plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Delta_v vs day
    ax1 = axes[0]
    valid = [(r['day'], r['Delta_v_total']) for r in scan_results
             if r['all_ok'] and r['Delta_v_total'] < np.inf]
    if valid:
        days, dvs = zip(*sorted(valid))
        ax1.scatter(days, dvs, s=3, alpha=0.6, color='steelblue')
        ax1.set_xlabel('Day of 2026')
        ax1.set_ylabel('Δv_total (km/s)')
        ax1.set_title('Δv_total vs Launch Date')
        ax1.grid(True, alpha=0.3)
        if dvs:
            ax1.set_ylim(min(dvs) - 0.5, min(dvs) + 5)

    # Panel 2: (day, r_p) heatmap
    ax2 = axes[1]
    valid2 = [(r['day'], r['r_p_au'], r['Delta_v_total'])
              for r in scan_results if r['all_ok']]
    if valid2:
        days2, rps, dvs2 = zip(*valid2)
        sc = ax2.scatter(days2, rps, c=dvs2, s=3, cmap='plasma', alpha=0.7)
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label('Δv_total (km/s)')
        ax2.set_xlabel('Day of 2026')
        ax2.set_ylabel('r_p (AU)')
        ax2.set_title('Δv_total (t0, r_p) Contour')

    plt.tight_layout()
    path = os.path.join(FIGS_DIR, output)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_energy_conservation(energy_log, output='energy_cons.png'):
    """Plot energy conservation over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    times = [e[0] / 86400.0 for e in energy_log]  # days
    errors = [e[2] for e in energy_log]

    ax.semilogy(times, errors, 'b-', linewidth=1)
    ax.axhline(1e-6, color='r', linestyle='--', linewidth=1, alpha=0.7,
               label='Tolerance $10^{-6}$')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('|ΔE / E₀|')
    ax.set_title('Energy Conservation — Velocity-Verlet Integrator')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(FIGS_DIR, output)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def plot_residuals(residual_log, output='residuals.png'):
    """Plot Horizons position residuals over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    days = [r['day'] for r in residual_log]
    earth_err = [r['earth_err'] for r in residual_log]
    moon_err = [r['moon_err'] for r in residual_log]

    ax1.plot(days, earth_err, 'b.-', markersize=3)
    ax1.axhline(6000, color='r', linestyle='--', alpha=0.7, label='6000 km limit')
    ax1.set_ylabel('Earth Pos Error (km)')
    ax1.set_title('N-body vs Horizons — Position Residuals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(days, moon_err, 'g.-', markersize=3)
    ax2.axhline(6000, color='r', linestyle='--', alpha=0.7, label='6000 km limit')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Moon Pos Error (km)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    path = os.path.join(FIGS_DIR, output)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def generate_all_figures():
    """Generate all required static figures for the report."""
    os.makedirs(FIGS_DIR, exist_ok=True)
    print('Generating static figures...\n')

    # Orbit plot for r_p = 0.2 AU
    plot_orbit(0.2 * AU, 'orbit_0.2au.png')

    # Orbit plot for r_p = 0.35 AU
    plot_orbit(0.35 * AU, 'orbit_0.35au.png')

    # Delta-v scan
    print('  Running scan (this may take a while)...')
    opt = scan_launch_window(r_p_range=(2 * R_SUN, 0.4 * AU, 30), verbose=False)
    plot_delta_v_scan(opt['all_results'])

    print('\nDone.')


if __name__ == '__main__':
    generate_all_figures()
