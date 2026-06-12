"""O5: Orbit Animation — 30-60 second MP4 with key event labels.

Generates a full trajectory animation:
  - Earth departure
  - Lunar swingby (if enabled)
  - Perihelion passage
  - Earth return
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

from conic_patch import AU, R_SUN, MU_SUN
from nbody import (
    velocity_verlet_step, earth_moon_analytic_state,
    system_energy, DAY,
)
from trajectory import get_ephemeris

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def generate_trajectory(date_str='2026-01-03', dt=3600, n_days=420):
    """Generate a 4-body trajectory for the animation.

    Returns arrays: t, sun_pos, earth_pos, moon_pos, rocket_pos
    """
    ep, ev, mp, mv = get_ephemeris(date_str)

    # Rocket initial state: slower than Earth to fall toward Sun
    v_earth = np.linalg.norm(ev)
    v_rocket = ev * (v_earth - 5.4) / v_earth  # Δv ~ 5.4 km/s

    y = np.zeros(24)
    y[0:3] = [0, 0, 0]
    y[3:6] = [0, 0, 0]
    y[6:9] = ep
    y[9:12] = ev
    y[12:15] = mp
    y[15:18] = mv
    y[18:21] = ep.copy()
    y[21:24] = v_rocket

    steps_per_day = int(DAY / dt)
    total_steps = n_days * steps_per_day

    times = []
    earth_traj = []
    moon_traj = []
    rocket_traj = []

    for step in range(total_steps):
        y = velocity_verlet_step(y, dt)

        if step % (steps_per_day // 4) == 0:  # 4 samples/day
            times.append(step * dt / DAY)
            earth_traj.append(y[6:9].copy())
            moon_traj.append(y[12:15].copy())
            rocket_traj.append(y[18:21].copy())

    return (np.array(times),
            np.array(earth_traj), np.array(moon_traj),
            np.array(rocket_traj), ep, mp)


def create_animation(output='orbit_animation.mp4', duration=45, fps=30):
    """Create the full orbital animation."""
    n_frames = duration * fps

    print('Generating trajectory...')
    t, earth, moon, rocket, ep0, mp0 = generate_trajectory()
    stride = max(1, len(t) // n_frames)
    print(f'  Trajectory: {len(t)} points, stride={stride}, frames={n_frames}')

    # Find perihelion index
    r_rocket = np.linalg.norm(rocket, axis=1)
    peri_idx = np.argmin(r_rocket)
    print(f'  Perihelion at day {t[peri_idx]:.1f}, r={r_rocket[peri_idx]/AU:.3f} AU')

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 8))

    def init():
        for ax in [ax_full, ax_zoom]:
            ax.clear()
            ax.set_aspect('equal')
        return []

    def update(frame):
        i = min(frame * stride, len(t) - 1)

        # --- Full view (heliocentric) ---
        ax_full.clear()
        ax_full.set_aspect('equal')
        lim = 1.8
        ax_full.set_xlim(-lim, lim)
        ax_full.set_ylim(-lim, lim)

        # Sun
        ax_full.plot(0, 0, 'yo', markersize=14, markeredgecolor='orange')

        # Earth orbit reference circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax_full.plot(np.cos(theta), np.sin(theta), '--', color='blue',
                     alpha=0.2, linewidth=0.8)

        # Trails
        ax_full.plot(earth[:i+1, 0] / AU, earth[:i+1, 1] / AU,
                     'b-', alpha=0.3, linewidth=0.5)
        ax_full.plot(moon[:i+1, 0] / AU, moon[:i+1, 1] / AU,
                     color='gray', alpha=0.3, linewidth=0.5)
        ax_full.plot(rocket[:i+1, 0] / AU, rocket[:i+1, 1] / AU,
                     'r-', alpha=0.6, linewidth=1.0)

        # Current positions
        ax_full.plot(earth[i, 0] / AU, earth[i, 1] / AU, 'bo', markersize=6)
        ax_full.plot(moon[i, 0] / AU, moon[i, 1] / AU, 'o', color='gray', markersize=3)
        ax_full.plot(rocket[i, 0] / AU, rocket[i, 1] / AU, 'ro', markersize=5)

        # Labels
        ax_full.set_xlabel('x (AU)')
        ax_full.set_ylabel('y (AU)')
        ax_full.set_title(f'Solar Return Trajectory  —  Day {t[i]:.1f}', fontsize=13)
        ax_full.legend(['Sun', 'Earth orbit', 'Earth', 'Moon', 'Rocket'],
                       loc='upper right', fontsize=8)

        # Key event markers
        if i >= peri_idx:
            rp = rocket[peri_idx]
            ax_full.plot(rp[0] / AU, rp[1] / AU, 'm*', markersize=12)
            ax_full.annotate('Perihelion', (rp[0] / AU, rp[1] / AU),
                             textcoords='offset points', xytext=(10, -15),
                             fontsize=9, color='magenta')

        # --- Zoom view (Earth-Moon neighborhood) ---
        ax_zoom.clear()
        ax_zoom.set_aspect('equal')
        zoom = 0.015  # ~0.015 AU ≈ 2.2M km
        ex, ey = earth[i, 0] / AU, earth[i, 1] / AU
        ax_zoom.set_xlim(ex - zoom, ex + zoom)
        ax_zoom.set_ylim(ey - zoom, ey + zoom)

        # Earth
        ax_zoom.plot(ex, ey, 'bo', markersize=10)
        # Moon
        ax_zoom.plot(moon[i, 0] / AU, moon[i, 1] / AU,
                     'o', color='gray', markersize=6)
        # Rocket
        rx, ry = rocket[i, 0] / AU, rocket[i, 1] / AU
        ax_zoom.plot(rx, ry, 'ro', markersize=5)

        # Moon SOI circle
        from conic_patch import R_MOON_SOI
        moon_soi_au = R_MOON_SOI / AU
        moon_circle = plt.Circle(
            (moon[i, 0] / AU, moon[i, 1] / AU), moon_soi_au,
            fill=False, linestyle=':', color='gray', alpha=0.5
        )
        ax_zoom.add_patch(moon_circle)

        ax_zoom.set_xlabel('x (AU)')
        ax_zoom.set_ylabel('y (AU)')
        ax_zoom.set_title('Earth–Moon Neighborhood (Zoom)', fontsize=11)

        # Progress bar
        progress = i / len(t)
        ax_full.text(0.02, 0.98, f'Progress: {progress*100:.0f}%',
                     transform=ax_full.transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        return []

    print(f'Rendering {n_frames} frames...')
    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, output)
    writer = FFMpegWriter(fps=fps, metadata=dict(
        title='Solar Return Trajectory with Lunar Swingby',
        artist='Wang Yue'),
        bitrate=2000)
    anim.save(outpath, writer=writer, dpi=120)
    plt.close(fig)

    size_mb = os.path.getsize(outpath) / (1024 * 1024)
    print(f'Saved: {outpath} ({size_mb:.1f} MB)')
    return outpath


if __name__ == '__main__':
    create_animation()
