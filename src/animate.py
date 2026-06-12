"""O5: 轨道动画生成 (MP4).

生成 30-60 秒的轨道动画，含关键事件 zoom-in：
  - 地球出发
  - 月球借力
  - 近日点经过
  - 地球再入
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from conic_patch import AU
from nbody import propagate, acceleration_sem


def create_animation(y0, dt, n_steps, output='animation.mp4', duration=45, fps=30):
    """创建轨道动画.

    Parameters
    ----------
    y0 : ndarray
        初始状态
    dt : float
        步长 (s)
    n_steps : int
        总步数
    output : str
        输出文件名
    duration : float
        动画时长 (s)
    fps : int
        帧率
    """
    n_frames = duration * fps
    stride = max(1, n_steps // n_frames)

    # 预计算轨迹
    traj = propagate(y0, dt, n_steps, accel_func=acceleration_sem)

    fig, ax = plt.subplots(figsize=(10, 10))

    # 轨道绘制
    earth_traj = np.array([[t[1][6], t[1][7]] for t in traj])
    moon_traj = np.array([[t[1][12], t[1][13]] for t in traj])
    rocket_traj = np.array([[t[1][18], t[1][19]] for t in traj])

    def init():
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        ax.set_title('Solar Return Trajectory')
        return []

    def update(frame):
        ax.clear()
        ax.set_aspect('equal')

        i = frame * stride
        if i >= len(earth_traj):
            i = len(earth_traj) - 1

        ax.plot(earth_traj[:i+1, 0], earth_traj[:i+1, 1], 'b-', alpha=0.3, label='Earth')
        ax.plot(moon_traj[:i+1, 0], moon_traj[:i+1, 1], 'gray', alpha=0.3, label='Moon')
        ax.plot(rocket_traj[:i+1, 0], rocket_traj[:i+1, 1], 'r-', label='Rocket')

        ax.plot(0, 0, 'yo', markersize=10, label='Sun')
        ax.plot(earth_traj[i, 0], earth_traj[i, 1], 'bo', markersize=6)
        ax.plot(moon_traj[i, 0], moon_traj[i, 1], 'o', color='gray', markersize=4)
        ax.plot(rocket_traj[i, 0], rocket_traj[i, 1], 'ro', markersize=4)

        ax.set_xlim(-2 * AU, 2 * AU)
        ax.set_ylim(-2 * AU, 2 * AU)
        ax.legend(loc='upper right')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False)
    anim.save(output, writer='ffmpeg', fps=fps, dpi=150)
    plt.close()
    print(f'Saved animation to {output}')


if __name__ == '__main__':
    print('Animation module loaded. Call create_animation() with initial state.')
