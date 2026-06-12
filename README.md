# 钱学森问题的扩展求解

> 含月球借力与发射窗口优化的绕日返回轨道设计

**课程**：人工智能与数学软件 | **学院**：浙江大学数学科学学院 | **学期**：2025–2026 春季

## 问题概述

在钱学森《星际航行概论》（2008 版）第5–6 章拼接圆锥曲线理论基础上，扩展求解一个完整的工程级行星际轨道问题：

> 火箭从地球出发，借助月球引力做一次速度修正，进入日心椭圆转移轨道；绕过太阳后返回与地球相遇。在 2026 年一年周期内搜索使总速度增量最小的发射窗口。

### 目标函数

$$\Delta v_{\text{total}}(t_0, r_m, \text{side}, r_p) = |\Delta v_{\text{发射}}| + |\Delta v_{\text{月借残差}}| + |\Delta v_{\text{再入}}|$$

### 设计变量

| 变量 | 范围 | 含义 |
|------|------|------|
| $t_0$ | 2026-01-01 – 2026-12-31 | 发射日期 |
| $r_m$ | [1838, 50000] km | 月球借力最近距 |
| side | {leading, trailing} | 绕月方向 |
| $r_p$ | [2 $R_{\odot}$, 0.4 AU] | 近日点距离 |

## 快速开始

### 环境要求

- Python 3.10+
- XeLaTeX（用于编译报告）

### 安装

```bash
git clone https://github.com/wy66341/FinalProject.git
cd FinalProject
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib astroquery astropy
```

### 运行

```bash
make all        # 完整流程：数值计算 + 图表 + 报告 PDF + 轨道动画
make pdf        # 仅编译 report.pdf
make clean      # 清理所有生成文件
make test       # 运行验证测试（二体基准 + Horizons 残差）
```

## 项目结构

```
FinalProject/
├── README.md           # 本文件
├── report.tex          # LaTeX 报告源文件
├── Makefile            # 构建脚本
├── AI-Agent.md         # AI 工具使用记录
├── .gitignore
├── src/
│   ├── conic_patch.py          # M1: 拼接圆锥曲线（report.tex §3–§5 编码）
│   ├── nbody.py                # M2: Sun-Earth-Moon-Rocket 四体积分器
│   ├── horizons_verify.py      # M3: JPL Horizons 历表对照
│   ├── lunar_swingby.py        # M4: 月球借力（解析 + 数值）
│   ├── trajectory.py           # M5: 单点轨道求解
│   ├── optimizer.py            # M6: 一年发射窗口扫描
│   ├── sensitivity.py          # M7: 灵敏度与误差分析
│   ├── visualize.py            # M8: 静态图表生成
│   ├── animate.py              # O5: 轨道动画 (MP4)
│   ├── jpl_forward.py          # Horizons 代理脚本（课程提供）
│   └── JPL_API.env             # 代理 URL + token
└── data/                       # 历表缓存、扫描结果等中间数据
```

## 必做项

| 编号 | 内容 | 说明 |
|------|------|------|
| M1 | 拼接圆锥曲线代码化 | 将 report.tex §3–§5 的步骤编码为可调用函数 |
| M2 | N-体数值积分器 | Velocity-Verlet，四体 Sun-Earth-Moon-Rocket |
| M3 | JPL Horizons 历表对照 | N-体传播 vs 真实历表，残差 ≤ 6000 km |
| M4 | 月球借力实现 | 解析公式 + 数值仿真，双套对比 |
| M5 | 单点轨道求解 | 固定日期完整轨道，三段 Δv 对比 |
| M6 | 最优发射窗口扫描 | 365 天扫描，输出最优解 |
| M7 | 灵敏度分析 | 近月距 / 发射日偏移 / 积分步长 |
| M8 | 可视化展示 | 轨道图、守恒量、扫描曲线、残差 |

## 选做加分项

- 3D 扩展（纳入月球 5.1° 轨道倾角）
- 广义相对论修正（近日点附近 $\sim 3\mu^2/(c^2 r^4)$）
- 自定义微分修正器（Newton-Raphson / Lambert）
- 多次借力轨道（Earth→Moon→Venus→Sun→Earth）
- 轨道动画 MP4（30–60 秒）
- 实时交互演示（Tkinter / Web）

## 参考资料

- 钱学森，《星际航行概论》（2008 年版）第5–6 章
- Vallado D. A., *Fundamentals of Astrodynamics and Applications*, 4th ed., §12.4
- Curtis H. D., *Orbital Mechanics for Engineering Students*, 3rd ed., §8.10
- [JPL Horizons User Manual](https://ssd.jpl.nasa.gov/horizons/manual.html)

## License

MIT
