多智能体协同路径规划与控制
---
本项目采用   A* 全局路径规划 + DWA 局部避障 + 内置集中控制的传统控制方案，实现三个差分驱动小车从起点至目标点的三角形编队保持与避障导航。完全基于 Python 和 NumPy，无需深度学习框架。

---
安装
```bash
git clone https://github.com/C1ancyX/08012302-multi-agent-sim.git
cd 08012302-multi-agent-sim
pip install -r requirements.txt
```
依赖
- Python 3.8+
- numpy
- pygame
- shapely

---
 快速开始
运行默认仿真（领航者导航 + 编队跟随）
```bash
python main.py
```

---
算法说明
1. 全局路径规划（A*）
领航者（车2）使用 A* 在栅格地图上规划从起点到终点的无碰撞路径。
每 50 步重新规划一次，适应动态障碍物。
路径点通过纯追踪（Pure Pursuit）转化为实时目标点。
2. 局部避障（DWA – 动态窗口法）
领航者  ：使用 DWA 在速度空间采样，通过朝向代价 + 避障代价 + 速度代价,选择最优速度指令。
跟随者  ：同样配备 DWA，仅在检测到前方障碍物时启用，目标点设为期望队形位置，实现独立避障。
3. 编队控制
领航者自由导航，跟随者（车1和车3）采用 距离‑角度保持控制  。
期望队形：等边三角形，跟随者位于领航者右后方 10°   (车1) 和   左后方 10°   (车3)，距离 1.2 米。
控制律：线速度 = 领航者速度 + 距离误差比例项；角速度 = 朝向误差 + 角度误差比例项，经限幅输出。
 4. 动力学模型
每个机器人独立封装（`models/robot1.py`、`robot2.py`、`robot3.py`），支持质量和阻尼参数。
采用  运动学模型  （直接积分线速度和角速度），保证响应平滑。
5. 性能评估
`evaluation/metrics.py` 提供 30+ 指标，包括：
任务完成度（到达时间、最终误差）
轨迹 RMSE、路径效率
控制平稳性（速度/角速度方差、加速度方差）
编队误差（成对距离偏差）
碰撞次数、危险时长
---
命令行选项

目前主程序 `main.py` 未使用 argparse，您可以直接修改 `main.py` 中的环境参数，或使用脚本参数调用（可自行添加）。
 环境参数（在 `MultiAgentEnv` 中设置）
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_agents` | 3 | 智能体数量 |
| `dt` | 0.02 | 仿真步长 (s) |
| `max_steps` | 500 | 每回合最大步数 |
| `num_obstacles` | 25 | 随机障碍物数量 |
| `enable_obstacles` | True | 是否启用障碍物 |
| `goal_position` | (25,25) | 目标点坐标 (m) |
控制参数（在 `env.py` 中调整）
-   DWA 参数  ：`max_v`, `max_w`, `predict_time`, `resolution`, `heading_gain`, `dist_gain`, `safe_dist`
-   编队参数  ：`rho_des` (期望距离), `phi_des` (期望角度), `k_rho`, `k_phi`
---
代码结构

```
08012302-multi-agent-sim/
├── models/
│   ├── robot1.py             车1动力学模型
│   ├── robot2.py             车2（领航者）动力学模型
│   └── robot3.py             车3动力学模型
├── control/
│   ├── dwa.py                动态窗口法局部避障
│   └── leader_follower.py    Leader-Follower编队控制器 //停用
├── planning/
│   └── astar.py              A* 全局路径规划
├── simulation/
│   ├── env.py                多智能体环境（核心）
│   └── renderer.py           Pygame可视化渲染器
├── evaluation/
│   └── metrics.py            性能指标收集与计算
├── main.py                   主仿真程序
├── requirements.txt
└── README.md
```
---
运行示例
默认仿真（无手动干预）
```bash
python main.py
```
- 领航者自动向 (25,25) 移动，跟随者保持横排编队。
- 遇到障碍物时，DWA 引导绕行。
- 控制台打印领航者速度，并生成性能指标 JSON。
评估单个回合仿真结束后，`logs/` 目录下会生成 `episode_0_metrics.json`，包含全部性能指标。
---
常见问题

Q: 领航者为什么有时原地转圈？    
A: 可能是 A* 路径的第一个点方向与车辆朝向偏差过大，增大目标增益即可。
Q: 跟随者避障刮蹭障碍物    
A: 请增大 `follower_dwa` 的 `safe_dist` 到 1.5 以上，并确保 `resolution` 不小于 0.2。同时提高 `dist_gain`。
Q: 编队保持不好（三角形变形）    
A: 提高 `LeaderFollower` 中的 `k_phi`（角度增益）和角速度限幅；确保 `_follower_control` 未被使用（应使用 `LeaderFollower.compute_control`）。
Q: 性能指标中 `success` 为 false 但实际已到达    
A: 确保 `env._get_info()` 返回了 `'goal_radius'` 字段。

未来改进方向

- 引入模型预测控制（MPC）替代 DWA，提高避障平滑性。
- 为跟随者增加动态编队调整 （如通过障碍区时临时收缩队形）。
- 支持任意数量智能体 （当前硬编码 3 个）。
- 增加实时绘制队形误差曲线  。
