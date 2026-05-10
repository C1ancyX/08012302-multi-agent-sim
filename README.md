  多智能体深度确定性策略梯度（MADDPG）

本项目实现了论文《Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments》中的MADDPG算法。配置为与自建的多智能体协同路径规划与控制环境协同运行，支持3个差分驱动机器人在2D平面内从起点出发，保持编队并避开障碍物到达目标点。算法基于PyTorch实现，环境使用Pygame进行可视化。

  注意  ：该代码库为项目重构版本，训练结果可能与论文报道有所不同。代码按原样提供。

---

安装

克隆仓库
```bash
git clone https://github.com/C1ancyX/08012302-multi-agent-sim.git
cd 08012302-multi-agent-sim
```

安装依赖
```bash
pip install -r requirements.txt
```

已知依赖
- Python (3.8+)
- PyTorch (2.0.0)
- numpy (1.24.0)
- pygame (2.5.0)
- shapely (2.0.0)

---

案例研究：多智能体编队控制环境

我们在此演示代码如何与自建的多智能体环境协同工作。环境包含：
- 3个矩形智能体（长0.4m，宽0.2m），差分驱动运动学模型
- 随机生成的圆形障碍物（可选）
- 碰撞检测与安全分级（绿/黄/红）
- 手动控制模式（键盘操作）
- Pygame可视化窗口（1024×768，跟随领航者视角）

运行训练
```bash
python main.py --num_episodes 500 --render
```

手动控制演示
```bash
python demo_manual.py
```

测试环境接口
```bash
python test_interface.py
```

---

命令行选项

环境选项
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_agents` | 智能体数量 | 3 |
| `--dt` | 仿真时间步长 (s) | 0.1 |
| `--max_steps` | 每回合最大步数 | 300 |
| `--obstacles` | 是否启用随机障碍物 | True |

训练参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_episodes` | 训练总回合数 | 5000 |
| `--batch_size` | 批大小 | 256 |
| `--buffer_capacity` | 经验池容量 | 1,000,000 |
| `--lr_actor` | Actor学习率 | 1e-3 |
| `--lr_critic` | Critic学习率 | 1e-3 |
| `--gamma` | 折扣因子 | 0.99 |
| `--tau` | 软更新系数 | 0.01 |
| `--noise_scale` | 动作探索噪声标准差 | 0.1 |

系统选项
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--render` | 启用Pygame可视化 | False |
| `--resume` | 从检查点继续训练 | False |
| `--save_interval` | 保存模型的回合间隔 | 100 |

---

代码结构

```
08012302-multi-agent-sim/
├── main.py                        训练主脚本
├── demo_manual.py                 手动控制演示
├── test_interface.py              环境接口测试
├── simulation/
│   ├── env.py                     多智能体环境（运动学、碰撞、奖励）
│   ├── renderer.py                Pygame可视化渲染器
│   └── manual_controller.py       键盘手动控制
├── agents/
│   ├── maddpg_agent.py            MADDPG智能体实现
│   ├── network.py                 Actor/Critic网络定义
│   └── replay_buffer.py           经验回放缓冲区
├── evaluation/
│   └── metrics.py                 性能评估指标
├── models/                        保存模型权重
├── buffers/                       保存经验池
├── logs/                          训练日志和指标
└── requirements.txt               Python依赖列表
```

核心文件说明

-   `simulation/env.py`  ：定义多智能体环境，包括车辆运动学、碰撞检测、奖励函数、障碍物生成。提供`reset()`、`step()`、`get_state_dim()`等标准接口。
-   `agents/network.py`  ：`Actor`和`Critic`网络结构。`Actor`输入局部观测(3维)，输出动作(2维)；`Critic`输入全局状态(9维)和所有动作(6维)，输出联合Q值。
-   `agents/maddpg_agent.py`  ：实现MADDPG智能体，包含动作选择、Actor/Critic更新、目标网络软更新、模型保存/加载。
-   `main.py`  ：训练循环，协调环境与多个智能体交互，使用经验回放和集中式Critic更新。

---

检查点与恢复训练

训练过程中自动保存：
- 每100回合保存模型到 `models/checkpoint_{episode}_actor_{i}.pth`
- 经验池保存到 `buffers/replay_buffer_{episode}.pkl`
- 最佳模型保存到 `models/best_actor_{i}.pth`

恢复训练
```bash
python main.py --resume --render
```

仅评估不训练
修改`main.py`中的`num_episodes=0`，或使用独立的评估脚本`evaluate.py`（需自行创建）。

---

可视化与手动控制

    实时训练可视化
```bash
python main.py --render --num_episodes 100
```

窗口中显示：
- 三辆智能体矩形轮廓（带安全颜色）
- 障碍物（棕色圆圈）
- 目标点（绿色圆）
- 每辆车的状态信息（位置、速度、模式、安全信号）
- 碰撞警告（红色感叹号）

    手动模式
在`demo_manual.py`中启用手动模式，可通过键盘控制车辆：
- `1/2/3`：选择车辆
- `W/S`：加减速
- `A/D`：转向
- `4`：团队移动
- `M`：切换自动/手动
- `R`：重置环境
- `ESC`：退出

---

性能评估

训练指标自动保存在`logs/training_metrics_*.json`，包含：
- 每回合总奖励
- 每回合步数
- 最佳奖励及对应的模型

运行后绘图：
```bash
python evaluation/plotter.py     需自行实现
```

---

常见问题

Q: 训练时车辆不动？    
A: 检查`start_stop`标志是否为1（默认是1）。也可在`env.py`中打印`set_v`值确认动作被应用。

Q: 碰撞检测不准确？    
A: 确保`shapely`已正确安装。矩形车辆多边形计算正确性可通过调试`_get_vehicle_polygon`验证。

Q: 如何修改障碍物数量？    
A: 在创建环境时传入`num_obstacles`参数，例如`MultiAgentEnv(num_obstacles=8)`。

Q: 训练不收敛（奖励始终为负）？    
A: 尝试增加`--num_episodes`，降低`--lr_actor`/`--lr_critic`（如1e-4），或调整`--noise_scale`（如0.2）。

---

引用

如果本项目对您的研究有帮助，请引用原始MADDPG论文：

```
@article{lowe2017multi,
  title={Multi-agent actor-critic for mixed cooperative-competitive environments},
  author={Lowe, Ryan and Wu, Yi and Tamar, Aviv and Harb, Jean and Abbeel, Pieter and Mordatch, Igor},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```
