# 强化学习迷宫求解器（MC & DQN）

一个用于求解 5x5 迷宫的强化学习小型项目，包含两种方法：
- MC（蒙特卡洛控制）：基于回合采样、ε-贪婪策略与策略改进
- DQN（深度 Q 网络）：经验回放、目标网络、ε-贪婪策略

项目以命令行（CLI）作为统一入口，并提供统一的环境构建接口，便于新同学快速上手与对比两种方法的效果。

## 运行环境与依赖

- Python >= `3.12`
- 依赖（见 `pyproject.toml`）：
  - `torch>=2.0.0`
  - `click>=8.0.0`
  - `numpy>=1.26`

安装依赖（使用 uv）：

```
uv sync
```

## 快速开始

打印命令帮助（maze 子项目）：

```
uv run maze/main.py --help
```

运行蒙特卡洛（MC）：

```
uv run maze/main.py mc
```

运行 DQN：

```
uv run maze/main.py dqn
```

程序启动后会先打印迷宫布局（S=起点，G=终点，X=障碍，·=空），随后输出训练摘要与最终策略评估结果。

## 输出与评估说明

- 迷宫布局：初始化后打印网格结构，便于确认地图与障碍。
- 训练摘要：
  - MC：每隔固定回合打印摘要（轨迹长度、当回合 Q 更新数量、涉及状态数量、Q 统计值）。
  - DQN：每隔固定回合打印回合回报、平均损失、当前 ε 值、缓冲区大小。
- 策略评估：打印“是否到达终点、步数、累计奖励”，并渲染最终策略棋盘（↑↓←→）。

## 代码结构

- `maze/main.py`
  - 使用 `click` 提供统一命令入口：`mc` 与 `dqn`
  - 调用统一环境构建接口 `build_env()`，在初始化后打印迷宫布局
- `maze/maze_env.py`
  - `MazeEnv`：离散网格环境（状态集合、动作集合、奖励与终止条件）
  - `build_env`：统一的环境构建接口（可通过参数定制尺寸、障碍与奖励）
  - `render_maze`：打印迷宫网格
  - `render_policy`：根据策略字典可视化策略棋盘
- `maze/mc_agent.py`
  - `MCConfig`：蒙特卡洛算法配置（`epsilon`、`gamma`、`episodes`、`max_steps` 等）
  - `MCAgent`：
    - 轨迹为四元组 `(state, action, reward, next_state)`
    - 单循环（反向遍历轨迹）累计回报并更新 `Q(s,a)`
    - 贪心策略改进：按最新 Q 值刷新 `Pi(state)`
  - `evaluate_policy`：从起点出发按最终策略执行，并打印评估结果
- `maze/mc_train.py`
  - `train_to_coverage`：按设定回合数执行训练（已简化，不再做覆盖统计），训练结束后打印最终 Q、Pi 与策略棋盘
- `maze/dqn_agent.py`
  - `ReplayBuffer`：经验回放（`state_idx, action_idx, reward, next_state_idx, done`）
  - `QNetwork`：MLP 模型，将独热编码状态映射为四个动作的 Q 值
  - `DQNConfig`：DQN 超参数（`gamma`、`epsilon_*`、`lr`、`batch_size`、`episodes` 等）
  - `DQNAgent`：
    - `select_action(state)`：ε-贪婪选择动作
    - `push_transition(state, action, reward, next_state, done_flag)`：写入回放缓冲区
    - `update()`：采样批次、计算目标值并执行一次梯度更新
- `maze/dqn_train.py`
  - `train(agent)`：标准训练循环（中文注释说明每一步）
  - `evaluate_policy(env, agent)`：贪心评估，返回 `(是否到达终点, 步数, 累计奖励)`
  - `print_final_policy(env, agent)`：打印最终贪心策略棋盘


## 统一的环境初始化接口（`build_env`）

参数：
- `width, height`：网格尺寸（默认 5x5）
- `start, goal`：起点与终点坐标，形如 `(row, col)`
- `obstacles`：障碍集合 `{(row, col), ...}`（默认采用项目内的标准布局）
- `step_reward`：每步奖励（默认 `-1.0`）
- `goal_reward`：到达终点奖励（默认 `10.0`）
- `invalid_penalty`：无效动作惩罚（撞墙，默认 `-5.0`）

你可以在 `main.py` 中显式传入 `build_env` 参数来修改地图与奖励设置。

## 设计与约定

- 命令参数简化：不暴露训练超参数给用户，使用代码中的默认配置，保持易用与可复现。
- 变量命名规范：尽量避免单字符变量名，使用有语义的名称（如 `state/action/next_state/reward`）。
- 中文注释：关键逻辑（轨迹、Q 更新、目标值与评估）配有中文注释，便于初学者理解。
- 日志策略：
  - MC：按间隔打印摘要日志，不逐条打印细节，避免刷屏。
  - DQN：按间隔打印训练回合的核心指标。

## 常见问题

- 运行时出现 NumPy 初始化警告（`Failed to initialize NumPy`）？
  - 已将 `numpy` 加入依赖；若仍有警告，先执行 `uv sync` 同步依赖。
- `torch` 未安装或版本不匹配？
  - 执行 `uv sync`；或检查 `pyproject.toml` 的 `requires-python` 与 `torch` 版本兼容性。
- 如何修改迷宫？
  - 在 `main.py` 的命令函数内，调整传给 `build_env()` 的参数（如 `obstacles` 布局）。

## 开发者提示

- 代码风格：保持变量名清晰、中文注释到位、输出信息简洁。
- 扩展方向：
  - 新增算法（如 SARSA/Q-learning），复用 `build_env` 与 CLI 结构。
  - 增加更复杂的地图与奖励设计，比较不同算法在复杂任务上的表现。
  - 新增子项目目录（例如 `openai/cartpole/`），约定每个子项目内包含 `main.py` 与其算法/环境实现；在根目录使用 `uv run openai/cartpole/main.py ...` 运行。
