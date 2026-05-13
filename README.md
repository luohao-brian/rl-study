# 强化学习经典算法实现与对比研究 (RL Study)

本项目针对迷宫路径规划问题，从零实现了强化学习（Reinforcement Learning, RL）的四大基础核心算法矩阵。旨在深入探究并对比基于模型（Model-based）与无模型（Model-free）、同策略（On-Policy）与异策略（Off-Policy）、以及表格型表示与深度网络逼近等不同强化学习范式的收敛特性与工程实现细节。

项目中集成的核心算法模块包括：
- **动态规划（DP, 价值迭代 Value Iteration）**
- **蒙特卡洛控制（MC, Monte Carlo Control）**
- **时间差分控制（TD, 涵盖 Q-Learning 与 Sarsa）**
- **深度 Q 网络（DQN, Deep Q-Network）**

---

## 📖 核心概念映射 (Core Concepts)

在具体的迷宫寻路设定下，强化学习的核心要素被定义如下：

1. **环境 (Environment / Env)**
   - **状态空间**：5x5 的离散二维网格坐标位置 `(row, col)`。由底层 `MazeEnv` 类统一维护转移逻辑及边界碰撞检测。
   - **动作空间**：离散动作集合，包含四个朝向移动：`上 (Up)`、`下 (Down)`、`左 (Left)`、`右 (Right)`。
   - **奖励函数 (Reward)**：
     - 单步常规移动：`-1.0`（引导算法寻优最短路径）
     - 障碍物/边界越界碰撞：`-5.0`（惩罚无效动作）
     - 到达目标状态：`+10.0`（触发回合终止与高额回报）

2. **策略 (Policy / $\pi$)**
   - 决定特定状态下动作选择的概率分布或确定性映射。表格型方法内部维护策略字典映射（如 `self.policy[state] = action`），深度方法通过输出层激活值代表策略。

3. **价值函数 (Value Function / Q-Table)**
   - 评估特定状态或状态-动作对的长期期望回报。表格型控制算法统一维护 `self.Q[(state, action)]` 字典作为核心载体。

---

## 🧠 算法设计架构与关键代码剖析

### 1. 动态规划 (DP - 价值迭代 Value Iteration)
**核心机制 (Model-based 上帝视角)**：
动态规划是典型的**基于模型（Model-based）**方法，与无模型（Model-free）范式存在根本性分界：
- **Model-free 视角**：智能体将环境视为未知黑盒。仅能在当前单一物理状态调用 `step()` 产生真实状态转移（副作用），依赖实际试错采集的时间序列轨迹来近似估值。
- **Model-based 视角**：智能体拥有环境的全量先验动态机制（转移函数与奖励映射）。DP 智能体直接跳过低效的物理试错：
  1. **全景状态特权**：直接读取底层静态清单 `self.env.states` 进行全状态遍历；
  2. **静态规则提问**：将 `self.env.step(s, a)` 作为纯函数式计算器调用。因 `step()` 底层仅做坐标增量与静态障碍物集合校验，无任何实例状态覆盖操作，从而充当了完美的【无副作用确定性转移查询接口】。

**关键结构与代码逻辑**：
- `DPAgent` 维护状态价值字典 `self.V`。
- 离线全盘推演：对每个状态向计算器传入四个动作得出目标后继，执行单步绝对更新：
  $$V(s) \leftarrow \max_{a \in \mathcal{A}} \left[ \mathcal{R}_s^a + \gamma V(s') \right]$$
- 当全网格最大价值变动量 $\delta < \theta$ 时判定算法收敛，实时推导出全局确定性贪心策略。


### 2. 蒙特卡洛控制 (MC - Monte Carlo Control)
**核心机制**：无模型（Model-free）算法。依赖完整回合（Episode）结束后的经验采样数据，通过事后计算各状态-动作对的平均经验累计回报更新 Q 估值。

**关键结构与代码逻辑**：
- 采用 ε-贪婪策略进行在线探索并完整记录单局轨迹 `(state, action, reward)` 序列。
- 回合终局后触发 `end_episode` 接口，逆序遍历整条回放轨迹计算折扣累计回报 $G$：
  $$G \leftarrow R_{t+1} + \gamma G$$
- 采用增量式均值更新法直接调整对应键值：
  $$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [G - Q(S_t, A_t)]$$
- **On-Policy 特性**：采样所用的行为策略与被优化的目标策略始终保持一致（均基于当前的 Q 表演化）。

### 3. 时间差分控制 (TD - Q-Learning & Sarsa)
**核心机制**：结合了 MC 的实时模型无关性与 DP 的自举更新（Bootstrapping）特性。每经历单步环境交互即可在线利用后继状态预估价值进行当场更新。

**关键结构与代码逻辑**：
- 核心自举更新范式：
  $$Q(S, A) \leftarrow Q(S, A) + \alpha \left[ R + \gamma Q_{target} - Q(S, A) \right]$$
- **Q-Learning (异策略 Off-Policy)**：
  - 目标构建逻辑：$Q_{target} = \max_{a} Q(S', a)$
  - 独立于后续真实动作，以绝对乐观上界作为更新依据。
- **Sarsa (同策略 On-Policy)**：
  - 目标构建逻辑：$Q_{target} = Q(S', A')$
  - 智能体内部设计了 `_cached_next_action` 缓存机制，确保预采样的自举目标动作与外部训练器下一轮进入迭代时真实消耗的动作分毫不差。

### 4. 深度 Q 网络 (DQN - Deep Q-Network)
**核心机制**：利用深度神经网络逼近高维状态空间的价值分布，通过目标值冻结与经验去相关解决深度网络训练不稳定的痛点。

**关键结构与代码逻辑**：
- **`QNetwork`**：多层全连接网络，将离散空间独热向量映射为对应动作的 Q 预测张量。
- **经验回放池 (`ReplayBuffer`)**：解耦数据的时间序列相关性。每次更新从缓存队列随机抽取 Mini-batch 进行梯度反向传播。
- **目标网络同步**：采用周期性硬更新或软更新平滑目标估计，避免自举震荡。

---

## 🚀 运行环境与启动指南

### 依赖配置
- Python 环境：`>= 3.12`
- 包管理与虚拟环境工具：`uv`
- 核心依赖库：`torch`, `click`, `numpy`

### 环境同步
```bash
uv sync
```

### 命令行执行
项目封装了 Click 命令行统一执行入口，支持动态指定运行模式与算法超参数。

查看全局使用说明：
```bash
uv run main.py --help
```

**运行动态规划 (DP) 算法**：
```bash
uv run main.py dp --mode train --model-path models/dp_model.json
uv run main.py dp --mode eval --model-path models/dp_model.json
```

**运行蒙特卡洛 (MC) 算法**：
```bash
uv run main.py mc --mode train --model-path models/mc_model.json
uv run main.py mc --mode eval --model-path models/mc_model.json
```

**运行时间差分 (TD) 算法**：
```bash
# 执行默认的 Q-Learning 内核
uv run main.py td --algo q_learning --mode train --model-path models/td_model.json

# 执行同策略 Sarsa 内核
uv run main.py td --algo sarsa --mode train --model-path models/td_sarsa_model.json
```

**运行 DQN 算法**：
```bash
uv run main.py dqn --mode train --model-path models/dqn_model.pth
uv run main.py dqn --mode eval --model-path models/dqn_model.pth
```

---

## 📦 模型序列化与架构验证

项目中导出的持久化策略模型分为两类标准存储格式，与底层计算范式强对应：

- **表格型字典结构 (`.json`)**：针对 DP、MC 及 TD 算法，将收敛后的多维键值对（如状态坐标与动作字符串）展平并序列化保存至标准 JSON 文件，支持高效读取并方便排查收敛数值分布。
- **深度张量权重 (`.pth`)**：针对 DQN 算法，使用 PyTorch 原生序列化引擎保存多层神经网络的参数字典。

程序每次完成全量训练迭代后，会自动加载持久化的模型文件执行确定性贪心验证，并在终端绘制出覆盖全状态空间的最终流向策略图表。
