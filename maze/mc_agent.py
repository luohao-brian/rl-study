"""蒙特卡洛控制智能体（mc_agent）

将原有的 mc_control 中的智能体与策略、Q值更新等逻辑抽离到独立模块，
与 DQN 的 "agent + train" 结构保持一致：

- MCConfig: 超参数配置数据类
- MCAgent: 负责采样、Q值更新、策略管理与评估

训练循环与早停/覆盖等过程由 mc_train.py 提供。
"""

from dataclasses import dataclass
from collections import defaultdict
import random
from typing import List, Tuple
import json
import os
from maze.base import BaseAgent


@dataclass
class MCConfig:
    # 训练与策略相关默认参数（统一在Config中管理）
    epsilon: float = 0.2
    gamma: float = 0.9
    episodes: int = 2000
    max_steps: int = 200
    check_every: int = 20
    seed: int = 42
    exploring_starts: bool = True
    # 日志配置：默认仅摘要，每隔 log_interval 打印一次
    log_interval: int = 20
    print_summary: bool = True


class MCAgent(BaseAgent):
    """蒙特卡洛控制智能体

    实现 BaseAgent 接口。负责：
    - 按 ε-贪婪策略选择动作
    - 记录单步经验到局部轨迹中
    - 回合结束时从轨迹计算回报并更新 Q(s,a)
    - 基于 Q 值生成/更新策略 Pi(s)
    """

    def __init__(self, env, cfg: MCConfig):
        self.env = env
        self.epsilon = cfg.epsilon
        self.gamma = cfg.gamma
        self.episodes = cfg.episodes
        self.max_steps = cfg.max_steps
        self.exploring_starts = cfg.exploring_starts
        self.log_interval = max(1, int(cfg.log_interval))
        self.print_summary = bool(cfg.print_summary)
        # Q 值与统计
        self.Q = defaultdict(float)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.policy = {}
        self.current_trajectory = []
        random.seed(cfg.seed)

    def _greedy_action(self, state):
        best_a = None
        best_q = float('-inf')
        for a in self.env.actions:
            q = self.Q[(state, a)]
            if q > best_q:
                best_q = q
                best_a = a
        if best_a is None:
            best_a = random.choice(self.env.actions)
        return best_a

    def select_action(self, state, is_training: bool = True):
        """实现 BaseAgent 接口：动作选择策略
        
        这里使用的是典型的 ε-贪婪策略 (Epsilon-Greedy)：
        - 在训练模式下，有 ε 的概率随机选择一个动作，目的是“探索”未知的地图区域。
        - 否则（或在测试模式下），查阅策略字典 policy，选择当前已知的最优动作（利用已知最优解）。
        """
        if is_training and random.random() < self.epsilon:
            return random.choice(self.env.actions)
        a = self.policy.get(state)
        if a is None:
            a = self._greedy_action(state)
        return a

    def step(self, state, action, reward, next_state, done):
        """实现 BaseAgent 接口：处理单步经验
        
        在蒙特卡洛方法中，智能体不会走一步学一步。
        它必须先把走的每一步记录下来，串成一条完整的“轨迹 (Trajectory)”。
        """
        self.current_trajectory.append((state, action, reward, next_state))

    def end_episode(self, episode_idx: int):
        """实现 BaseAgent 接口：回合结束时的核心学习逻辑
        
        当一局游戏结束（到达终点或步数耗尽），MC 算法开始“事后诸葛亮”式的反思：
        1. 逆序遍历这条轨迹，从最后一步开始往前倒推。
        2. 计算累计回报 G（当前的 reward + 未来的 G 折扣）。
        3. 用这个实际获得的 G 去更新当前 (状态, 动作) 在 Q-Table 中的平均分。
        """
        G = 0.0
        visited_states = set()
        updated_keys = []
        # 反向遍历
        for state, action, reward, _ in reversed(self.current_trajectory):
            G = reward + self.gamma * G
            key = (state, action)
            self.returns_sum[key] += G
            self.returns_count[key] += 1
            self.Q[key] = self.returns_sum[key] / self.returns_count[key]
            visited_states.add(state)
            updated_keys.append(key)

        # 刷新策略（贪心）
        for state in visited_states:
            self.policy[state] = self._greedy_action(state)

        # 摘要日志
        if self.print_summary and (episode_idx % self.log_interval == 0):
            total_pairs = len(updated_keys)
            total_states = len(visited_states)
            q_values_updated = [self.Q[k] for k in updated_keys]
            if q_values_updated:
                q_mean = sum(q_values_updated) / len(q_values_updated)
                q_min = min(q_values_updated)
                q_max = max(q_values_updated)
            else:
                q_mean = q_min = q_max = float('nan')
            print(
                f"[第 {episode_idx} 回] len={len(self.current_trajectory)} Q更新={total_pairs} Pi更新={total_states} "
                f"Q统计(mean={q_mean:.3f}, min={q_min:.3f}, max={q_max:.3f})"
            )
            
        self.current_trajectory = []

    def save(self, path: str):
        """将 Q-Table 和 Policy 保存为 JSON 格式"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 将 (tuple) 键转换为字符串键
        q_serializable = {f"{s[0]},{s[1]},{a}": v for (s, a), v in self.Q.items()}
        policy_serializable = {f"{s[0]},{s[1]}": a for s, a in self.policy.items()}
        
        data = {
            "Q": q_serializable,
            "policy": policy_serializable
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"MC模型已保存至: {path}")

    def load(self, path: str):
        """从 JSON 文件加载 Q-Table 和 Policy"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.Q = defaultdict(float)
        for k, v in data["Q"].items():
            r, c, a = k.split(',')
            self.Q[((int(r), int(c)), a)] = float(v)
            
        self.policy = {}
        for k, v in data["policy"].items():
            r, c = k.split(',')
            self.policy[(int(r), int(c))] = v
            
        print(f"MC模型已从 {path} 加载")
