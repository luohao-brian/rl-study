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


class MCAgent:
    """蒙特卡洛控制智能体

    负责：
    - 按 ε-贪婪策略采样轨迹
    - 从轨迹计算回报并更新 Q(s,a)
    - 基于 Q 值生成/更新策略 Pi(s)
    - 提供策略评估与可视化辅助
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

    def _select_action(self, state):
        """根据当前策略与ε-贪婪选择动作，仅返回动作字符。"""
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        a = self.policy.get(state)
        if a is None:
            a = self._greedy_action(state)
        return a

    def _simulate_episode(self) -> List[Tuple[tuple, str, float, tuple]]:
        """
        采样一个完整回合，返回轨迹：[(state, action, reward, next_state), ...]
        """
        trajectory = []
        # Exploring Starts：随机起点+随机首动作，增加覆盖
        if self.exploring_starts:
            candidates = [st for st in self.env.states if st != self.env.goal]
            state = random.choice(candidates)
            first_action = random.choice(self.env.actions)
            next_state, reward, done = self.env.step(state, first_action)
            trajectory.append((state, first_action, reward, next_state))
            state = next_state
            if done:
                return trajectory
        else:
            state = self.env.reset()

        for _ in range(self.max_steps):
            if self.env.is_terminal(state):
                break
            action = self._select_action(state)
            next_state, reward, done = self.env.step(state, action)
            trajectory.append((state, action, reward, next_state))
            state = next_state
            if done:
                break
        return trajectory

    def _update_from_trajectory(self, trajectory, episode_idx: int):
        """
        单循环处理轨迹：反向累计回报并更新 Q，随后刷新涉及状态的策略。

        参数:
            trajectory: [(state, action, reward, next_state), ...]
            episode_idx: 当前回合编号（用于间隔摘要日志）
        """
        G = 0.0
        visited_states = set()
        updated_keys = []
        # 反向遍历：一步计算G并即时更新Q
        for state, action, reward, _ in reversed(trajectory):
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

        # 摘要日志（轻量，不影响单循环处理）
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
                f"[第 {episode_idx} 回] len={len(trajectory)} Q更新={total_pairs} Pi更新={total_states} "
                f"Q统计(mean={q_mean:.3f}, min={q_min:.3f}, max={q_max:.3f})"
            )


    def evaluate_policy(self, max_steps=200):
        print("\n===== 策略验证：按最终策略从起点尝试到达终点 =====")
        state = self.env.reset()
        total_reward = 0.0
        path = [state]
        for _ in range(max_steps):
            if self.env.is_terminal(state):
                break
            action = self.policy.get(state)
            if action is None:
                action = self._greedy_action(state)
            next_state, reward, done = self.env.step(state, action)
            total_reward += reward
            path.append(next_state)
            state = next_state
            if done:
                break
        reached = self.env.is_terminal(state)
        print(f"是否到达终点：{'是' if reached else '否'}；步数：{len(path)-1}；累计奖励：{total_reward:.2f}")
        return reached, path, total_reward

    def _print_final_results(self):
        valid = set(self.env.states)
        self.policy = {s: a for s, a in self.policy.items() if s in valid}
        print("\n最终 Q(s,a)：")
        keys = sorted(self.Q.keys())
        for (s, a) in keys:
            print(f"  s={s} a={a} -> Q={self.Q[(s,a)]:.4f}")
        print("\n最终 Pi(s)：")
        for s in sorted(self.env.states):
            print(f"  s={s} -> a={self.policy.get(s, self._greedy_action(s))}")
        print("\n最佳策略棋盘（S=起点，G=终点，X=障碍，箭头=动作）：")
        print(self.env.render_policy(self.policy))
