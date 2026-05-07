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
    epsilon: float = 0.2  # epsilon-greedy 策略的探索率，值越大随机探索概率越高，通常在 0 到 1 之间
    gamma: float = 0.9  # 折扣因子，决定未来奖励的权重，值越接近 1 越看重长期回报
    episodes: int = 2000  # 训练的总回合数
    max_steps: int = 200  # 每个回合允许的最大行动步数，防止智能体陷入死循环
    check_every: int = 20  # 评估策略或记录进度的频率（每隔多少个回合执行一次）
    seed: int = 42  # 随机种子，用于保证实验结果的可复现性
    exploring_starts: bool = True  # 是否开启探索性起点，保证所有状态-动作对都有可能作为初始状态被访问到
    # 日志配置：默认仅摘要，每隔 log_interval 打印一次
    log_interval: int = 20  # 打印训练日志的时间间隔（回合数）
    print_summary: bool = True  # 是否打印简要的统计摘要信息


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
        """根据当前的 Q 表，选出在指定状态下预期得分（Q值）最高的动作"""
        best_a = None
        # 初始化当前找到的最大 Q 值为负无穷大，作为比较的基准（打擂台算法的初始擂主）
        # 确保第一个动作的 Q 值无论多小，都能成功赋值给 best_q
        best_q = float('-inf')
        
        for a in self.env.actions:
            q = self.Q[(state, a)]  # 从知识库中查出这个动作的预期得分
            if q > best_q:          # 如果发现得分更高的动作
                best_q = q          # 更新最高分记录（局部临时变量，循环结束后销毁）
                best_a = a          # 记录下这个最高分对应的动作
                
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
        # 反向遍历，从最后一步倒推计算每一步的真实回报 G
        for state, action, reward, _ in reversed(self.current_trajectory):
            G = reward + self.gamma * G
            key = (state, action)
            
            # 第一步：基于经验更新 Q 值（Policy Evaluation 策略评估）
            self.returns_sum[key] += G    # 累计获得的总回报（分子）
            self.returns_count[key] += 1  # 累计经历该状态-动作对的次数（分母）
            # 计算实际跑出的平均成绩，更新知识库（Q 表）
            self.Q[key] = self.returns_sum[key] / self.returns_count[key]
            
            visited_states.add(state)
            updated_keys.append(key)

        # 第二步：基于最新的 Q 表刷新策略（Policy Improvement 策略改进）
        # 将刚才经历过的状态的“行动手册”更新为当前已知得分最高的最优动作
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
