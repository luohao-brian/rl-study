"""时间差分智能体 (TD Agent) 实现

本模块集成了强化学习中两大经典的时间差分（Temporal Difference）无模型算法：
- Q-Learning: 异策略（Off-policy）时间差分控制算法
- Sarsa: 同策略（On-policy）时间差分控制算法

包含组件：
- TDConfig: 算法的超参数与切换配置
- TDAgent: 实现了单步自举更新、动作缓存与策略导出的智能体类

核心思想：
TD 算法结合了蒙特卡洛（MC）的采样思想和动态规划（DP）的自举（Bootstrapping）思想。
它无需等待整个完整回合结束，而是每走一步就利用下一状态的现有估计 Q(S', A') 或 max Q(S', a)
当场进行自举更新，具有方差小、可在线实时学习的巨大优势。
"""

from dataclasses import dataclass
from collections import defaultdict
import random
from typing import List, Tuple, Dict, Set
import json
import os
from maze.base import BaseAgent


@dataclass
class TDConfig:
    """时间差分算法配置参数类"""
    algorithm: str = "q_learning"  # 核心算法选择：支持 "q_learning" 或 "sarsa"
    alpha: float = 0.1  # 学习率，决定新知识覆盖旧估计的权重
    gamma: float = 0.9  # 折扣因子，权衡即时奖励与未来长期回报
    epsilon: float = 0.2  # epsilon-greedy 策略的随机探索概率
    episodes: int = 2000  # 训练循环总回合数
    max_steps: int = 200  # 每回合允许的最大步数
    seed: int = 42  # 随机数生成器种子
    log_interval: int = 20  # 日志打印的回合间隔
    print_summary: bool = True  # 是否打印训练摘要日志


class TDAgent(BaseAgent):
    """时间差分智能体（支持 Q-Learning 与 Sarsa）

    继承自 BaseAgent 接口。职责包括：
    - 维护基于字典的 Q-Table 与贪心策略 policy
    - 针对 Sarsa 实现动作预采样缓存，确保同策略 Bootstrap 目标与实际执行动作严丝合缝
    - 在 step() 中实时执行单步自举更新
    """

    def __init__(self, env, config: TDConfig):
        self.env = env
        self.cfg = config
        self.algorithm = config.algorithm.lower()
        if self.algorithm not in ["q_learning", "sarsa"]:
            raise ValueError(f"不支持的 TD 算法类型: {self.algorithm}，仅支持 q_learning 或 sarsa")

        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.log_interval = max(1, int(config.log_interval))
        self.print_summary = bool(config.print_summary)

        # 核心数据结构
        self.Q: Dict[Tuple[Tuple[int, int], str], float] = defaultdict(float)
        self.policy: Dict[Tuple[int, int], str] = {}

        # Sarsa 算法专用：用于完美对接外部 Trainer 循环的动作缓存
        self._cached_next_action: str = None

        # 用于收集单回合级别的统计信息（供 end_episode 打印精美摘要）
        self._ep_visited_states: Set[Tuple[int, int]] = set()
        self._ep_updated_keys: List[Tuple[Tuple[int, int], str]] = []
        self._ep_steps: int = 0

        random.seed(config.seed)

    def _greedy_action(self, state: Tuple[int, int]) -> str:
        """从当前 Q 表中查出指定状态下价值最高的动作（打擂台算法）"""
        best_a = None
        best_q = float("-inf")
        best_actions = []

        for a in self.env.actions:
            q = self.Q[(state, a)]
            if q > best_q + 1e-8:
                best_q = q
                best_actions = [a]
            elif abs(q - best_q) <= 1e-8:
                best_actions.append(a)

        if best_actions:
            best_a = random.choice(best_actions)
        else:
            best_a = random.choice(self.env.actions)
        return best_a

    def select_action(self, state: Tuple[int, int], is_training: bool = True) -> str:
        """实现 BaseAgent 接口：选择动作

        采用标准的 epsilon-greedy 策略。
        对于 Sarsa 算法，如果在上一步 step() 中已经提前预采样了下一动作作为自举目标，
        则在此处直接消耗并返回该预采样动作，保证理论上的同策略执行一致性。
        """
        # 如果有同策略 Sarsa 预留的缓存动作，优先消耗缓存
        if is_training and self._cached_next_action is None and self.algorithm == "sarsa":
            # 这种情况通常发生在回合的第一步，正常往下走选取逻辑即可
            pass
        elif is_training and self._cached_next_action is not None:
            action = self._cached_next_action
            self._cached_next_action = None
            return action

        # 标准的 epsilon-greedy 选择逻辑
        if is_training and random.random() < self.epsilon:
            return random.choice(self.env.actions)

        # 查阅现有确定性策略或当场推导贪心动作
        action = self.policy.get(state)
        if action is None:
            action = self._greedy_action(state)
        return action

    def step(self, state, action, reward, next_state, done):
        """实现 BaseAgent 接口：处理单步经验并当场执行 TD 更新

        核心自举公式：
        Q(S, A) <- Q(S, A) + alpha * [ TD_Target - Q(S, A) ]
        """
        self._ep_steps += 1
        self._ep_visited_states.add(state)
        key = (state, action)
        self._ep_updated_keys.append(key)

        # 计算 TD 目标值 (TD Target)
        if done:
            # 达到终止状态，无未来期望回报
            td_target = reward
        else:
            if self.algorithm == "q_learning":
                # Q-Learning (异策略 Off-policy)：直接假定下一步采取最优动作 max_a Q(S', a)
                max_next_q = max(self.Q[(next_state, a)] for a in self.env.actions)
                td_target = reward + self.gamma * max_next_q
            else:
                # Sarsa (同策略 On-policy)：依据当前 epsilon-greedy 策略预先抽取真实的下一动作 A'
                # 这样计算出的 Q(S', A') 才是同策略下的真实期望目标
                next_action = self.select_action(next_state, is_training=True)
                # 存入缓存，供紧接着的下一轮循环实际执行
                self._cached_next_action = next_action
                td_target = reward + self.gamma * self.Q[(next_state, next_action)]

        # 单步更新 Q 表
        current_q = self.Q[key]
        self.Q[key] = current_q + self.alpha * (td_target - current_q)

        # 实时同步刷新当前状态下的贪心策略手册
        self.policy[state] = self._greedy_action(state)

    def end_episode(self, episode_idx: int):
        """实现 BaseAgent 接口：回合结束逻辑

        清理残余缓存并输出精美的单回合级训练摘要统计。
        """
        # 清空 Sarsa 动作缓存，避免跨回合干扰
        self._cached_next_action = None

        # 打印精美摘要日志（与 MC 模块风格完美契合）
        if self.print_summary and (episode_idx % self.log_interval == 0):
            total_pairs = len(self._ep_updated_keys)
            total_states = len(self._ep_visited_states)
            q_values = [self.Q[k] for k in self._ep_updated_keys]
            if q_values:
                q_mean = sum(q_values) / len(q_values)
                q_min = min(q_values)
                q_max = max(q_values)
            else:
                q_mean = q_min = q_max = float("nan")

            alg_name = "Q-Learning" if self.algorithm == "q_learning" else "Sarsa"
            print(
                f"[第 {episode_idx} 回 ({alg_name})] 步数={self._ep_steps} Q更新={total_pairs} Pi覆盖={total_states} "
                f"Q统计(mean={q_mean:.3f}, min={q_min:.3f}, max={q_max:.3f})"
            )

        # 清空回合级统计容器
        self._ep_visited_states.clear()
        self._ep_updated_keys.clear()
        self._ep_steps = 0

    def save(self, path: str):
        """将 Q-Table 和 Policy 序列化保存为 JSON 格式"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        q_serializable = {f"{s[0]},{s[1]},{a}": v for (s, a), v in self.Q.items()}
        policy_serializable = {f"{s[0]},{s[1]}": a for s, a in self.policy.items()}

        data = {
            "algorithm": self.algorithm,
            "Q": q_serializable,
            "policy": policy_serializable,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"TD模型 ({self.algorithm}) 已保存至: {path}")

    def load(self, path: str):
        """从 JSON 文件反序列化恢复 Q-Table 和 Policy"""
        with open(path, "r") as f:
            data = json.load(f)

        loaded_alg = data.get("algorithm", self.algorithm)
        print(f"检测到存档算法类型为: {loaded_alg}")

        self.Q = defaultdict(float)
        for k, v in data["Q"].items():
            r, c, a = k.split(",")
            self.Q[((int(r), int(c)), a)] = float(v)

        self.policy = {}
        for k, v in data["policy"].items():
            r, c = k.split(",")
            self.policy[(int(r), int(c))] = v

        print(f"TD模型已从 {path} 成功加载")
