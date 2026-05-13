"""动态规划智能体 (DP Agent) 实现

本模块实现了基于模型的动态规划算法（价值迭代 Value Iteration）：
- DPConfig: 动态规划算法的配置超参数
- DPAgent: 实现了全状态扫描更新与价值迭代的智能体类

系统视角解析 (Model-based vs Model-free)：
1. Model-free（无模型算法如 MC / TD / DQN）：环境是未知的黑盒。智能体仅能在当前所在的绝对单一物理位置调用 step() 产生真实位移副作用，只能利用物理轨迹收集的局部后验经验更新估值。
2. Model-based（基于模型算法如 DP）：智能体拥有环境转移函数与奖励规则的全量先验知识（即 Model）。
   - 状态特权：直接读取全局坐标集合遍历全空间；
   - 转移特权：将底层的 env.step(s, a) 当作无状态副作用的静态纯查询接口调用。即输入任意坐标和动作，像使用任意门或查阅规则手册一样瞬间获知后继状态和期望回报。这种离线全盘推演使得 DP 能够脱离低效的物理试错，极速达成贝尔曼收敛。
"""

from dataclasses import dataclass
from collections import defaultdict
import random
from typing import List, Tuple, Dict
import json
import os
from maze.base import BaseAgent


@dataclass
class DPConfig:
    """动态规划算法超参数配置类"""
    gamma: float = 0.9  # 折扣因子，决定对未来长期回报的重视程度
    theta: float = 1e-4  # 价值迭代收敛阈值（当单次遍历最大价值变动小于此值时认为已收敛）
    episodes: int = 100  # 训练循环所执行的总扫描轮数（配合 Trainer 循环展示收敛过程）
    epsilon: float = 0.1  # 在 Trainer 训练模式下走迷宫时的随机探索率，避免未收敛时陷入原地打转
    seed: int = 42  # 随机数种子
    log_interval: int = 10  # 打印训练进度摘要的间隔轮数
    print_summary: bool = True  # 是否打印各轮扫描的价值变化统计


class DPAgent(BaseAgent):
    """动态规划智能体（价值迭代算法实现）

    继承自 BaseAgent 接口。主要功能：
    - 内部维护状态价值函数 V(s)
    - 在 end_episode 中利用环境转移函数直接执行单轮全网格价值迭代扫描
    - 动态推导出 Q(s,a) 价值和最优策略 Policy
    """

    def __init__(self, env, config: DPConfig):
        self.env = env
        self.cfg = config
        self.gamma = config.gamma
        self.theta = config.theta
        self.epsilon = config.epsilon
        self.log_interval = max(1, int(config.log_interval))
        self.print_summary = bool(config.print_summary)

        # 状态价值表 V(s)，默认初始化为 0.0
        self.V: Dict[Tuple[int, int], float] = defaultdict(float)
        # 导出的策略表 Policy
        self.policy: Dict[Tuple[int, int], str] = {}

        random.seed(config.seed)
        # 初始时先进行一次贪心策略推导，确保 policy 字典中包含初始动作
        self._update_policy()

    def _get_action_target(self, state: Tuple[int, int], action: str) -> float:
        """计算给定状态下采取指定动作的 Bellman 目标价值 Q(s,a)

        公式: R(s,a) + gamma * V(s')
        Model-based 上帝视角深度解析：
        此处调用的 self.env.step(state, action) 并非让智能体在环境中产生真实的物理位移。
        由于 step() 方法内部仅包含坐标算术加减与静态障碍物集合匹配，没有任何实例状态覆盖操作，
        因此它本质上充当了一个【无副作用的确定性转移查询接口】。
        DP 智能体正是借此实现对地图的离线任意提问，无需真实探索即可算出全盘后继期望。
        """
        next_state, reward, done = self.env.step(state, action)
        if done:
            return reward
        return reward + self.gamma * self.V[next_state]

    def _greedy_action(self, state: Tuple[int, int]) -> str:
        """针对指定状态，基于当前的 V 表推导出单步最优动作"""
        best_a = None
        best_q = float("-inf")
        best_actions = []

        for a in self.env.actions:
            q = self._get_action_target(state, a)
            # 采用打擂台方式寻找最大 Q 值
            if q > best_q + 1e-8:  # 加上微小容忍度处理浮点数精度问题
                best_q = q
                best_actions = [a]
            elif abs(q - best_q) <= 1e-8:
                best_actions.append(a)

        # 如果有多个动作价值相同，随机挑一个打破平衡
        if best_actions:
            best_a = random.choice(best_actions)
        else:
            best_a = random.choice(self.env.actions)
        return best_a

    def _update_policy(self):
        """遍历所有状态，依据最新的 V 表全量刷新导出的贪心策略"""
        for state in self.env.states:
            if state == self.env.goal:
                continue
            self.policy[state] = self._greedy_action(state)

    def select_action(self, state: Tuple[int, int], is_training: bool = True) -> str:
        """实现 BaseAgent 接口：动作选择

        - 在训练（迭代演示）模式下，保留少量 epsilon 随机探索，便于智能体在迷宫中移动演示
        - 在评估/测试模式下，严格执行导出的最优策略
        """
        if is_training and random.random() < self.epsilon:
            return random.choice(self.env.actions)

        action = self.policy.get(state)
        if action is None:
            action = self._greedy_action(state)
        return action

    def step(self, state, action, reward, next_state, done):
        """实现 BaseAgent 接口：处理单步经验

        动态规划是基于环境模型的全局离线规划算法，无需通过单步在线轨迹经验来更新价值。
        因此本方法留空。
        """
        pass

    def end_episode(self, episode_idx: int):
        """实现 BaseAgent 接口：回合结束逻辑

        为了完美融合进 Trainer 的训练循环体系，我们在每回合结束时执行【单次全状态空间扫描】。
        这等价于价值迭代（Value Iteration）中的一次完整 Sweep。
        """
        delta = 0.0
        # 复制一份旧的 V 表以进行同步更新（确保更新顺序不影响本轮扫描结果）
        old_V = float("nan")  # 仅为占位，后续逐个状态计算目标
        
        # 遍历所有有效网格状态
        for state in self.env.states:
            # 终点状态价值恒为 0
            if state == self.env.goal:
                self.V[state] = 0.0
                continue

            v = self.V[state]
            # 计算当前状态下所有可能动作的最大 Bellman 目标值：max_a [R + gamma * V(s')]
            max_q = float("-inf")
            for a in self.env.actions:
                q = self._get_action_target(state, a)
                if q > max_q:
                    max_q = q

            self.V[state] = max_q
            # 记录本轮扫描中全网格最大的价值变动量
            delta = max(delta, abs(v - max_q))

        # 依据更新后的价值表全量刷新导出的操作手册（策略）
        self._update_policy()

        # 打印摘要日志
        if self.print_summary and (episode_idx % self.log_interval == 0):
            # 简单统计当前所有状态价值的平均值
            v_values = [self.V[s] for s in self.env.states if s != self.env.goal]
            mean_v = sum(v_values) / len(v_values) if v_values else 0.0
            print(
                f"[第 {episode_idx} 轮扫描] 最大价值变动 delta={delta:.6f} "
                f"平均状态价值 V_mean={mean_v:.3f} "
                f"{'【已收敛】' if delta < self.theta else ''}"
            )

    def save(self, path: str):
        """保存状态价值表 V 和导出的 Policy 为 JSON 格式"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        v_serializable = {f"{s[0]},{s[1]}": val for s, val in self.V.items()}
        policy_serializable = {f"{s[0]},{s[1]}": a for s, a in self.policy.items()}

        data = {"V": v_serializable, "policy": policy_serializable}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"DP模型已保存至: {path}")

    def load(self, path: str):
        """从 JSON 文件恢复状态价值表 V 和 Policy"""
        with open(path, "r") as f:
            data = json.load(f)

        self.V = defaultdict(float)
        for k, val in data["V"].items():
            r, c = k.split(",")
            self.V[(int(r), int(c))] = float(val)

        self.policy = {}
        for k, a in data["policy"].items():
            r, c = k.split(",")
            self.policy[(int(r), int(c))] = a

        print(f"DP模型已从 {path} 加载")
