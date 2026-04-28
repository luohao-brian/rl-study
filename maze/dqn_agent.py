"""DQN (深度Q网络) 实现，用于迷宫环境求解

本模块定义了DQN算法的核心组件：
- ReplayBuffer: 经验回放缓冲区，用于离策略学习
- QNetwork: 深度神经网络，将状态映射到动作价值
- DQNAgent: DQN智能体，实现了ε-贪心策略、目标网络和经验回放
- DQNConfig: DQN算法的配置参数类

设计思路：
1. 使用经验回放打破数据相关性，提高训练稳定性
2. 使用目标网络减少Q值估计的波动性
3. 使用ε-贪心策略平衡探索与利用
4. 使用神经网络近似Q值函数
"""

from dataclasses import dataclass
from typing import List, Tuple
import random
import collections
import math

import torch
import torch.nn as nn
import torch.optim as optim
import os

from maze.base import BaseAgent


class ReplayBuffer:
    """固定大小的经验回放缓冲区

    存储智能体与环境交互的经验轨迹 (state, action, reward, next_state, done)，并提供随机批量采样功能。
    经验回放可以打破数据的时间相关性，提高训练稳定性和样本利用率。
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, done: bool):
        """将一条经验轨迹存入缓冲区
        
        参数:
            state_idx: 状态的索引值 (int)
            action_idx: 动作的索引值 (int)
            reward: 获得的奖励值 (float)
            next_state_idx: 下一状态的索引值 (int)
            done: 是否到达终止状态 (bool)
        """
        self.buffer.append((state_idx, action_idx, reward, next_state_idx, done))

    def sample(self, batch_size: int):
        """从缓冲区中随机采样一批经验
        
        参数:
            batch_size: 采样的批次大小
            
        返回:
            一个元组包含五个张量：(状态索引, 动作索引, 奖励, 下一状态索引, 终止标志)
        """
        batch = random.sample(self.buffer, batch_size)
        # 解包批次数据，生成列表：states/actions/rewards/next_states/dones
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.long),
            torch.tensor(dones, dtype=torch.bool),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q网络实现：使用多层感知机(MLP)将独热编码状态映射到动作价值

    输入：状态的独热编码 (大小为 width*height)
    输出：四个动作的价值 (上、下、左、右)
    """

    def __init__(self, num_states: int, num_actions: int, hidden_size: int = 64):
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(num_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        """前向传播函数
        
        参数:
            state_indices: 状态索引张量，形状为 (B,)，每个值在 [0, num_states) 范围内
            
        处理过程:
            1. 将状态索引转换为独热编码向量
            2. 将独热向量输入MLP网络
            3. 输出每个动作的Q值
        """
        batch_size = state_indices.shape[0]
        one_hot = torch.zeros(batch_size, self.num_states, dtype=torch.float32, device=state_indices.device)
        one_hot.scatter_(1, state_indices.view(-1, 1), 1.0)
        return self.net(one_hot)


@dataclass
class DQNConfig:
    """DQN算法配置参数类
    
    参数说明:
        gamma: 折扣因子，用于计算未来奖励的现值
        epsilon_start: ε-贪婪策略的初始探索率
        epsilon_end: ε-贪婪策略的最终探索率
        epsilon_decay_episodes: ε值衰减的总回合数
        lr: 学习率
        batch_size: 每次训练的批次大小
        buffer_capacity: 经验回放缓冲区的最大容量
        target_update_interval: 目标网络的硬更新间隔（回合数）
        hidden_size: 神经网络隐藏层的大小
        episodes: 训练总回合数（训练循环使用）
        max_steps_per_episode: 每回合最大步数（训练循环使用）
        log_interval: 日志打印间隔（训练循环使用）
    """
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 300
    lr: float = 1e-3
    batch_size: int = 64
    buffer_capacity: int = 10000
    target_update_interval: int = 20  # 每N回合硬更新一次目标网络
    hidden_size: int = 64
    episodes: int = 600
    max_steps_per_episode: int = 200
    log_interval: int = 20


class DQNAgent(BaseAgent):
    """DQN智能体类，用于迷宫环境的强化学习
    
    实现 BaseAgent 接口。主要职责:
    - 创建和管理Q网络与目标网络
    - 实现ε-贪心策略的动作选择
    - 在经验回放缓冲区中存储和采样轨迹数据
    - 执行梯度更新优化Q网络
    - 提供状态坐标与索引之间的映射工具
    """

    def __init__(self, env, config: DQNConfig, seed: int = 42):
        self.env = env
        self.cfg = config
        self.num_states = env.width * env.height
        self.num_actions = len(env.actions)
        random.seed(seed)
        torch.manual_seed(seed)

        # Q-network and target network
        self.q_net = QNetwork(self.num_states, self.num_actions, hidden_size=self.cfg.hidden_size)
        self.target_net = QNetwork(self.num_states, self.num_actions, hidden_size=self.cfg.hidden_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # target net used only for value targets

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)

        # Experience replay buffer
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity)

        # Epsilon setup
        self.epsilon = self.cfg.epsilon_start

        # Pre-compute action index mapping for convenience
        self.action_to_idx = {a: i for i, a in enumerate(env.actions)}
        self.idx_to_action = {i: a for i, a in enumerate(env.actions)}

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """将二维坐标状态 (行, 列) 转换为一维索引
        
        参数:
            state: 二维坐标形式的状态 (row, col)
            
        返回:
            对应的一维索引值
        """
        r, c = state
        return r * self.env.width + c

    def select_action(self, state: Tuple[int, int], is_training: bool = True) -> str:
        """实现 BaseAgent 接口：使用ε-贪婪策略选择动作"""
        if is_training and random.random() < self.epsilon:
            return random.choice(self.env.actions)
        # Greedy based on current Q-network
        with torch.no_grad():
            # 将二维坐标状态转换为索引，并送入Q网络计算各动作Q值
            state_idx = torch.tensor([self.state_to_index(state)], dtype=torch.long)
            q_values = self.q_net(state_idx)
            # 选择Q值最大的动作索引
            action_idx = int(torch.argmax(q_values, dim=1).item())
            return self.idx_to_action[action_idx]

    def step(self, state, action, reward, next_state, done):
        """实现 BaseAgent 接口：将单步经验记入回放缓冲区，并执行更新"""
        self.buffer.push(
            self.state_to_index(state),
            self.action_to_idx[action],
            reward,
            self.state_to_index(next_state),
            done,
        )
        return self._update_network()

    def end_episode(self, episode_idx: int):
        """实现 BaseAgent 接口：处理 epsilon 衰减和目标网络更新"""
        self._decay_epsilon(episode_idx)
        if episode_idx % self.cfg.target_update_interval == 0:
            self._hard_update_target()

    def save(self, path: str):
        """保存 DQN 网络权重到指定路径"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.q_net.state_dict(), path)
        print(f"DQN模型已保存至: {path}")

    def load(self, path: str):
        """从指定路径加载 DQN 网络权重"""
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        self.target_net.eval()
        print(f"DQN模型已从 {path} 加载")

    def _update_network(self):
        """执行一次Q网络的梯度更新（如果缓冲区有足够样本）
        
        返回:
            本次更新的损失值，如果样本不足则返回None
        """
        if len(self.buffer) < self.cfg.batch_size:
            return None  # not enough samples yet

        # 从回放缓冲区采样一批经验数据
        # state_indices: 当前状态索引张量 (B,)
        # action_indices: 动作索引张量 (B,)
        # rewards: 奖励值张量 (B,)
        # next_state_indices: 下一状态索引张量 (B,)
        # done_flags: 终止标志张量 (B,)，True 表示到达终点
        state_indices, action_indices, rewards, next_state_indices, done_flags = self.buffer.sample(self.cfg.batch_size)
        
        # 计算当前状态下选择动作的Q值 Q(s,a)
        q_values = self.q_net(state_indices)
        # 从Q值张量中提取每个样本对应的动作Q值
        q_sa = q_values.gather(1, action_indices.view(-1, 1)).squeeze(1)

        # 计算目标Q值：
        # 对非终止状态：target = reward + gamma * max_a' Q_target(s', a')
        # 对终止状态：target = reward
        with torch.no_grad():
            next_q = self.target_net(next_state_indices)
            max_next_q, _ = torch.max(next_q, dim=1)
            target = rewards + (1.0 - done_flags.float()) * self.cfg.gamma * max_next_q

        loss = nn.MSELoss()(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _decay_epsilon(self, episode_idx: int):
        """线性衰减ε值
        
        参数:
            episode_idx: 当前训练回合数
            
        过程:
            在设定的回合数内，将ε从初始值线性衰减到最小值
        """
        t = min(episode_idx, self.cfg.epsilon_decay_episodes)
        frac = 1.0 - t / float(self.cfg.epsilon_decay_episodes)
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * max(0.0, frac)

    def _hard_update_target(self):
        """硬更新目标网络的权重
        
        过程:
            将当前Q网络的所有权重直接复制到目标网络
        """
        self.target_net.load_state_dict(self.q_net.state_dict())
