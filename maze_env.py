"""迷宫环境模块

本模块定义了强化学习迷宫环境的核心组件：
- MazeEnv: 迷宫环境类，实现状态转换、奖励计算和策略可视化
- build_env(): 统一的环境构建接口（可选参数，默认生成含障碍的5x5迷宫）

设计思路：
1. 基于网格的离散环境，支持自定义尺寸和障碍
2. 提供完整的强化学习环境接口：reset(), step(), is_terminal()
3. 实现策略可视化功能，便于观察智能体学习结果
"""
import random


def build_env(width=5, height=5, start=(0, 0), goal=(4, 4), obstacles=None,
              step_reward=-1.0, goal_reward=10.0, invalid_penalty=-5.0):
    """
    构建自定义迷宫环境
    
    参数:
        width (int): 迷宫宽度
        height (int): 迷宫高度
        start (tuple): 起点位置 (row, col)
        goal (tuple): 终点位置 (row, col)
        obstacles (set): 障碍位置集合 {(row, col), ...}
        step_reward (float): 每步奖励
        goal_reward (float): 到达终点奖励
        invalid_penalty (float): 无效动作惩罚
    
    返回:
        MazeEnv: 配置后的迷宫环境实例（若未提供obstacles，将使用默认障碍布局）
    """
    # 默认障碍布局（与历史默认环境一致）
    default_obstacles = {
        (1, 1), (1, 3),
        (2, 2),
        (3, 1), (3, 3),
    }
    if obstacles is None:
        obstacles = default_obstacles
    return MazeEnv(
        width=width,
        height=height,
        start=start,
        goal=goal,
        obstacles=obstacles,
        step_reward=step_reward,
        goal_reward=goal_reward,
        invalid_penalty=invalid_penalty
    )


class MazeEnv:
    """迷宫环境类
    
    实现了一个基于网格的离散强化学习环境，智能体在网格中移动，
    目标是从起点到达终点，同时避开障碍物。
    """
    
    def __init__(self, width, height, start, goal, obstacles=None,
                 step_reward=-1.0, goal_reward=10.0, invalid_penalty=-5.0):
        """初始化迷宫环境
        
        参数:
            width (int): 迷宫的宽度（列数）
            height (int): 迷宫的高度（行数）
            start (tuple): 起点位置 (row, col)
            goal (tuple): 终点位置 (row, col)
            obstacles (set): 障碍物位置集合 {(row, col), ...}
            step_reward (float): 每走一步的奖励
            goal_reward (float): 到达终点的奖励
            invalid_penalty (float): 执行无效动作（撞墙）的惩罚
        """
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = set(obstacles or [])
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.invalid_penalty = invalid_penalty
        self.actions = ['U', 'D', 'L', 'R']
        self._delta = {
            'U': (-1, 0),
            'D': (1, 0),
            'L': (0, -1),
            'R': (0, 1),
        }
        # 生成所有有效状态（排除障碍物）
        self.states = [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if (row, col) not in self.obstacles
        ]

    def reset(self):
        """重置环境到初始状态
        
        返回:
            初始状态（起点位置）
        """
        return self.start

    def is_terminal(self, state):
        """检查状态是否为终止状态（到达终点）
        
        参数:
            state: 当前状态 (row, col)
            
        返回:
            True 如果到达终点，否则返回 False
        """
        return state == self.goal

    def _valid_pos(self, row, col):
        """检查位置是否有效（在网格内且非障碍物）
        
        参数:
            row: 行坐标
            col: 列坐标
            
        返回:
            True 如果位置有效，否则返回 False
        """
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return False
        if (row, col) in self.obstacles:
            return False
        return True

    def step(self, state, action):
        """执行动作并返回新状态
        
        参数:
            state: 当前状态 (row, col)
            action: 执行的动作 ('U', 'D', 'L', 'R')
            
        返回:
            一个元组 (next_state, reward, done)，其中：
            - next_state: 执行动作后的状态
            - reward: 获得的奖励
            - done: 是否到达终止状态
        """
        # 获取动作对应的位移 (delta_row, delta_col)
        delta_row, delta_col = self._delta[action]
        # 计算新位置 (new_row, new_col)
        new_row, new_col = state[0] + delta_row, state[1] + delta_col
        
        # 检查新位置是否有效
        if not self._valid_pos(new_row, new_col):
            # 无效动作，保持当前状态并给予惩罚
            next_state = state
            reward = self.invalid_penalty
            done = False
            return next_state, reward, done
            
        next_state = (new_row, new_col)
        
        # 检查是否到达终点
        if next_state == self.goal:
            return next_state, self.goal_reward, True
            
        # 普通移动，给予步骤奖励
        return next_state, self.step_reward, False

    def render_policy(self, policy):
        """可视化策略（在迷宫上显示每个状态的最优动作）
        
        参数:
            policy: 策略字典，键为状态 (row, col)，值为动作 ('U', 'D', 'L', 'R')
            
        返回:
            字符串形式的策略可视化结果，包含：
            - ↑↓←→: 动作方向箭头
            - S: 起点
            - G: 终点
            - X: 障碍物
            - ·: 未定义策略的状态
        """
        lines = []
        # 动作到箭头的映射
        arrow = {'U': '↑', 'D': '↓', 'L': '←', 'R': '→'}
        
        # 逐行构建可视化
        for row in range(self.height):
            line = []
            for col in range(self.width):
                position = (row, col)
                
                if position in self.obstacles:
                    line.append('X')  # 障碍物
                elif position == self.start:
                    line.append('S')  # 起点
                elif position == self.goal:
                    line.append('G')  # 终点
                else:
                    action = policy.get(position)
                    # 未定义策略位置使用·（点）表示
                    line.append(arrow.get(action, '·'))
            lines.append(' '.join(line))
            
        return '\n'.join(lines)

    def render_maze(self):
        """渲染迷宫网格（不含策略箭头，仅显示结构）
        
        返回:
            字符串形式的迷宫布局：
            - S: 起点
            - G: 终点
            - X: 障碍物
            - ·: 空格（可通行）
        """
        lines = []
        for row in range(self.height):
            line = []
            for col in range(self.width):
                position = (row, col)
                if position in self.obstacles:
                    line.append('X')
                elif position == self.start:
                    line.append('S')
                elif position == self.goal:
                    line.append('G')
                else:
                    line.append('·')
            lines.append(' '.join(line))
        return '\n'.join(lines)
