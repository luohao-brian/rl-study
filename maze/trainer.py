"""训练器模块

本模块定义了统一的 Trainer 类，封装了强化学习的经典训练循环。
不依赖于具体的算法（MC 或 DQN），只需传入实现了 BaseAgent 接口的对象即可。
"""

from typing import Tuple
from maze.base import BaseAgent


class Trainer:
    """统一训练器
    
    负责管理环境与智能体的交互过程，提供：
    - train(): 执行训练循环
    - evaluate(): 评估当前策略
    - print_summary(): 打印最终策略棋盘
    """

    def __init__(self, env, agent: BaseAgent):
        self.env = env
        self.agent = agent

    def train(self, episodes: int, max_steps: int = 200, log_interval: int = 20):
        """执行训练循环
        
        参数:
            episodes: 训练总回合数
            max_steps: 每回合最大步数
            log_interval: 日志打印间隔
        """
        print(f"\n===== 开始训练过程（最多 {episodes} 回）=====")
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            ep_return = 0.0
            
            for _ in range(max_steps):
                if self.env.is_terminal(state):
                    break
                    
                # 1. 智能体根据当前状态选择动作（训练模式，开启探索）
                # 这里就是强化学习里 Agent 对 Environment 做出的 Action
                action = self.agent.select_action(state, is_training=True)
                
                # 2. 环境执行动作，返回反馈
                # Environment 响应 Action，给出新的状态和这步的得分 Reward
                next_state, reward, done = self.env.step(state, action)
                
                # 3. 智能体接收单步反馈
                # 智能体把这步的经验收下（如果是 MC，就记在小本本上；如果是 DQN，就扔进回放池并当场学习）
                self.agent.step(state, action, reward, next_state, done)
                
                state = next_state
                ep_return += reward
                if done:
                    break
                    
            # 4. 回合结束，智能体进行回合级别的结算
            # (如 MC 的轨迹结算、计算 G 值，或者 DQN 的衰减 epsilon、拷贝目标网络等)
            self.agent.end_episode(ep)
            
            # (可选) 针对没有自己打印日志的 Agent（例如 DQN），在这里补上简要日志
            # MC Agent在end_episode里有专门的打印逻辑，DQN我们也可以将它的日志移到此处或保留在agent内
            # 为了兼容性，这里对DQN做个简单的回显
            if hasattr(self.agent, "epsilon") and not hasattr(self.agent, "print_summary"):
                if ep % log_interval == 0:
                    print(f"[第 {ep} 回] return={ep_return:.2f}  epsilon={self.agent.epsilon:.3f}")
                    
        print("===== 训练结束 =====")

    def evaluate(self, max_steps: int = 200) -> Tuple[bool, int, float]:
        """策略验证：按当前学到的策略从起点尝试到达终点
        
        返回:
            (是否到达终点, 步数, 累计奖励)
        """
        print("\n===== 策略验证：按最终策略从起点尝试到达终点 =====")
        state = self.env.reset()
        total_reward = 0.0
        steps = 0
        path = [state]
        
        for _ in range(max_steps):
            if self.env.is_terminal(state):
                break
                
            # 测试模式，关闭探索，纯贪心
            action = self.agent.select_action(state, is_training=False)
            
            next_state, reward, done = self.env.step(state, action)
            total_reward += reward
            path.append(next_state)
            state = next_state
            steps += 1
            if done:
                break
                
        reached = self.env.is_terminal(state)
        print(f"是否到达终点：{'是' if reached else '否'}；步数：{steps}；累计奖励：{total_reward:.2f}")
        return reached, steps, total_reward

    def print_summary(self):
        """打印最终策略棋盘与相关信息"""
        print("\n最终策略棋盘（S=起点，G=终点，X=障碍，箭头=动作）：")
        policy = {}
        for state in self.env.states:
            policy[state] = self.agent.select_action(state, is_training=False)
        print(self.env.render_policy(policy))
