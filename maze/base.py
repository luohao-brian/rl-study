from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """所有强化学习智能体的基类接口
    
    规定了智能体必须实现的核心方法，以便 Trainer 可以使用统一的训练循环。
    """
    
    @abstractmethod
    def select_action(self, state, is_training: bool = True):
        """选择动作
        
        参数:
            state: 当前状态
            is_training: 是否处于训练模式（若是，则允许探索，例如 ε-贪婪；若否，则始终贪婪）
        返回:
            被选中的动作
        """
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        """处理环境的反馈（单步经验）
        
        参数:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否终止
        """
        pass

    @abstractmethod
    def end_episode(self, episode_idx: int):
        """处理回合结束时的逻辑
        
        参数:
            episode_idx: 当前回合编号
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """序列化并保存模型到指定路径"""
        pass

    @abstractmethod
    def load(self, path: str):
        """从指定路径加载反序列化的模型"""
        pass
