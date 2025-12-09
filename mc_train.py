from typing import Optional
from mc_agent import MCAgent

def train_to_coverage(agent: MCAgent, max_episodes: Optional[int] = None, check_every: int = 20):
    """训练过程：按最大回合数进行训练（不再计算覆盖统计）。"""
    max_episode_count = max_episodes or agent.episodes
    print(f"\n===== 训练过程（最多 {max_episode_count} 回）=====")
    for episode_idx in range(1, max_episode_count + 1):
        trajectory = agent._simulate_episode()
        agent._update_from_trajectory(trajectory, episode_idx)
    print("\n===== 训练结束：最终的 Q、Pi 和最优策略 =====")
    # 训练结束后，为所有状态生成贪心策略以确保打印完整棋盘
    for state in agent.env.states:
        agent.policy[state] = agent._greedy_action(state)
    agent._print_final_results()
