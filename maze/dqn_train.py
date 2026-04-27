"""
Training routines for DQNAgent on the MazeEnv.

This module provides:
- train(agent): standard DQN training loop
- evaluate_policy(env, agent): greedy evaluation
- print_final_policy(env, agent): render final greedy policy
"""

import torch

from maze.maze_env import build_env
from maze.dqn_agent import DQNAgent, DQNConfig


# 统一使用 maze_env 提供的环境构建接口


def evaluate_policy(env, agent, max_steps=200):
    """贪心评估：从起点开始，始终选择当前Q网络下的最佳动作。

    返回 (是否到达终点, 步数, 累计奖励)
    """
    state = env.reset()
    total_reward = 0.0
    steps = 0
    for _ in range(max_steps):
        if env.is_terminal(state):
            break
        # 贪心选择动作：选择当前Q网络下最大Q值对应的动作
        with torch.no_grad():
            state_idx = torch.tensor([agent.state_to_index(state)], dtype=torch.long)
            q_values = agent.q_net(state_idx)
            action_idx = int(torch.argmax(q_values, dim=1).item())
            action = agent.idx_to_action[action_idx]
        next_state, reward, done_flag = env.step(state, action)
        total_reward += reward
        state = next_state
        steps += 1
        if done_flag:
            break
    return env.is_terminal(state), steps, total_reward

def train(agent: DQNAgent):
    """标准 DQN 训练循环：基于 agent.cfg 执行若干回合训练。"""
    env = agent.env
    episodes = agent.cfg.episodes
    max_steps_per_episode = agent.cfg.max_steps_per_episode
    log_interval = agent.cfg.log_interval
    print("DQN 训练开始：episodes=%d, max_steps=%d" % (episodes, max_steps_per_episode))
    for ep in range(1, episodes + 1):
        state = env.reset()              # 本回合初始状态
        ep_return = 0.0                  # 本回合累计奖励
        ep_losses = []                   # 本回合累计的损失记录
        for _ in range(max_steps_per_episode):
            if env.is_terminal(state):
                break
            action, _mode = agent.select_action(state)      # ε-贪婪选择动作
            next_state, reward, done_flag = env.step(state, action)
            agent.push_transition(state, action, reward, next_state, done_flag)
            loss = agent.update()                            # 执行一次Q网络更新
            if loss is not None:
                ep_losses.append(loss)
            state = next_state
            ep_return += reward
            if done_flag:
                break

        # Epsilon decay per episode
        agent.decay_epsilon(ep)

        # Hard update target network
        if ep % agent.cfg.target_update_interval == 0:
            agent.hard_update_target()

        if ep % log_interval == 0:
            avg_loss = sum(ep_losses) / len(ep_losses) if len(ep_losses) > 0 else float('nan')
            print(
                f"[第 {ep} 回] return={ep_return:.2f}  avg_loss={avg_loss:.6f}  epsilon={agent.epsilon:.3f}  buffer={len(agent.buffer)}"
            )


def print_final_policy(env, agent):
    """打印最终贪心策略棋盘。"""
    policy = {}
    with torch.no_grad():
        for state in env.states:
            state_idx = torch.tensor([agent.state_to_index(state)], dtype=torch.long)
            q_values = agent.q_net(state_idx)
            action_idx = int(torch.argmax(q_values, dim=1).item())
            policy[state] = agent.idx_to_action[action_idx]
    print("\n最终策略棋盘（S=起点，G=终点，X=障碍，箭头=动作）：")
    print(env.render_policy(policy))


# 独立脚本入口与冗余函数已移除，作为 main 的训练模块使用
