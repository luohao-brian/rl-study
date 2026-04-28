#!/usr/bin/env python3
"""
强化学习迷宫求解器主程序

这是一个用于求解5x5迷宫问题的强化学习框架，支持多种强化学习算法：
- dqn: DQN（深度Q网络）深度强化学习方法
- mc: 蒙特卡洛控制方法

设计思路：
1. 使用命令行接口(CLI)提供统一入口，支持多种算法
2. 模块化设计，将环境、算法、主程序分离
3. 统一的结果输出格式，便于比较不同算法效果
4. 提供可视化策略棋盘，直观展示学习结果

实现路径：
1. 初始化命令行接口(CLI)
2. 根据用户选择调用对应算法模块
3. 构建迷宫环境
4. 初始化算法代理
5. 执行训练过程
6. 评估学习到的策略
7. 可视化策略棋盘

使用示例：
  uv run main.py mc      # 使用蒙特卡洛方法求解
  uv run main.py dqn     # 使用DQN方法求解
"""

import click
from maze.maze_env import build_env


@click.group()
def cli():
    """强化学习迷宫求解器命令行接口
    
    作为所有强化学习方法的统一入口点，提供命令行参数解析和命令分发功能。
    """
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["train", "eval"]), default="train", help="运行模式：train(训练) 或 eval(评估)")
@click.option("--model-path", type=str, default="models/mc_model.json", help="MC模型文件的保存/加载路径")
def mc(mode, model_path):
    """使用蒙特卡洛控制方法求解迷宫"""
    from maze.mc_agent import MCAgent, MCConfig
    from maze.trainer import Trainer

    print(f"=== 蒙特卡洛控制方法求解迷宫 (模式: {mode}) ===")

    env = build_env(
        width=5,
        height=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)},
        step_reward=-1.0,
        goal_reward=10.0,
        invalid_penalty=-5.0,
    )
    print("环境初始化完成：5x5 迷宫（含障碍），起点(0,0)，终点(4,4)。")
    print("迷宫布局（S=起点，G=终点，X=障碍，·=空）：")
    print(env.render_maze())

    # 使用 Config 默认值进行初始化
    cfg = MCConfig()
    print(f"参数: episodes={cfg.episodes}, check_every={cfg.check_every}, epsilon={cfg.epsilon}, gamma={cfg.gamma}")
    
    agent = MCAgent(env, cfg)
    trainer = Trainer(env, agent)

    if mode == "train":
        trainer.train(episodes=cfg.episodes, max_steps=cfg.max_steps, log_interval=cfg.log_interval)
        agent.save(model_path)
        
        # 训练结束后顺便评估一下
        trainer.evaluate(max_steps=100)
        trainer.print_summary()
    elif mode == "eval":
        agent.load(model_path)
        trainer.evaluate(max_steps=100)
        trainer.print_summary()

@cli.command()
@click.option("--mode", type=click.Choice(["train", "eval"]), default="train", help="运行模式：train(训练) 或 eval(评估)")
@click.option("--model-path", type=str, default="models/dqn_model.pth", help="DQN模型文件的保存/加载路径")
def dqn(mode, model_path):
    """使用DQN深度强化学习方法求解迷宫"""
    from maze.dqn_agent import DQNAgent, DQNConfig
    from maze.trainer import Trainer

    print(f"=== DQN深度强化学习方法求解迷宫 (模式: {mode}) ===")

    env = build_env(
        width=5,
        height=5,
        start=(0, 0),
        goal=(4, 4),
        obstacles={(1, 1), (1, 3), (2, 2), (3, 1), (3, 3)},
        step_reward=-1.0,
        goal_reward=10.0,
        invalid_penalty=-5.0,
    )
    print("环境初始化完成：5x5 迷宫（含障碍），起点(0,0)，终点(4,4)。")
    print("迷宫布局（S=起点，G=终点，X=障碍，·=空）：")
    print(env.render_maze())

    # 使用 Config 默认值进行初始化
    cfg = DQNConfig()
    print(f"参数: episodes={cfg.episodes}, log_interval={cfg.log_interval}, batch_size={cfg.batch_size}, lr={cfg.lr}")
    
    agent = DQNAgent(env, cfg, seed=42)
    trainer = Trainer(env, agent)

    if mode == "train":
        trainer.train(episodes=cfg.episodes, max_steps=cfg.max_steps_per_episode, log_interval=cfg.log_interval)
        agent.save(model_path)
        
        # 训练结束后顺便评估一下
        trainer.evaluate(max_steps=200)
        trainer.print_summary()
    elif mode == "eval":
        agent.load(model_path)
        trainer.evaluate(max_steps=200)
        trainer.print_summary()


if __name__ == "__main__":
    cli()
