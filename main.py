#!/usr/bin/env python3
"""
强化学习学习项目的主入口点。

通过此文件可以调用项目中不同环境（如 maze）下的强化学习算法。
使用示例:
    uv run main.py maze mc
    uv run main.py maze dqn
    # 如果作为唯一的顶层 CLI:
    uv run main.py mc
"""

from maze.main import cli as maze_cli

if __name__ == "__main__":
    # 直接调用 maze 的 cli，因为目前只有 maze 环境
    maze_cli()
