"""
Comparison script for Q-Learning vs SARSA on Level 1 (Task 2).

Usage:
    python -m gridworld.compare --level 1
    python -m gridworld.compare --level 1 --episodes 1000
"""

import argparse
import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld.environment import GridWorldEnv
from gridworld.agents.q_learning import QLearningAgent
from gridworld.agents.sarsa import SARSAAgent
from gridworld.train import train, load_config, MODELS_DIR, LOGS_DIR


def trace_greedy_path(env, agent):
    """Trace the trajectory and positions taken by an agent under greedy policy."""
    agent.epsilon = 0.0
    state = env.reset()
    path = [env.agent_pos]
    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < 100:
        action = agent.choose_action(state)
        state, reward, done, info = env.step(action)
        path.append(env.agent_pos)
        total_reward += reward
        steps += 1

    return path, total_reward, steps, info


def main():
    parser = argparse.ArgumentParser(description="Compare Q-Learning vs SARSA on Hazard Level")
    parser.add_argument("--level", type=int, default=1, help="Level ID (default: 1)")
    parser.add_argument("--episodes", type=int, default=1000, help="Training episodes per agent")
    parser.add_argument("--window", type=int, default=50, help="Moving average smoothing window")
    args = parser.parse_args()

    config = load_config()
    monster_chance = config.get("monster", {}).get("move_chance", 0.4)
    tc = config["training"]

    print(f"=== Comparing Q-Learning vs SARSA on Level {args.level} ===")
    print(f"    Episodes: {args.episodes} | Alpha: {tc['alpha']} | Gamma: {tc['gamma']}")
    print()

    # 1. Train Q-Learning
    print("[-] Training Q-Learning...")
    env_q = GridWorldEnv(args.level, monster_move_chance=monster_chance)
    q_agent = QLearningAgent(
        alpha=tc["alpha"], gamma=tc["gamma"],
        epsilon_start=tc["epsilon_start"], epsilon_end=tc["epsilon_end"],
        num_episodes=args.episodes,
    )
    q_rewards, _ = train(env_q, q_agent, args.episodes)

    # 2. Train SARSA
    print("\n[-] Training SARSA...")
    env_s = GridWorldEnv(args.level, monster_move_chance=monster_chance)
    sarsa_agent = SARSAAgent(
        alpha=tc["alpha"], gamma=tc["gamma"],
        epsilon_start=tc["epsilon_start"], epsilon_end=tc["epsilon_end"],
        num_episodes=args.episodes,
    )
    sarsa_rewards, _ = train(env_s, sarsa_agent, args.episodes)

    # 3. Save Models
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    q_agent.save(os.path.join(MODELS_DIR, f"level{args.level}_qlearning.pkl"))
    sarsa_agent.save(os.path.join(MODELS_DIR, f"level{args.level}_sarsa.pkl"))

    # 4. Evaluate Greedy Paths
    q_path, q_r, q_steps, q_info = trace_greedy_path(env_q, q_agent)
    s_path, s_r, s_steps, s_info = trace_greedy_path(env_s, sarsa_agent)

    print("\n" + "=" * 60)
    print("=== EVALUATION & BEHAVIOR COMPARISON ===")
    print(f"Q-Learning Path Length : {q_steps} steps | Reward: {q_r:.1f} | Result: {'VICTORY' if q_info.get('victory') else 'DIED'}")
    print(f"Q-Learning Trajectory  : {q_path}")
    print("-" * 60)
    print(f"SARSA Path Length      : {s_steps} steps | Reward: {s_r:.1f} | Result: {'VICTORY' if s_info.get('victory') else 'DIED'}")
    print(f"SARSA Trajectory       : {s_path}")
    print("=" * 60)

    # 5. Generate Comparison Plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    w = args.window

    # Rolling averages
    if len(q_rewards) >= w:
        q_roll = np.convolve(q_rewards, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(q_rewards)), q_roll,
                color="#E53935", linewidth=2.2, label=f"Q-Learning ({w}-ep rolling avg)")

    if len(sarsa_rewards) >= w:
        s_roll = np.convolve(sarsa_rewards, np.ones(w) / w, mode="valid")
        ax.plot(range(w - 1, len(sarsa_rewards)), s_roll,
                color="#1E88E5", linewidth=2.2, label=f"SARSA ({w}-ep rolling avg)")

    ax.plot(q_rewards, alpha=0.12, color="#E53935", linestyle="--")
    ax.plot(sarsa_rewards, alpha=0.12, color="#1E88E5", linestyle="--")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(f"Level {args.level} (Fire Hazards): Q-Learning vs SARSA Learning Dynamics", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    comp_plot_path = os.path.join(LOGS_DIR, f"level{args.level}_comparison_qlearning_vs_sarsa.png")
    fig.savefig(comp_plot_path, dpi=150)
    plt.close(fig)
    print(f"\n[+] Comparison plot saved -> {comp_plot_path}")


if __name__ == "__main__":
    main()
