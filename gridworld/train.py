"""
Training script for gridworld agents.

Usage:
    python -m gridworld.train --level 0 --agent qlearning
    python -m gridworld.train --level 0 --agent qlearning --episodes 1000
    python -m gridworld.train --level 0 --agent qlearning --render
"""

import argparse
import json
import os
import sys
import numpy as np

# Use non-interactive backend for headless plot generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gridworld.environment import GridWorldEnv
from gridworld.agents.q_learning import QLearningAgent
from gridworld.renderer import GridWorldRenderer


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
MODELS_DIR = os.path.join(ROOT_DIR, "models", "gridworld")
LOGS_DIR = os.path.join(ROOT_DIR, "logs", "gridworld")


def load_config():
    """Load configuration from config.json."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_agent(agent_type, config, num_episodes, use_intrinsic=False):
    """Instantiate agent according to configuration."""
    tc = config["training"]
    intrinsic_strength = config.get("intrinsic", {}).get("reward_strength", 1.0)

    if agent_type == "qlearning":
        return QLearningAgent(
            alpha=tc["alpha"],
            gamma=tc["gamma"],
            epsilon_start=tc["epsilon_start"],
            epsilon_end=tc["epsilon_end"],
            num_episodes=num_episodes,
            use_intrinsic=use_intrinsic,
            intrinsic_strength=intrinsic_strength,
        )
    elif agent_type == "sarsa":
        from gridworld.agents.sarsa import SARSAAgent
        return SARSAAgent(
            alpha=tc["alpha"],
            gamma=tc["gamma"],
            epsilon_start=tc["epsilon_start"],
            epsilon_end=tc["epsilon_end"],
            num_episodes=num_episodes,
            use_intrinsic=use_intrinsic,
            intrinsic_strength=intrinsic_strength,
        )
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def train(env, agent, num_episodes, renderer=None, max_steps=500):
    """Run training loop and record episode rewards."""
    episode_rewards = []
    episode_steps = []

    for ep in range(num_episodes):
        state = env.reset()
        agent.reset_episode_visits()
        agent.record_visit(state)

        total_reward = 0.0
        step_count = 0

        action = agent.choose_action(state)

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)
            agent.record_visit(next_state)
            total_reward += reward
            step_count += 1

            if hasattr(agent, "__class__") and agent.__class__.__name__ == "SARSAAgent":
                next_action = agent.choose_action(next_state) if not done else None
                agent.update(state, action, reward, next_state, done, next_action=next_action)
                action = next_action
            else:
                agent.update(state, action, reward, next_state, done)
                action = agent.choose_action(next_state) if not done else None

            if renderer is not None:
                renderer.render(episode=ep + 1, step=step_count, total_reward=total_reward)

            state = next_state
            if done:
                break

        episode_rewards.append(total_reward)
        episode_steps.append(step_count)
        agent.decay_epsilon()

        # Logging intervals
        if (ep + 1) % max(1, num_episodes // 20) == 0 or ep == 0 or (ep + 1) == num_episodes:
            avg_rew = np.mean(episode_rewards[-min(50, len(episode_rewards)):])
            print(f"  Episode {ep + 1:>5}/{num_episodes} | "
                  f"Reward: {total_reward:>5.1f} | "
                  f"Avg50: {avg_rew:>5.2f} | "
                  f"Steps: {step_count:>3} | "
                  f"Eps: {agent.epsilon:.3f}")

    return episode_rewards, episode_steps


def save_training_plot(rewards, path, title="Training Curve", window=50):
    """Generate and save training reward curve with moving average."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards, alpha=0.25, color="#1976D2", label="Episode Reward")

    if len(rewards) >= window:
        rolling_mean = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), rolling_mean,
                color="#0D47A1", linewidth=2, label=f"{window}-Episode Moving Avg")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train Gridworld RL Agent")
    parser.add_argument("--level", type=int, default=0, help="Level ID (0-6)")
    parser.add_argument("--agent", type=str, default="qlearning",
                        choices=["qlearning", "sarsa"], help="Agent type")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Override number of training episodes")
    parser.add_argument("--intrinsic", action="store_true",
                        help="Enable intrinsic curiosity reward")
    parser.add_argument("--render", action="store_true",
                        help="Render training visually (slower)")
    args = parser.parse_args()

    config = load_config()
    num_episodes = args.episodes or config["training"]["episodes"]
    monster_chance = config.get("monster", {}).get("move_chance", 0.4)

    suffix = "_intrinsic" if args.intrinsic else ""
    print(f"=== Training {args.agent.upper()} on Level {args.level}{suffix} ===")
    print(f"    Episodes: {num_episodes} | Alpha: {config['training']['alpha']} | Gamma: {config['training']['gamma']}")
    print(f"    Epsilon Decay: {config['training']['epsilon_start']} -> {config['training']['epsilon_end']}")

    env = GridWorldEnv(args.level, monster_move_chance=monster_chance)
    agent = make_agent(args.agent, config, num_episodes, use_intrinsic=args.intrinsic)

    renderer = None
    if args.render:
        renderer = GridWorldRenderer(
            env,
            cell_size=config["rendering"]["cell_size"],
            fps=config["rendering"]["fps"],
            title=f"Training Level {args.level} ({args.agent})",
        )

    rewards, _ = train(env, agent, num_episodes, renderer=renderer)

    if renderer:
        renderer.close()

    # Ensure output directories exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Save trained model
    model_filename = f"level{args.level}_{args.agent}{suffix}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)
    agent.save(model_path)
    print(f"  [+] Model saved -> {model_path}")

    # Save plot
    plot_filename = f"level{args.level}_{args.agent}{suffix}.png"
    plot_path = os.path.join(LOGS_DIR, plot_filename)
    plot_title = f"Level {args.level} - {args.agent.upper()}{' (with Intrinsic Reward)' if args.intrinsic else ''}"
    save_training_plot(rewards, plot_path, title=plot_title)
    print(f"  [+] Training plot saved -> {plot_path}")

    # Save rewards CSV
    csv_filename = f"level{args.level}_{args.agent}{suffix}_rewards.csv"
    csv_path = os.path.join(LOGS_DIR, csv_filename)
    np.savetxt(csv_path, rewards, delimiter=",", header="episode_reward", comments="")
    print(f"  [+] Rewards CSV saved -> {csv_path}")

    print("=== Training Complete ===")


if __name__ == "__main__":
    main()
