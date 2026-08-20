"""
SARSA Agent for the Gridworld environment.

Task 2: Basic SARSA for Level 1
Update rule (on-policy):
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * Q(s', a') - Q(s, a)]

Key Difference from Q-Learning:
    - Q-Learning is off-policy: it updates using max_a' Q(s', a') assuming optimal future play.
    - SARSA is on-policy: it updates using Q(s', a') where a' is the action ACTUALLY selected
      by the epsilon-greedy policy. Because exploratory mistakes into hazards (fire) are
      accounted for during training, SARSA learns a safer, more conservative route away from danger.
"""

import pickle
from collections import defaultdict
import numpy as np


# Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
NUM_ACTIONS = 4


class SARSAAgent:
    """Tabular SARSA agent with on-policy temporal difference updates."""

    def __init__(self, alpha=0.1, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, num_episodes=1000,
                 use_intrinsic=False, intrinsic_strength=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.num_episodes = num_episodes

        # Linear decay step per episode
        self.epsilon_decay = (epsilon_start - epsilon_end) / max(num_episodes - 1, 1)

        # Q-table: state tuple -> numpy array of shape (NUM_ACTIONS,)
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS))

        # Intrinsic reward parameters (for Task 5)
        self.use_intrinsic = use_intrinsic
        self.intrinsic_strength = intrinsic_strength
        self.visit_counts = {}

    # ------------------------------------------------------------------
    # Exploration & Action Selection
    # ------------------------------------------------------------------

    def choose_action(self, state):
        """Epsilon-greedy action selection with random tie-breaking."""
        if np.random.random() < self.epsilon:
            return np.random.randint(NUM_ACTIONS)

        q_values = self.q_table[state]
        max_q = np.max(q_values)
        # Random tie-breaking among all actions sharing the maximal Q-value
        best_actions = np.where(np.isclose(q_values, max_q))[0]
        return int(np.random.choice(best_actions))

    # ------------------------------------------------------------------
    # On-Policy Q-Value Update
    # ------------------------------------------------------------------

    def update(self, state, action, reward, next_state, done, next_action=None):
        """On-policy SARSA update rule.

        Parameters
        ----------
        state : tuple
            Current state.
        action : int
            Action taken in state.
        reward : float
            Reward received.
        next_state : tuple
            Next state transitioned into.
        done : bool
            Whether episode terminated.
        next_action : int or None
            The actual next action chosen in next_state (required if not done).
        """
        total_reward = reward

        # Add intrinsic curiosity bonus if enabled (Task 5)
        if self.use_intrinsic:
            n = self.visit_counts.get(state, 0)
            intrinsic = self.intrinsic_strength / np.sqrt(n + 1)
            total_reward += intrinsic

        if done:
            td_target = total_reward
        else:
            assert next_action is not None, "SARSA requires next_action when not done"
            td_target = total_reward + self.gamma * self.q_table[next_state][next_action]

        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Intrinsic Reward & Visit Tracking Helpers
    # ------------------------------------------------------------------

    def reset_episode_visits(self):
        """Reset visit counts at the beginning of each episode."""
        self.visit_counts = {}

    def record_visit(self, state):
        """Record a visit to the given state."""
        self.visit_counts[state] = self.visit_counts.get(state, 0) + 1

    # ------------------------------------------------------------------
    # Epsilon Decay
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        """Apply linear decay to epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    # ------------------------------------------------------------------
    # Model Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        """Save Q-table and agent configuration to disk."""
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        """Load Q-table and configuration from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS), data["q_table"])
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.gamma = data["gamma"]
