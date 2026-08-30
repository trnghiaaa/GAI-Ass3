"""Deterministic-testable Gridworld mechanics for Part I.

The environment owns rules and state only; rendering and learning rewards stay
outside it. Environment rewards are exactly those in the assignment: apples
give ``+1``, chests opened while holding a key give ``+2``, and every other
transition gives ``0``.

State schema (stable across all seven levels)
------------------------------------------------
``(row, column, has_key, collectible_flags, monster_observation)``

``monster_observation`` represents every monster without exploding the tabular
state space: threats within Manhattan distance two use exact relative offsets,
while the second field counts monsters outside that actionable sensor radius.
The full absolute positions remain available to the renderer.  This deliberate
local-threat abstraction lets the agents generalise avoidance behaviour across
thousands of stochastic layouts while never hiding how many monsters exist.
"""

from collections import deque
from collections.abc import Mapping, Sequence
import hashlib
import math
from numbers import Integral, Real
import random

from gridworld.levels import LEVELS


# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
NUM_ACTIONS = len(ACTION_NAMES)

ALLOWED_TILES = frozenset(".SRFAKCM")
STATE_SCHEMA = (
    "agent_row",
    "agent_column",
    "has_key",
    "collectible_flags",
    "monster_observation",
)
MONSTER_SENSOR_RADIUS = 2


def validate_level_definition(level_id, level):
    """Validate one level and return its grid as an immutable tuple.

    Validation is deliberately strict so a malformed assignment map fails at
    startup with a useful message instead of becoming a silent empty tile or a
    mid-episode ``IndexError``.
    """

    prefix = f"Level {level_id!r}"
    if not isinstance(level, Mapping):
        raise TypeError(f"{prefix} definition must be a mapping")
    if "grid" not in level:
        raise ValueError(f"{prefix} is missing required 'grid' data")

    raw_grid = level["grid"]
    if isinstance(raw_grid, (str, bytes)) or not isinstance(raw_grid, Sequence):
        raise TypeError(f"{prefix} grid must be a sequence of row strings")
    if not raw_grid:
        raise ValueError(f"{prefix} grid cannot be empty")
    if any(not isinstance(row, str) for row in raw_grid):
        raise TypeError(f"{prefix} grid rows must all be strings")
    if not raw_grid[0]:
        raise ValueError(f"{prefix} grid rows cannot be empty")

    width = len(raw_grid[0])
    for row_index, row in enumerate(raw_grid):
        if len(row) != width:
            raise ValueError(
                f"{prefix} grid must be rectangular; row {row_index} has "
                f"width {len(row)} instead of {width}"
            )
        invalid = sorted(set(row) - ALLOWED_TILES)
        if invalid:
            raise ValueError(
                f"{prefix} row {row_index} contains invalid tile(s): {invalid}"
            )

    positions = {
        tile: [
            (row_index, column_index)
            for row_index, row in enumerate(raw_grid)
            for column_index, value in enumerate(row)
            if value == tile
        ]
        for tile in ALLOWED_TILES
    }

    if len(positions["S"]) != 1:
        raise ValueError(
            f"{prefix} must contain exactly one start tile 'S'; "
            f"found {len(positions['S'])}"
        )
    reward_count = len(positions["A"]) + len(positions["C"])
    if reward_count == 0:
        raise ValueError(f"{prefix} must contain at least one apple or chest")
    if len(positions["K"]) > 1:
        raise ValueError(f"{prefix} supports at most one key")
    if positions["C"] and len(positions["K"]) != 1:
        raise ValueError(f"{prefix} contains a chest but not exactly one key")
    if positions["K"] and not positions["C"]:
        raise ValueError(f"{prefix} contains a key but no chest to unlock")

    # Rewards, keys, chests, and monsters must be in the start component when
    # rocks and lethal fire are treated as impassable. This prevents impossible
    # objectives while still allowing fire to border a valid route.
    start = positions["S"][0]
    reachable = {start}
    frontier = deque([start])
    height = len(raw_grid)
    while frontier:
        row, column = frontier.popleft()
        for row_delta, column_delta in ACTION_DELTAS.values():
            neighbour = (row + row_delta, column + column_delta)
            nr, nc = neighbour
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if raw_grid[nr][nc] in ("R", "F") or neighbour in reachable:
                continue
            reachable.add(neighbour)
            frontier.append(neighbour)

    required_positions = []
    for tile in ("A", "K", "C", "M"):
        required_positions.extend(positions[tile])
    unreachable = sorted(position for position in required_positions if position not in reachable)
    if unreachable:
        raise ValueError(
            f"{prefix} has unreachable objective/entity tile(s): {unreachable}"
        )

    return tuple(raw_grid)


def level_layout_fingerprint(level_id: int) -> str:
    """Return a stable identity for the current tile layout of one level.

    Tabular policies are map-specific.  Persisting this fingerprint with each
    model prevents a visually valid but obsolete Q-table from being played
    after a level redesign.
    """

    if level_id not in LEVELS:
        raise ValueError(
            f"Unknown level {level_id!r}. Available levels: {sorted(LEVELS)}"
        )
    grid = validate_level_definition(level_id, LEVELS[level_id])
    payload = "gridworld-layout-v1\n" + "\n".join(grid)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_monster_move_chance(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("monster_move_chance must be a real number between 0 and 1")
    chance = float(value)
    if not math.isfinite(chance) or not 0.0 <= chance <= 1.0:
        raise ValueError("monster_move_chance must be between 0 and 1 inclusive")
    return chance


class GridWorldEnv:
    """Gridworld environment implementing the assignment's exact mechanics."""

    state_schema = STATE_SCHEMA

    def __init__(self, level_id, monster_move_chance=0.4, seed=None):
        if level_id not in LEVELS:
            raise ValueError(
                f"Unknown level {level_id!r}. Available levels: {sorted(LEVELS)}"
            )

        level = LEVELS[level_id]
        raw_grid = validate_level_definition(level_id, level)

        self.level_id = level_id
        self.layout_fingerprint = level_layout_fingerprint(level_id)
        self.level_metadata = {
            key: value for key, value in level.items() if key != "grid"
        }
        self.title = level.get("title", f"Level {level_id}")
        self.task = level.get("task")
        self.description = level.get("description", "")
        self.objectives = tuple(level.get("objectives", ()))
        self.mechanics = tuple(level.get("mechanics", ()))
        self.monster_move_chance = _validate_monster_move_chance(
            monster_move_chance
        )
        self._rng = random.Random(seed)

        self.rows = len(raw_grid)
        self.cols = len(raw_grid[0])
        self.static_grid = []
        self.start_pos = None
        self.initial_collectibles = []
        self.initial_key_pos = None
        self.initial_monster_positions = []

        for row_index, row in enumerate(raw_grid):
            static_row = []
            for column_index, tile in enumerate(row):
                position = (row_index, column_index)
                if tile == "S":
                    self.start_pos = position
                    static_row.append(".")
                elif tile == "R":
                    static_row.append("R")
                elif tile == "F":
                    static_row.append("F")
                elif tile == "A":
                    self.initial_collectibles.append((*position, "apple"))
                    static_row.append(".")
                elif tile == "K":
                    self.initial_key_pos = position
                    static_row.append(".")
                elif tile == "C":
                    self.initial_collectibles.append((*position, "chest"))
                    static_row.append(".")
                elif tile == "M":
                    self.initial_monster_positions.append(position)
                    static_row.append(".")
                else:  # validated empty tile
                    static_row.append(".")
            self.static_grid.append(static_row)

        self.num_collectibles = len(self.initial_collectibles)
        self.has_monsters = bool(self.initial_monster_positions)

        # Mutable episode state is populated by reset(). Keeping construction
        # separate makes accidental use-before-reset fail clearly.
        self.agent_pos = None
        self.has_key = False
        self.key_pos = None
        self.collectibles = []
        self.monster_positions = []
        self.done = False

    def reset(self, seed=None):
        """Reset the level and optionally restart its independent RNG stream."""

        if seed is not None:
            self._rng.seed(seed)
        self.agent_pos = self.start_pos
        self.has_key = False
        self.key_pos = self.initial_key_pos
        self.collectibles = list(self.initial_collectibles)
        self.monster_positions = [list(position) for position in self.initial_monster_positions]
        self.done = False
        return self.get_state()

    def step(self, action):
        """Apply one action and return ``(state, env_reward, done, info)``.

        ``info['env_reward']`` always mirrors the returned reward. No intrinsic
        bonus, step penalty, death penalty, or other shaping is applied here.
        """

        if self.agent_pos is None:
            raise RuntimeError("Environment has not been reset; call reset() first")
        if self.done:
            raise RuntimeError("Episode has ended; call reset() before stepping again")
        action = self._validate_action(action)

        origin = self.agent_pos
        info = {
            "level_id": self.level_id,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "agent_from": origin,
            "agent_to": origin,
            "agent_moved": False,
            "blocked": False,
            "blocked_reason": None,
            "env_reward": 0.0,
            "event": "move",
            "events": [],
            "monster_moves": [],
        }
        reward = 0.0
        primary_event = "move"

        row_delta, column_delta = ACTION_DELTAS[action]
        destination = (
            origin[0] + row_delta,
            origin[1] + column_delta,
        )
        if not self._in_bounds(*destination):
            info["blocked"] = True
            info["blocked_reason"] = "boundary"
            info["events"].append("blocked_boundary")
            primary_event = "blocked"
        elif self.static_grid[destination[0]][destination[1]] == "R":
            info["blocked"] = True
            info["blocked_reason"] = "rock"
            info["events"].append("blocked_rock")
            primary_event = "blocked"
        else:
            self.agent_pos = destination
            info["agent_to"] = destination
            info["agent_moved"] = True
            info["events"].append("agent_moved")

        # Entering fire or an occupied monster tile kills immediately.
        if self.static_grid[self.agent_pos[0]][self.agent_pos[1]] == "F":
            self.done = True
            info["death"] = "fire"
            info["events"].append("death_fire")
            return self._finish_step(reward, info, "death_fire")
        if self.agent_pos in self._monster_position_set():
            self.done = True
            info["death"] = "monster"
            info["events"].append("death_monster")
            return self._finish_step(reward, info, "death_monster")

        # Key collection has no environment reward.
        if self.key_pos is not None and self.agent_pos == self.key_pos:
            self.has_key = True
            self.key_pos = None
            info["picked_up"] = "key"
            info["events"].append("key_collected")
            primary_event = "key_collected"

        remaining = []
        for row, column, collectible_type in self.collectibles:
            if self.agent_pos != (row, column):
                remaining.append((row, column, collectible_type))
                continue

            if collectible_type == "apple":
                reward += 1.0
                info["picked_up"] = "apple"
                info["events"].append("apple_collected")
                primary_event = "apple_collected"
            elif self.has_key:
                reward += 2.0
                info["picked_up"] = "chest"
                info["events"].append("chest_opened")
                primary_event = "chest_opened"
            else:
                # A locked chest remains available for a later visit.
                remaining.append((row, column, collectible_type))
                info["chest_locked"] = True
                info["events"].append("chest_locked")
                primary_event = "chest_locked"
        self.collectibles = remaining

        # Obtaining the final reward ends the episode immediately. This gives
        # the assignment's reward-completion terminal condition precedence over
        # a subsequent monster phase.
        if not self.collectibles:
            self.done = True
            info["victory"] = True
            info["events"].append("victory")
            return self._finish_step(reward, info, "victory")

        if self.has_monsters:
            monster_moves = self._move_monsters()
            info["monster_moves"] = monster_moves
            if monster_moves:
                info["events"].append("monsters_moved")

        if self.agent_pos in self._monster_position_set():
            self.done = True
            info["death"] = "monster"
            info["events"].append("death_monster")
            return self._finish_step(reward, info, "death_monster")

        return self._finish_step(reward, info, primary_event)

    def get_state(self):
        """Return the stable, hashable tabular-learning state."""

        if self.agent_pos is None:
            raise RuntimeError("Environment has not been reset; call reset() first")
        collectible_flags = tuple(
            int((row, column, kind) in self.collectibles)
            for row, column, kind in self.initial_collectibles
        )
        if self.has_monsters:
            nearby_offsets = []
            far_monsters = 0
            for monster_row, monster_column in self._monster_position_set():
                row_offset = monster_row - self.agent_pos[0]
                column_offset = monster_column - self.agent_pos[1]
                if abs(row_offset) + abs(column_offset) <= MONSTER_SENSOR_RADIUS:
                    nearby_offsets.append((row_offset, column_offset))
                else:
                    far_monsters += 1
            monster_observation = (tuple(sorted(nearby_offsets)), far_monsters)
        else:
            monster_observation = ()
        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.has_key,
            collectible_flags,
            monster_observation,
        )

    def get_render_snapshot(self):
        """Return independent terrain, item, entity, and status layers.

        A renderer can draw the layers in order without losing a fire tile under
        a dead agent or an apple temporarily occupied by a monster.
        """

        if self.agent_pos is None:
            raise RuntimeError("Environment has not been reset; call reset() first")
        apples = []
        chests = []
        for row, column, kind in self.collectibles:
            target = apples if kind == "apple" else chests
            target.append((row, column))

        return {
            "level_id": self.level_id,
            "metadata": dict(self.level_metadata),
            "terrain": tuple(tuple(row) for row in self.static_grid),
            "items": {
                "apples": tuple(sorted(apples)),
                "chests": tuple(sorted(chests)),
                "key": self.key_pos,
            },
            "entities": {
                "agent": self.agent_pos,
                "monsters": tuple(sorted(self._monster_position_set())),
            },
            "status": {
                "has_key": self.has_key,
                "collectibles_remaining": len(self.collectibles),
                "done": self.done,
            },
        }

    def get_grid_for_render(self):
        """Return the legacy single-character composite render grid."""

        snapshot = self.get_render_snapshot()
        grid = [list(row) for row in snapshot["terrain"]]
        for row, column in snapshot["items"]["apples"]:
            grid[row][column] = "A"
        for row, column in snapshot["items"]["chests"]:
            grid[row][column] = "C"
        if snapshot["items"]["key"] is not None:
            row, column = snapshot["items"]["key"]
            grid[row][column] = "K"
        for row, column in snapshot["entities"]["monsters"]:
            grid[row][column] = "M"
        row, column = snapshot["entities"]["agent"]
        grid[row][column] = "P"
        return grid

    def _validate_action(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError(f"action must be an integer in 0..{NUM_ACTIONS - 1}")
        action = int(action)
        if action not in ACTION_DELTAS:
            raise ValueError(
                f"Unknown action {action}; valid actions are 0..{NUM_ACTIONS - 1}"
            )
        return action

    def _finish_step(self, reward, info, event):
        reward = float(reward)
        info["event"] = event
        info["env_reward"] = reward
        info["agent_to"] = self.agent_pos
        info["has_key"] = self.has_key
        info["collectibles_remaining"] = len(self.collectibles)
        return self.get_state(), reward, self.done, info

    def _in_bounds(self, row, column):
        return 0 <= row < self.rows and 0 <= column < self.cols

    def _monster_position_set(self):
        return {tuple(position) for position in self.monster_positions}

    def _move_monsters(self):
        """Independently sample and execute one allowed move per monster."""

        occupied = self._monster_position_set()
        moves = []
        for monster_index, monster_position in enumerate(self.monster_positions):
            if self._rng.random() >= self.monster_move_chance:
                continue

            valid_moves = []
            for row_delta, column_delta in ACTION_DELTAS.values():
                candidate = (
                    monster_position[0] + row_delta,
                    monster_position[1] + column_delta,
                )
                row, column = candidate
                if not self._in_bounds(row, column):
                    continue
                if self.static_grid[row][column] in ("R", "F"):
                    continue
                if candidate in occupied:
                    continue
                valid_moves.append(candidate)

            if not valid_moves:
                continue
            old_position = tuple(monster_position)
            new_position = self._rng.choice(valid_moves)
            self.monster_positions[monster_index] = list(new_position)
            occupied.remove(old_position)
            occupied.add(new_position)
            moves.append(
                {
                    "monster": monster_index,
                    "from": old_position,
                    "to": new_position,
                }
            )
        return moves
