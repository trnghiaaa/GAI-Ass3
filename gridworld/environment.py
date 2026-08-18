"""
GridWorld environment with support for all tile types:
    apples, keys, chests, fire, rocks, and monsters.

State representation (hashable tuple for Q-table lookup):
    (row, col, has_key, collectibles_remaining, monster_positions)

    - collectibles_remaining : tuple of 0/1 flags for each collectible
    - monster_positions      : sorted tuple of (r, c) pairs (empty if none)
"""

import copy
import random
from gridworld.levels import LEVELS

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ACTION_DELTAS = {
    UP:    (-1, 0),
    DOWN:  ( 1, 0),
    LEFT:  ( 0,-1),
    RIGHT: ( 0, 1),
}
NUM_ACTIONS = 4


class GridWorldEnv:
    """Gridworld environment following assignment rules.

    Episode ends when:
        - All collectible rewards (apples + chests) are obtained, OR
        - The agent dies (fire or monster collision).
    """

    def __init__(self, level_id, monster_move_chance=0.4):
        if level_id not in LEVELS:
            raise ValueError(f"Unknown level {level_id}. Available: {list(LEVELS.keys())}")

        level = LEVELS[level_id]
        self.level_id = level_id
        self.monster_move_chance = monster_move_chance

        # Parse the grid
        raw_grid = level["grid"]
        self.rows = len(raw_grid)
        self.cols = len(raw_grid[0])

        # Static grid (rocks, fire, empty — things that never change)
        self.static_grid = []
        # Positions parsed from the original layout
        self.start_pos = None
        self.initial_collectibles = []   # list of (row, col, type)
        self.initial_key_pos = None
        self.initial_monster_positions = []

        for r, row_str in enumerate(raw_grid):
            static_row = []
            for c, ch in enumerate(row_str):
                if ch == "S":
                    self.start_pos = (r, c)
                    static_row.append(".")
                elif ch == "R":
                    static_row.append("R")
                elif ch == "F":
                    static_row.append("F")
                elif ch == "A":
                    self.initial_collectibles.append((r, c, "apple"))
                    static_row.append(".")
                elif ch == "K":
                    self.initial_key_pos = (r, c)
                    static_row.append(".")
                elif ch == "C":
                    self.initial_collectibles.append((r, c, "chest"))
                    static_row.append(".")
                elif ch == "M":
                    self.initial_monster_positions.append((r, c))
                    static_row.append(".")
                else:  # '.'
                    static_row.append(".")
            self.static_grid.append(static_row)

        assert self.start_pos is not None, "Level must have a start tile 'S'"
        self.num_collectibles = len(self.initial_collectibles)
        self.has_monsters = len(self.initial_monster_positions) > 0

        # Mutable state (set by reset)
        self.agent_pos = None
        self.has_key = False
        self.key_pos = None          # None once picked up
        self.collectibles = []       # list of (row, col, type) still remaining
        self.monster_positions = []  # list of [row, col]
        self.done = False

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset environment to initial state and return the state tuple."""
        self.agent_pos = self.start_pos
        self.has_key = False
        self.key_pos = self.initial_key_pos  # might be None
        self.collectibles = list(self.initial_collectibles)  # copy
        self.monster_positions = [list(p) for p in self.initial_monster_positions]
        self.done = False
        return self.get_state()

    def step(self, action):
        """Execute an action and return (next_state, reward, done, info).

        Order of operations:
            1. Move agent (wall / rock → no movement)
            2. Check agent on fire / monster → death
            3. Pick up items (apple, key, chest)
            4. Check if all collectibles obtained → done
            5. Move monsters (each 40 % chance)
            6. Check if any monster moved onto agent → death
        """
        assert not self.done, "Episode has ended. Call reset()."
        reward = 0.0
        info = {}

        # 1. Move agent ---------------------------------------------------
        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        if self._in_bounds(new_r, new_c) and self.static_grid[new_r][new_c] != "R":
            self.agent_pos = (new_r, new_c)

        # 2. Death check: fire --------------------------------------------
        if self.static_grid[self.agent_pos[0]][self.agent_pos[1]] == "F":
            self.done = True
            info["death"] = "fire"
            return self.get_state(), reward, True, info

        # 2b. Death check: agent stepped on a monster
        for mp in self.monster_positions:
            if self.agent_pos == tuple(mp):
                self.done = True
                info["death"] = "monster"
                return self.get_state(), reward, True, info

        # 3. Item pickup --------------------------------------------------
        # Key
        if self.key_pos is not None and self.agent_pos == self.key_pos:
            self.has_key = True
            self.key_pos = None  # consumed
            info["picked_up"] = "key"

        # Collectibles (apples and chests)
        remaining = []
        for (cr, cc, ctype) in self.collectibles:
            if self.agent_pos == (cr, cc):
                if ctype == "apple":
                    reward += 1.0
                    info["picked_up"] = "apple"
                elif ctype == "chest":
                    if self.has_key:
                        reward += 2.0
                        info["picked_up"] = "chest"
                    else:
                        # Chest stays — agent doesn't have the key
                        remaining.append((cr, cc, ctype))
                        continue
                # Item consumed (don't add back to remaining)
            else:
                remaining.append((cr, cc, ctype))
        self.collectibles = remaining

        # 4. All collectibles done? ----------------------------------------
        if len(self.collectibles) == 0:
            self.done = True
            info["victory"] = True
            return self.get_state(), reward, True, info

        # 5. Move monsters -------------------------------------------------
        if self.has_monsters:
            self._move_monsters()

        # 6. Monster-onto-agent check --------------------------------------
        for mp in self.monster_positions:
            if self.agent_pos == tuple(mp):
                self.done = True
                info["death"] = "monster"
                return self.get_state(), reward, True, info

        return self.get_state(), reward, False, info

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self):
        """Return a hashable state tuple for Q-table lookup.

        Format: (row, col, has_key, collectibles_remaining, monster_positions)
        """
        coll_remaining = tuple(
            1 if (cr, cc, ct) in self.collectibles else 0
            for (cr, cc, ct) in self.initial_collectibles
        )

        if self.has_monsters:
            m_pos = tuple(sorted(tuple(mp) for mp in self.monster_positions))
        else:
            m_pos = ()

        return (
            self.agent_pos[0],
            self.agent_pos[1],
            self.has_key,
            coll_remaining,
            m_pos,
        )

    # ------------------------------------------------------------------
    # Helpers for rendering / evaluation
    # ------------------------------------------------------------------

    def get_grid_for_render(self):
        """Build a 2-D list of tile chars reflecting the *current* state.

        Useful for the renderer to know exactly what to draw.
        """
        grid = [row[:] for row in self.static_grid]  # deep copy

        # Place remaining collectibles
        for (cr, cc, ct) in self.collectibles:
            if ct == "apple":
                grid[cr][cc] = "A"
            elif ct == "chest":
                grid[cr][cc] = "C"

        # Place key if still on map
        if self.key_pos is not None:
            grid[self.key_pos[0]][self.key_pos[1]] = "K"

        # Place monsters
        for mp in self.monster_positions:
            grid[mp[0]][mp[1]] = "M"

        # Place agent
        grid[self.agent_pos[0]][self.agent_pos[1]] = "P"  # Player

        return grid

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _move_monsters(self):
        """Each monster has a chance to move to a random valid neighbour."""
        occupied = set(tuple(mp) for mp in self.monster_positions)

        for i, mp in enumerate(self.monster_positions):
            if random.random() > self.monster_move_chance:
                continue

            # Collect valid moves (not rock, not fire, not another monster,
            # not out of bounds)
            valid_moves = []
            for action in range(NUM_ACTIONS):
                dr, dc = ACTION_DELTAS[action]
                nr, nc = mp[0] + dr, mp[1] + dc
                if (self._in_bounds(nr, nc)
                        and self.static_grid[nr][nc] not in ("R", "F")
                        and (nr, nc) not in occupied):
                    valid_moves.append((nr, nc))

            if valid_moves:
                old_pos = tuple(mp)
                new_pos = random.choice(valid_moves)
                self.monster_positions[i] = list(new_pos)
                occupied.discard(old_pos)
                occupied.add(new_pos)
