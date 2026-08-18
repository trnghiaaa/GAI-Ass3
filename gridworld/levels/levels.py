"""
Level definitions for the Gridworld environment.

Tile legend:
    .  = Empty (walkable)
    S  = Start (agent spawn)
    R  = Rock (blocks movement)
    F  = Fire (instant death)
    A  = Apple (+1 reward, consumed)
    K  = Key (no reward, allows opening chests)
    C  = Chest (+2 reward if agent has key)
    M  = Monster (instant death, moves probabilistically)
"""

LEVELS = {
    # ------------------------------------------------------------------
    # Level 0 — Basic Apples (Task 1: Q-Learning)
    # Apples on the right side, some rocks as obstacles.
    # ------------------------------------------------------------------
    0: {
        "description": "Basic apples – learn shortest path with Q-Learning",
        "grid": [
            "S.........",
            "..........",
            "..RR......",
            "........A.",
            "..........",
            ".....R....",
            ".....R..A.",
            "..........",
            "........A.",
            "..........",
        ],
    },

    # ------------------------------------------------------------------
    # Level 1 — Fire Corridor / Cliff-Walk (Task 2: SARSA)
    # Fire blocks the direct east path along the bottom.
    # Q-Learning walks the cliff edge (row 7); SARSA routes higher.
    # ------------------------------------------------------------------
    1: {
        "description": "Fire corridor – SARSA learns a safer route than Q-Learning",
        "grid": [
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "..........",
            "........A.",
            "........A.",
            "SFFFFFFF..",
            "........A.",
        ],
    },

    # ------------------------------------------------------------------
    # Level 2 — Key and Chest, simple (Task 3)
    # Agent must collect key before opening chest.
    # ------------------------------------------------------------------
    2: {
        "description": "Key and chest – simple layout",
        "grid": [
            "S.........",
            "..R.......",
            "..R.A.....",
            "..........",
            "....K.....",
            ".......R..",
            "..A....R..",
            "..........",
            "........C.",
            "..........",
        ],
    },

    # ------------------------------------------------------------------
    # Level 3 — Key and Chest, harder (Task 3)
    # More rocks funnel the agent through corridors.
    # ------------------------------------------------------------------
    3: {
        "description": "Key and chest – harder, rock corridors",
        "grid": [
            "S..R......",
            "...R......",
            "...R..A...",
            ".....RRR..",
            "..A.......",
            "....R.....",
            "RRR.R..K..",
            "..........",
            ".....RRRR.",
            "........C.",
        ],
    },

    # ------------------------------------------------------------------
    # Level 4 — Monsters, simple (Task 4)
    # One monster in an open area.  Agent must avoid it while
    # collecting apples.
    # ------------------------------------------------------------------
    4: {
        "description": "One monster – stochastic danger in open area",
        "grid": [
            "S.........",
            "..........",
            ".......A..",
            "...R......",
            "...R......",
            "......M...",
            "..........",
            "..A.......",
            "........A.",
            "..........",
        ],
    },

    # ------------------------------------------------------------------
    # Level 5 — Monsters, harder (Task 4)
    # Two monsters with tighter corridors created by rocks.
    # ------------------------------------------------------------------
    5: {
        "description": "Two monsters – tighter corridors, more danger",
        "grid": [
            "S.R.......",
            "..R.......",
            "......M.A.",
            ".RRR......",
            ".....RR...",
            "..M.......",
            "....R..A..",
            "....R.....",
            "..A.......",
            "..........",
        ],
    },

    # ------------------------------------------------------------------
    # Level 6 — Intrinsic Reward (Task 5)
    # Sparse rewards far from start; maze-like rocks create dead ends.
    # Intrinsic exploration bonus helps the agent discover distant
    # apples faster.
    # ------------------------------------------------------------------
    6: {
        "description": "Sparse rewards in a maze – intrinsic reward needed",
        "grid": [
            "S..R......",
            ".R.R..R...",
            ".R....R...",
            "...RR.R...",
            "RR......R.",
            "A..R......",
            "...R.RR...",
            ".R.......A",
            ".RRR..R...",
            "......R..A",
        ],
    },
}
