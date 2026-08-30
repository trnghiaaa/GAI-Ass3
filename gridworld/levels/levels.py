"""Canonical level definitions for the Part I Gridworld.

Tile legend
-----------
``.`` empty, ``S`` start, ``R`` rock, ``F`` fire, ``A`` apple,
``K`` key, ``C`` chest, and ``M`` monster.

The descriptive metadata is intentionally kept beside each map so launchers and
renderers can present a useful level-select screen without duplicating assignment
knowledge elsewhere.
"""


LEVELS = {
    0: {
        "title": "Orchard Sprint",
        "task": 1,
        "task_name": "Basic Q-Learning",
        "description": "Plan one efficient route through the orchard to every apple on the right.",
        "objectives": (
            "Collect all three apples.",
            "Demonstrate a shortest-path Q-learning policy.",
        ),
        "mechanics": ("apples", "rocks", "shortest-path planning"),
        # The Task 1 wording says Level 0 contains only apples on the right
        # side. Multiple apples are therefore valid here: all three are in the
        # two rightmost columns, and no key, chest, fire, or monster is present.
        "grid": [
            "S..R....A.",
            "...R.R....",
            ".....R....",
            ".RR..R.R..",
            ".......R.A",
            "..R.R.....",
            "..R...RR..",
            "....R.....",
            ".R......A.",
            "..........",
        ],
    },
    1: {
        "title": "Cliffside Choice",
        "task": 2,
        "task_name": "SARSA Hazard Avoidance",
        "description": "Choose a fast fire-edge lane or a longer protected route.",
        "objectives": (
            "Collect the apple without stepping into fire.",
            "Compare Q-learning's short route with SARSA's conservative route.",
        ),
        "mechanics": ("apple", "fire", "safe-versus-risky routing"),
        # One apple keeps Task 2 focused on the policy difference rather than
        # the multi-reward/key/chest planning combination introduced in Task 3.
        # The direct row-7 route is shortest, but exploratory DOWN actions can
        # enter the fire below it.  The connected upper lanes give SARSA room
        # to learn a longer, more conservative policy while Q-learning can
        # still demonstrate the greedy shortest route.
        "grid": [
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "R........R",
            "R.RRRRR..R",
            "R........R",
            "R..RRR...R",
            "R........R",
            "RS......AR",
            "R.FFFFFF.R",
            "RRRRRRRRRR",
        ],
    },
    2: {
        "title": "Keykeeper's Meadow",
        "task": 3,
        "task_name": "Keys, Chests, and Multiple Rewards",
        "description": "Plan a complete route through branching gardens, then unlock the chest.",
        "objectives": (
            "Collect all three apples.",
            "Collect the key before opening the chest.",
        ),
        "mechanics": ("multiple apples", "key", "locked chest", "rocks"),
        "grid": [
            "S..R....A.",
            "...R.R....",
            ".A...R....",
            ".RR..R.R..",
            "....K..R..",
            "..R....R..",
            "..R.R.....",
            "....R.RR..",
            ".A......C.",
            "..........",
        ],
    },
    3: {
        "title": "Vault Labyrinth",
        "task": 3,
        "task_name": "Keys, Chests, and Multiple Rewards",
        "description": "Search a denser labyrinth for rewards, the vault key, and the chest.",
        "objectives": (
            "Collect all three apples.",
            "Navigate to the key and unlock the chest.",
        ),
        "mechanics": ("multiple apples", "key", "locked chest", "maze"),
        "grid": [
            "S.R....A..",
            "..R.RRR...",
            "..R.....R.",
            "..RRR.R.R.",
            "A....R....",
            ".RR..RR.R.",
            "...R...K..",
            ".R...R.RR.",
            ".R.A...R.C",
            "..........",
        ],
    },
    4: {
        "title": "Monster Meadow",
        "task": 4,
        "task_name": "Stochastic Monster Transitions",
        "description": "Use loops and cover to collect every apple around one roaming monster.",
        "objectives": (
            "Collect all three apples.",
            "Avoid stepping onto the monster or letting it move onto the agent.",
        ),
        "mechanics": ("multiple apples", "one monster", "stochastic movement"),
        "grid": [
            "S..R...A..",
            ".R.R.R....",
            ".R...R.R..",
            "...R...R..",
            "RR...M....",
            "...R.R.R..",
            ".A...R....",
            ".RR.R..R..",
            "......R.A.",
            "..........",
        ],
    },
    5: {
        "title": "Predator Passages",
        "task": 4,
        "task_name": "Stochastic Monster Transitions",
        "description": "Outmanoeuvre two independent predators across connected tactical passages.",
        "objectives": (
            "Collect all three apples.",
            "Adapt to two independently moving monsters.",
        ),
        "mechanics": ("multiple apples", "two monsters", "rock corridors"),
        "grid": [
            "S.R....A..",
            "..R.RR....",
            "....R..M..",
            ".RR...R...",
            "...R....R.",
            ".M...RR...",
            "..R....A..",
            "..RR.R....",
            "A....R....",
            "..........",
        ],
    },
    6: {
        "title": "Curiosity Maze",
        "task": 5,
        "task_name": "Intrinsic Exploration Reward",
        "description": "Explore a sparse branching maze and use curiosity to discover distant rewards.",
        "objectives": (
            "Collect all three apples.",
            "Compare learning with and without the count-based intrinsic bonus.",
        ),
        "mechanics": ("sparse apples", "maze", "intrinsic exploration"),
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
