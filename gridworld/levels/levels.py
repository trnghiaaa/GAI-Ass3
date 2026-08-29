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
        "description": "Collect the apples on the right using a shortest path.",
        "objectives": (
            "Collect all three apples.",
            "Demonstrate a shortest-path Q-learning policy.",
        ),
        "mechanics": ("apples", "rocks", "shortest-path planning"),
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
    1: {
        "title": "Cliffside Choice",
        "task": 2,
        "task_name": "SARSA Hazard Avoidance",
        "description": (
            "Choose between a short fire-edge lane and a slightly longer safe lane."
        ),
        "objectives": (
            "Collect the apple without stepping into fire.",
            "Compare Q-learning's short route with SARSA's conservative route.",
        ),
        "mechanics": ("apple", "fire", "safe-versus-risky routing"),
        # Rows 0-6 constrain the experiment to two reachable lanes. Row 7 is
        # the safe seven-step route; row 8 is the risky five-step route; fire
        # on row 9 punishes exploratory DOWN actions from the risky lane.
        "grid": [
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RRRRRRRRRR",
            "RR......RR",
            "RRS....ARR",
            "RRRFFFFRRR",
        ],
    },
    2: {
        "title": "Keykeeper's Meadow",
        "task": 3,
        "task_name": "Keys, Chests, and Multiple Rewards",
        "description": "Collect two apples, find the key, then open the chest.",
        "objectives": (
            "Collect both apples.",
            "Collect the key before opening the chest.",
        ),
        "mechanics": ("multiple apples", "key", "locked chest", "rocks"),
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
    3: {
        "title": "Vault Labyrinth",
        "task": 3,
        "task_name": "Keys, Chests, and Multiple Rewards",
        "description": "Solve a rock labyrinth containing apples, a key, and a chest.",
        "objectives": (
            "Collect both apples.",
            "Navigate to the key and unlock the chest.",
        ),
        "mechanics": ("multiple apples", "key", "locked chest", "maze"),
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
    4: {
        "title": "Monster Meadow",
        "task": 4,
        "task_name": "Stochastic Monster Transitions",
        "description": "Collect every apple while one roaming monster moves randomly.",
        "objectives": (
            "Collect all three apples.",
            "Avoid stepping onto the monster or letting it move onto the agent.",
        ),
        "mechanics": ("multiple apples", "one monster", "stochastic movement"),
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
    5: {
        "title": "Predator Passages",
        "task": 4,
        "task_name": "Stochastic Monster Transitions",
        "description": "Outmanoeuvre two monsters in tighter rock passages.",
        "objectives": (
            "Collect all three apples.",
            "Adapt to two independently moving monsters.",
        ),
        "mechanics": ("multiple apples", "two monsters", "rock corridors"),
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
    6: {
        "title": "Curiosity Maze",
        "task": 5,
        "task_name": "Intrinsic Exploration Reward",
        "description": "Explore a maze with sparse, distant apples.",
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
