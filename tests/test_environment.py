import random

import pytest

from gridworld.environment import (
    DOWN,
    LEFT,
    RIGHT,
    STATE_SCHEMA,
    GridWorldEnv,
    validate_level_definition,
)
from gridworld.levels import LEVELS


def test_all_canonical_levels_validate_and_expose_metadata():
    assert sorted(LEVELS) == list(range(7))
    for level_id, definition in LEVELS.items():
        grid = validate_level_definition(level_id, definition)
        env = GridWorldEnv(level_id, seed=17)
        state = env.reset()
        assert grid
        assert env.title and env.description and env.objectives and env.mechanics
        assert env.state_schema == STATE_SCHEMA
        assert len(state) == len(STATE_SCHEMA)


def test_level_progression_keeps_each_rubric_role_and_unique_layout():
    grids = {level: tuple(definition["grid"]) for level, definition in LEVELS.items()}
    assert len(set(grids.values())) == 7
    assert all(len(grid) == 10 and all(len(row) == 10 for row in grid)
               for grid in grids.values())

    assert set("".join(grids[0])) <= set(".SRA")
    assert "".join(grids[0]).count("A") == 3
    assert all(column >= 8 for row in range(10) for column, tile in enumerate(grids[0][row])
               if tile == "A")
    assert "".join(grids[1]).count("A") == 1
    assert "F" in "".join(grids[1])
    for level in (2, 3):
        joined = "".join(grids[level])
        assert joined.count("A") >= 2
        assert joined.count("K") == joined.count("C") == 1
    assert "".join(grids[4]).count("M") == 1
    assert "".join(grids[5]).count("M") == 2
    assert "".join(grids[6]).count("A") == 3


@pytest.mark.parametrize(
    "definition, error",
    [
        ({"grid": ["S.", "..A"]}, "rectangular"),
        ({"grid": ["SS.A"]}, "exactly one start"),
        ({"grid": ["SX.A"]}, "invalid tile"),
        ({"grid": ["S.."]}, "at least one apple or chest"),
        ({"grid": ["S.C"]}, "chest but not exactly one key"),
        ({"grid": ["SKA"]}, "key but no chest"),
        ({"grid": ["SR", "RA"]}, "unreachable"),
    ],
)
def test_map_validation_rejects_invalid_definitions(definition, error):
    with pytest.raises((TypeError, ValueError), match=error):
        validate_level_definition("bad", definition)


def test_rocks_and_boundaries_block_without_reward(level_factory):
    level_id = level_factory(["SRA", "..."])
    env = GridWorldEnv(level_id)
    start = env.reset()
    next_state, reward, done, info = env.step(RIGHT)
    assert next_state == start
    assert env.agent_pos == (0, 0)
    assert reward == 0 and not done
    assert info["blocked"] and info["blocked_reason"] == "rock"

    next_state, reward, done, info = env.step(LEFT)
    assert next_state == start
    assert info["blocked_reason"] == "boundary"


def test_fire_causes_immediate_death_and_keeps_reward_zero(level_factory):
    level_id = level_factory(["SF.", "..A"])
    env = GridWorldEnv(level_id)
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert done and info["death"] == "fire"
    assert reward == info["env_reward"] == 0.0
    snapshot = env.get_render_snapshot()
    assert snapshot["terrain"][0][1] == "F"
    assert snapshot["entities"]["agent"] == (0, 1)


def test_apple_reward_and_reward_completion_terminal(level_factory):
    level_id = level_factory(["SA"])
    env = GridWorldEnv(level_id)
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert reward == info["env_reward"] == 1.0
    assert done and info["victory"]
    assert env.collectibles == []


def test_key_locked_chest_and_exact_reward_accounting(level_factory):
    level_id = level_factory(["S.C", ".K.", "A.."])
    env = GridWorldEnv(level_id)
    env.reset()

    env.step(RIGHT)
    _, reward, done, info = env.step(RIGHT)
    assert reward == 0.0 and not done and info["chest_locked"]
    assert any(item[2] == "chest" for item in env.collectibles)

    env.step(LEFT)
    _, reward, done, info = env.step(DOWN)
    assert reward == 0.0 and not done and info["picked_up"] == "key"
    assert env.has_key and env.key_pos is None

    env.step(0)  # up
    _, reward, done, info = env.step(RIGHT)
    assert reward == 2.0 and not done and info["picked_up"] == "chest"

    env.step(LEFT)
    env.step(LEFT)
    env.step(DOWN)
    _, reward, done, info = env.step(DOWN)
    assert reward == 1.0 and done and info["victory"]


def test_stepping_onto_monster_kills_before_monster_phase(level_factory):
    level_id = level_factory(["SMA"])
    env = GridWorldEnv(level_id, monster_move_chance=1.0, seed=1)
    env.reset()
    _, reward, done, info = env.step(RIGHT)
    assert reward == 0.0 and done and info["death"] == "monster"
    assert info["monster_moves"] == []


class _CollisionRng:
    def random(self):
        return 0.0

    def choice(self, values):
        return (0, 1) if (0, 1) in values else values[0]


def test_monster_can_move_onto_agent_and_full_state_tracks_move(level_factory):
    level_id = level_factory(["S.A", ".M.", "..."])
    env = GridWorldEnv(level_id, monster_move_chance=1.0)
    before = env.reset()
    env._rng = _CollisionRng()
    after, reward, done, info = env.step(RIGHT)
    assert reward == 0.0 and done and info["death"] == "monster"
    assert info["monster_moves"] == [
        {"monster": 0, "from": (1, 1), "to": (0, 1)}
    ]
    assert before[4] != after[4]
    # Nearby monsters use exact relative offsets; all monsters are still counted.
    assert before[4] == (((1, 1),), 0)
    assert after[4] == (((0, 0),), 0)


def test_monster_move_chance_zero_and_one_are_exact(level_factory):
    level_id = level_factory(["S.A", ".M.", "..."])
    stationary = GridWorldEnv(level_id, monster_move_chance=0.0, seed=4)
    stationary.reset()
    _, _, _, info = stationary.step(LEFT)
    assert info["blocked"] and info["blocked_reason"] == "boundary"
    assert info["monster_moves"] == []

    moving = GridWorldEnv(level_id, monster_move_chance=1.0, seed=4)
    moving.reset()
    _, _, _, info = moving.step(LEFT)
    assert info["blocked"] and info["blocked_reason"] == "boundary"
    assert len(info["monster_moves"]) == 1


def test_seeded_monster_transitions_are_reproducible():
    left = GridWorldEnv(5, seed=2026)
    right = GridWorldEnv(5, seed=2026)
    assert left.reset() == right.reset()
    for action in [RIGHT, DOWN, DOWN, RIGHT, LEFT, DOWN]:
        if left.done:
            break
        assert left.step(action) == right.step(action)


def test_monster_probability_is_statistically_close_to_config(level_factory):
    level_id = level_factory(["S.A", ".M.", "..."])
    env = GridWorldEnv(level_id, monster_move_chance=0.4, seed=12345)
    moved = 0
    trials = 5000
    for _ in range(trials):
        env.reset()
        _, _, _, info = env.step(LEFT)
        moved += bool(info["monster_moves"])
    assert moved / trials == pytest.approx(0.4, abs=0.025)


@pytest.mark.parametrize("action", [-1, 4, 1.5, True, "RIGHT"])
def test_invalid_actions_fail_clearly(action):
    env = GridWorldEnv(0)
    env.reset()
    with pytest.raises((TypeError, ValueError)):
        env.step(action)


def test_level1_has_deliberate_safe_and_risky_lanes():
    env = GridWorldEnv(1)
    env.reset()
    assert env.start_pos == (7, 1)
    assert env.initial_collectibles[0][:2] == (7, 8)
    assert all(env.static_grid[8][column] == "F" for column in range(2, 8))
