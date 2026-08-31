"""Tests for evaluator-only phase transition diagnostics."""

import unittest

from arena.evaluate import _aggregate_phase_diagnostics


class ArenaEvaluationTests(unittest.TestCase):
    def test_phase_diagnostics_use_reached_cohort_and_weighted_segments(self) -> None:
        phase1_death = {
            "final_phase": 1,
            "end_reason": "player_destroyed",
            "health_on_first_entry_phase2": None,
            "health_on_first_entry_phase3": None,
        }
        phase2_death = {
            "final_phase": 2,
            "end_reason": "player_destroyed",
            "health_on_first_entry_phase2": 60.0,
            "health_on_first_entry_phase3": None,
            "damage_taken_before_phase2": 40.0,
            "damage_taken_after_phase2": 60.0,
            "contact_events_before_phase2": 2,
            "contact_events_after_phase2": 3,
            "steps_survived_after_first_entering_phase2": 100,
            "steps_before_phase2": 100,
            "steps_after_phase2": 100,
            "danger_step_fraction_before_phase2": 0.5,
            "danger_step_fraction_after_phase2": 0.8,
            "mean_crowd_pressure_before_phase2": 0.2,
            "mean_crowd_pressure_after_phase2": 0.3,
            "minimum_enemy_distance_before_phase2": 5.0,
            "minimum_enemy_distance_after_phase2": 2.0,
        }
        phase3_timeout = {
            "final_phase": 3,
            "end_reason": "time_limit",
            "health_on_first_entry_phase2": 80.0,
            "health_on_first_entry_phase3": 50.0,
            "damage_taken_before_phase2": 20.0,
            "damage_taken_after_phase2": 10.0,
            "contact_events_before_phase2": 1,
            "contact_events_after_phase2": 1,
            "steps_survived_after_first_entering_phase2": 200,
            "steps_before_phase2": 200,
            "steps_after_phase2": 200,
            "danger_step_fraction_before_phase2": 0.25,
            "danger_step_fraction_after_phase2": 0.4,
            "mean_crowd_pressure_before_phase2": 0.1,
            "mean_crowd_pressure_after_phase2": 0.2,
            "minimum_enemy_distance_before_phase2": 15.0,
            "minimum_enemy_distance_after_phase2": 4.0,
        }

        result = _aggregate_phase_diagnostics(
            [phase1_death, phase2_death, phase3_timeout]
        )

        self.assertEqual(result["phase2_reached_cohort_episodes"], 2)
        self.assertEqual(result["phase3_reached_cohort_episodes"], 1)
        self.assertEqual(result["mean_health_on_first_entry_phase2"], 70.0)
        self.assertEqual(result["median_health_on_first_entry_phase2"], 70.0)
        self.assertEqual(result["mean_health_on_first_entry_phase3"], 50.0)
        self.assertEqual(result["median_health_on_first_entry_phase3"], 50.0)
        self.assertEqual(result["mean_damage_taken_before_phase2_transition"], 30.0)
        self.assertEqual(result["mean_damage_taken_after_phase2_transition"], 35.0)
        self.assertEqual(result["mean_contact_events_before_phase2"], 1.5)
        self.assertEqual(result["mean_contact_events_after_phase2"], 2.0)
        self.assertEqual(
            result["mean_steps_survived_after_first_entering_phase2"], 150.0
        )
        self.assertEqual(result["death_count_phase1"], 1)
        self.assertEqual(result["death_count_phase2"], 1)
        self.assertEqual(result["death_count_phase3_or_higher"], 0)
        self.assertAlmostEqual(result["death_rate_phase1"], 1.0 / 3.0)
        self.assertAlmostEqual(result["death_rate_phase2"], 1.0 / 3.0)
        self.assertAlmostEqual(
            result["danger_step_fraction_before_phase2"], 1.0 / 3.0
        )
        self.assertAlmostEqual(
            result["danger_step_fraction_after_phase2"], 160.0 / 300.0
        )
        self.assertAlmostEqual(
            result["mean_crowd_pressure_before_phase2"], 40.0 / 300.0
        )
        self.assertAlmostEqual(
            result["mean_crowd_pressure_after_phase2"], 70.0 / 300.0
        )
        self.assertEqual(result["mean_minimum_enemy_distance_before_phase2"], 10.0)
        self.assertEqual(result["mean_minimum_enemy_distance_after_phase2"], 3.0)


if __name__ == "__main__":
    unittest.main()
