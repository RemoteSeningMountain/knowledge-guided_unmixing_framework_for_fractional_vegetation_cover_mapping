from __future__ import annotations

import unittest

import numpy as np

from method_equations import (
    assign_vertices,
    compute_dsint,
    compute_mtv,
    polar_phenological_cycles,
    project_temporal_index,
    temporal_polar_angles,
)
from endmember_extraction import parse_band_date, parse_date_sequence
from mixing_models import (
    bilinear_mixing,
    intimate_like_mixing,
    linear_mixing,
    polynomial_nonlinear_mixing,
)
from spectral_indices import compute_indices
from mixed_spectra_synthesis import sample_batch


class WorkflowEquationTests(unittest.TestCase):
    def test_yyyymmdd_dates_are_parsed_as_calendar_dates(self) -> None:
        parsed = parse_date_sequence(["20250101", "20250109"])
        self.assertEqual(int(np.diff(parsed).astype("timedelta64[D]").astype(np.int64)[0]), 8)
        self.assertEqual(parse_band_date("0_20250101_EVI", "example.tif", 1), "20250101")

    def test_temporal_index_is_projected_into_cartesian_coordinates(self) -> None:
        dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2021-01-01"), np.timedelta64(8, "D"))
        values = np.linspace(0.1, 0.9, dates.size)[:, None, None]
        angles, _ = temporal_polar_angles(dates)
        projected_x, projected_y = project_temporal_index(values, dates)
        np.testing.assert_allclose(projected_x[:, 0, 0], values[:, 0, 0] * np.cos(angles))
        np.testing.assert_allclose(projected_y[:, 0, 0], values[:, 0, 0] * np.sin(angles))

    def test_dynamic_polar_cycles_supply_target_and_two_preceding_years(self) -> None:
        dates = np.arange(np.datetime64("2022-10-01"), np.datetime64("2026-04-01"), np.timedelta64(8, "D"))
        angles, _ = temporal_polar_angles(dates)
        evi = (0.5 - 0.3 * np.cos(angles))[:, None, None]
        model = polar_phenological_cycles(evi, dates, minimum_observations_per_cycle=30)
        self.assertTrue(model["valid_three_cycle_support"][0, 0])
        self.assertGreaterEqual(model["target_mask"][:, 0, 0].sum(), 30)
        self.assertGreaterEqual(model["previous_mask_1"][:, 0, 0].sum(), 30)
        self.assertGreaterEqual(model["previous_mask_2"][:, 0, 0].sum(), 30)
        self.assertFalse(np.any(model["target_mask"] & model["previous_mask_1"]))
        target_duration_days = (
            model["target_cycle_end"][0, 0] - model["target_cycle_start"][0, 0]
        ) * 365.0
        self.assertNotAlmostEqual(float(target_duration_days), 365.0, places=3)
        angular_difference = np.mod(
            model["long_term_phenological_year_start_angle"][0, 0]
            - model["mean_vector_angle"][0, 0],
            2.0 * np.pi,
        )
        self.assertAlmostEqual(float(angular_difference), np.pi)

    def test_descending_series_end_is_not_a_complete_dormant_boundary(self) -> None:
        dates = np.arange(np.datetime64("2022-10-01"), np.datetime64("2026-01-01"), np.timedelta64(8, "D"))
        angles, _ = temporal_polar_angles(dates)
        evi = (0.5 - 0.3 * np.cos(angles))[:, None, None]
        evi[-12:, 0, 0] = np.linspace(0.4, -0.2, 12)
        model = polar_phenological_cycles(evi, dates, minimum_observations_per_cycle=30)
        self.assertFalse(model["valid_three_cycle_support"][0, 0])

    def test_dsint_uses_15_to_80_percent_cumulative_interval(self) -> None:
        evi = np.asarray([1.0, 1.0, 2.0, 3.0, 2.0, 1.0])[:, None, None]
        self.assertEqual(compute_dsint(evi, time_step_days=8.0)[0, 0], 16.0)

    def test_mtv_is_population_standard_deviation(self) -> None:
        ndmi = np.asarray([0.0, 1.0, 2.0])[:, None, None]
        np.testing.assert_allclose(compute_mtv(ndmi)[0, 0], np.std([0.0, 1.0, 2.0], ddof=0))

    def test_vertex_assignment_matches_feature_space_positions(self) -> None:
        triangle = np.asarray([[0.01, 0.01], [0.90, 0.25], [0.30, 0.95]])
        assigned = assign_vertices(triangle)
        np.testing.assert_array_equal(assigned["bare_land"], triangle[0])
        np.testing.assert_array_equal(assigned["woody_vegetation"], triangle[1])
        np.testing.assert_array_equal(assigned["herbaceous_vegetation"], triangle[2])

    def test_linear_and_bilinear_equations(self) -> None:
        endmembers = np.asarray([[[[0.2]], [[0.8]], [[0.0]]]])
        abundances = np.asarray([[0.25, 0.75, 0.0]])
        linear = linear_mixing(endmembers, abundances)
        np.testing.assert_allclose(linear, 0.65)
        bilinear = bilinear_mixing(endmembers, abundances, np.asarray([0.5]))
        np.testing.assert_allclose(bilinear, 0.65 + 0.5 * 0.25 * 0.75 * 0.2 * 0.8)

    def test_polynomial_equation(self) -> None:
        endmembers = np.asarray([[[[0.2]], [[0.8]], [[0.0]]]])
        abundances = np.asarray([[0.25, 0.75, 0.0]])
        result = polynomial_nonlinear_mixing(endmembers, abundances, np.asarray([-0.2]))
        np.testing.assert_allclose(result, 0.65 - 0.2 * 0.65**2)

    def test_intimate_like_equation(self) -> None:
        endmembers = np.asarray([[[[0.25]], [[1.0]], [[0.0]]]])
        abundances = np.asarray([[0.5, 0.5, 0.0]])
        alpha = np.asarray([0.5])
        beta = np.asarray([0.2])
        generalized = (0.5 * 0.25**0.5 + 0.5 * 1.0**0.5) ** (1.0 / 0.5)
        interaction = 0.5 * 0.5 * np.sqrt(0.25 * 1.0)
        np.testing.assert_allclose(
            intimate_like_mixing(endmembers, abundances, alpha, beta), generalized + 0.2 * interaction
        )

    def test_indices_are_computed_after_reflectance_mixing(self) -> None:
        reflectance = np.asarray([[0.10, 0.20, 0.30, 0.31, 0.32, 0.33, 0.60, 0.55, 0.40, 0.35]])
        indices = compute_indices(reflectance)
        np.testing.assert_allclose(indices[0, 0], (0.60 - 0.30) / (0.60 + 0.30))
        np.testing.assert_allclose(indices[0, 3], (0.60 - 0.40) / (0.60 + 0.40))

    def test_mixing_ratios_reuse_endmember_combinations_and_abundances(self) -> None:
        reflectance = np.linspace(0.05, 0.75, 6 * 5 * 10).reshape(6, 5, 10)
        labels = np.asarray([
            "herbaceous_vegetation",
            "herbaceous_vegetation",
            "woody_vegetation",
            "woody_vegetation",
            "bare_land",
            "bare_land",
        ])
        class_probabilities = np.asarray([1.0 / 3.0] * 3)
        seeds_a = np.random.SeedSequence(42).spawn(2)
        seeds_b = np.random.SeedSequence(42).spawn(2)
        design_a, mixing_a = np.random.default_rng(seeds_a[0]), np.random.default_rng(seeds_a[1])
        design_b, mixing_b = np.random.default_rng(seeds_b[0]), np.random.default_rng(seeds_b[1])
        probabilities_a = np.asarray([1.0, 0.0, 0.0, 0.0])
        probabilities_b = np.asarray([0.0, 1.0, 0.0, 0.0])
        for _ in range(2):
            batch_a = sample_batch(
                reflectance, labels, 2048, probabilities_a, class_probabilities, 1.0, design_a, mixing_a
            )
            batch_b = sample_batch(
                reflectance, labels, 2048, probabilities_b, class_probabilities, 1.0, design_b, mixing_b
            )
            np.testing.assert_array_equal(batch_a["fractions"], batch_b["fractions"])
            np.testing.assert_array_equal(batch_a["endmember_ids"], batch_b["endmember_ids"])


if __name__ == "__main__":
    unittest.main()
