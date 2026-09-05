import itertools
import math

import numpy as np


class FeatureSpaceError(RuntimeError):
    pass


def temporal_polar_angles(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(dates, dtype="datetime64[D]")
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Dates must be a one-dimensional sequence with at least two observations.")
    ordinal_days = values.astype(np.int64).astype(np.float64)
    if np.any(np.diff(ordinal_days) <= 0.0):
        raise ValueError("Dates must be unique and chronological.")
    year_values = values.astype("datetime64[Y]")
    year_starts = year_values.astype("datetime64[D]")
    next_year_starts = (year_values + np.timedelta64(1, "Y")).astype("datetime64[D]")
    elapsed_days = (values - year_starts).astype("timedelta64[D]").astype(np.float64)
    days_in_year = (next_year_starts - year_starts).astype("timedelta64[D]").astype(np.float64)
    year_fraction = elapsed_days / days_in_year
    absolute_cycles = year_values.astype(np.int64).astype(np.float64) + year_fraction
    angles = 2.0 * np.pi * year_fraction
    return angles, absolute_cycles


def project_temporal_index(
    index_values: np.ndarray,
    dates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if index_values.ndim != 3:
        raise ValueError("The temporal index must have dimensions (time, rows, columns).")
    angles, _ = temporal_polar_angles(dates)
    if index_values.shape[0] != angles.size:
        raise ValueError("The temporal index and date sequence have different observation counts.")
    phase_shape = (angles.size,) + (1,) * (index_values.ndim - 1)
    return index_values * np.cos(angles.reshape(phase_shape)), index_values * np.sin(angles.reshape(phase_shape))


def polar_phenological_cycles(
    evi: np.ndarray,
    dates: np.ndarray,
    minimum_years: float = 3.0,
    minimum_observations_per_cycle: int = 30,
    dormant_search_days: float = 91.0,
    minimum_observations_per_boundary: int = 3,
) -> dict[str, np.ndarray]:
    angles, absolute_cycles = temporal_polar_angles(dates)
    median_step = float(np.median(np.diff(absolute_cycles)))
    if absolute_cycles[-1] - absolute_cycles[0] + median_step < minimum_years:
        raise ValueError(f"The time series must span at least {minimum_years:g} years.")
    projected_x, projected_y = project_temporal_index(evi, dates)
    valid = np.isfinite(evi)
    counts = np.sum(valid, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = np.sum(np.where(valid, projected_x, 0.0), axis=0) / counts
        mean_y = np.sum(np.where(valid, projected_y, 0.0), axis=0) / counts
    mean_vector_angle = np.mod(np.arctan2(mean_y, mean_x), 2.0 * np.pi)
    long_term_start_angle = np.mod(mean_vector_angle + np.pi, 2.0 * np.pi)
    long_term_start_fraction = long_term_start_angle / (2.0 * np.pi)
    safe_start_fraction = np.where(np.isfinite(long_term_start_fraction), long_term_start_fraction, 0.0)
    search_fraction = dormant_search_days / 365.0
    last_recurrence = np.floor(absolute_cycles[-1] - safe_start_fraction + search_fraction).astype(np.int64)
    boundaries = []
    boundary_validity = []
    time_index = np.arange(evi.shape[0], dtype=np.int64)[:, None, None]
    for offset in (-3, -2, -1, 0):
        nominal_boundary = last_recurrence + offset + safe_start_fraction
        search_mask = (
            np.abs(absolute_cycles[:, None, None] - nominal_boundary[None, :, :])
            <= search_fraction
        )
        search_valid = search_mask & valid
        search_count = np.sum(search_valid, axis=0)
        candidates = np.where(search_valid, evi, np.inf)
        minimum_index = np.argmin(candidates, axis=0)
        dynamic_boundary = absolute_cycles[minimum_index]
        minimum_value = np.take_along_axis(candidates, minimum_index[None, :, :], axis=0)[0]
        has_valid_before = np.any(search_valid & (time_index < minimum_index[None, :, :]), axis=0)
        has_valid_after = np.any(search_valid & (time_index > minimum_index[None, :, :]), axis=0)
        boundaries.append(dynamic_boundary)
        boundary_validity.append(
            (search_count >= minimum_observations_per_boundary)
            & np.isfinite(minimum_value)
            & has_valid_before
            & has_valid_after
        )
    previous_start_2, previous_start_1, target_start, target_end = boundaries
    previous_mask_2 = (
        (absolute_cycles[:, None, None] >= previous_start_2[None, :, :])
        & (absolute_cycles[:, None, None] < previous_start_1[None, :, :])
    )
    previous_mask_1 = (
        (absolute_cycles[:, None, None] >= previous_start_1[None, :, :])
        & (absolute_cycles[:, None, None] < target_start[None, :, :])
    )
    target_mask = (
        (absolute_cycles[:, None, None] >= target_start[None, :, :])
        & (absolute_cycles[:, None, None] < target_end[None, :, :])
    )
    vector_magnitude = np.hypot(mean_x, mean_y)
    complete = (previous_start_2 >= absolute_cycles[0]) & (target_end <= absolute_cycles[-1])
    complete &= previous_start_2 < previous_start_1
    complete &= previous_start_1 < target_start
    complete &= target_start < target_end
    complete &= np.logical_and.reduce(boundary_validity)
    complete &= np.isfinite(vector_magnitude) & (vector_magnitude > np.finfo(np.float64).eps)
    complete &= np.sum(target_mask & valid, axis=0) >= minimum_observations_per_cycle
    complete &= np.sum(previous_mask_1 & valid, axis=0) >= minimum_observations_per_cycle
    complete &= np.sum(previous_mask_2 & valid, axis=0) >= minimum_observations_per_cycle
    target_mask &= complete[None, :, :]
    previous_mask_1 &= complete[None, :, :]
    previous_mask_2 &= complete[None, :, :]
    return {
        "angles": angles,
        "absolute_cycles": absolute_cycles,
        "mean_vector_angle": mean_vector_angle,
        "long_term_phenological_year_start_angle": long_term_start_angle,
        "phenological_year_start_angle": np.mod(target_start, 1.0) * (2.0 * np.pi),
        "previous_cycle_2_start": previous_start_2,
        "previous_cycle_1_start": previous_start_1,
        "target_cycle_start": target_start,
        "target_cycle_end": target_end,
        "target_mask": target_mask,
        "previous_mask_1": previous_mask_1,
        "previous_mask_2": previous_mask_2,
        "valid_three_cycle_support": complete,
    }


def compute_dsint(evi: np.ndarray, time_step_days: float) -> np.ndarray:
    if evi.ndim != 3:
        raise ValueError("EVI must have dimensions (time, rows, columns).")
    time_count, rows, columns = evi.shape
    flat = evi.reshape(time_count, -1)
    valid = np.isfinite(flat)
    series = np.where(valid, flat, 0.0)
    total = np.sum(series, axis=0)
    cumulative = np.cumsum(series, axis=0)
    lower_crossing = cumulative >= (0.15 * total)[None, :]
    upper_crossing = cumulative >= (0.80 * total)[None, :]
    usable = (valid.sum(axis=0) > 0) & (total > 0.0) & lower_crossing.any(axis=0) & upper_crossing.any(axis=0)
    start = np.argmax(lower_crossing, axis=0)
    end = np.argmax(upper_crossing, axis=0)
    time_index = np.arange(time_count)[:, None]
    growing = (time_index >= start[None, :]) & (time_index <= end[None, :])
    growing_integral = np.sum(np.where(growing & valid, series, 0.0), axis=0) * time_step_days
    dsint = total * time_step_days - growing_integral
    dsint[~usable] = np.nan
    return dsint.reshape(rows, columns)


def compute_mtv(ndmi: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        result = np.nanstd(ndmi, axis=0, ddof=0)
    result[np.sum(np.isfinite(ndmi), axis=0) == 0] = np.nan
    return result


def maximum_area_triangle(points: np.ndarray) -> tuple[np.ndarray, float]:
    from scipy.spatial import ConvexHull, QhullError

    if points.shape[0] < 3:
        raise FeatureSpaceError("At least three valid feature-space pixels are required.")
    try:
        hull = ConvexHull(points)
    except QhullError as error:
        raise FeatureSpaceError("The feature-space observations do not form a two-dimensional hull.") from error
    vertices = points[hull.vertices]
    best_triangle = None
    best_area = -math.inf
    for first, second, third in itertools.combinations(vertices, 3):
        edge_a = second - first
        edge_b = third - first
        area = 0.5 * abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0])
        if area > best_area:
            best_area = float(area)
            best_triangle = np.stack([first, second, third])
    if best_triangle is None:
        raise FeatureSpaceError("No convex-hull triangle could be constructed.")
    return best_triangle, best_area


def assign_vertices(triangle: np.ndarray) -> dict[str, np.ndarray]:
    bare_index = int(np.argmin(np.linalg.norm(triangle, axis=1)))
    remaining = [index for index in range(3) if index != bare_index]
    woody_index = max(remaining, key=lambda index: triangle[index, 0])
    herbaceous_index = next(index for index in remaining if index != woody_index)
    return {
        "woody_vegetation": triangle[woody_index],
        "herbaceous_vegetation": triangle[herbaceous_index],
        "bare_land": triangle[bare_index],
    }
