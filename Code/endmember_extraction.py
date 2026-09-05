import argparse
from datetime import datetime
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, box, mapping

from method_equations import (
    FeatureSpaceError,
    assign_vertices,
    compute_dsint,
    compute_mtv,
    maximum_area_triangle,
    polar_phenological_cycles,
)
from spectral_indices import BAND_NAMES, FEATURE_NAMES, compute_indices


CLASS_NAMES = ("herbaceous_vegetation", "woody_vegetation", "bare_land")
CVM_BANDS = ("B03", "B04", "B08", "B11", "B12")
DATE_PATTERN = re.compile(r"(?<!\d)(\d{8})(?!\d)")


class MethodError(RuntimeError):
    pass


@dataclass
class RasterStack:
    values: np.ndarray
    transform: rasterio.Affine
    crs: Any


@dataclass
class TimeSeriesSource:
    products: dict[str, list[Path]]
    start_date: str | None
    end_date: str | None


@dataclass
class Candidate:
    component: str
    grid_id: str
    x: float
    y: float
    dsint: float
    mtv: float
    cvm: float
    phenological_year_start_degrees: float
    reflectance: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract endmembers from polar-projected temporal indices.")
    parser.add_argument("--config", required=True, type=Path, help="Path to a JSON configuration file.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated output files.")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    required = ("output_dir", "time_series", "target_sequence")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {', '.join(missing)}")
    return config


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def discover(root: Path, pattern: str) -> list[Path]:
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No rasters matched {pattern!r} below {root}.")
    return paths


def parse_band_date(description: str | None, source_name: str, band_index: int) -> str:
    if description is None:
        raise MethodError(f"Band {band_index} in {source_name} has no date description.")
    match = DATE_PATTERN.search(description)
    if match is None:
        raise MethodError(f"Band {band_index} in {source_name} has no YYYYMMDD date token.")
    date_text = match.group(1)
    try:
        datetime.strptime(date_text, "%Y%m%d")
    except ValueError as error:
        raise MethodError(f"Band {band_index} in {source_name} has an invalid date: {date_text}.") from error
    return date_text


def parse_date_sequence(dates: Sequence[str]) -> np.ndarray:
    parsed = [datetime.strptime(str(value), "%Y%m%d").date() for value in dates]
    return np.asarray(parsed, dtype="datetime64[D]")


def configured_time_series(settings: Mapping[str, Any], config_base: Path) -> TimeSeriesSource:
    root = resolve_path(str(settings["root"]), config_base)
    products = {
        "EVI": discover(root, str(settings["evi_glob"])),
        "NDMI": discover(root, str(settings["ndmi_glob"])),
    }
    for band, pattern in settings.get("reflectance_globs", {}).items():
        products[band] = discover(root, str(pattern))
    start_date = settings.get("start_date")
    end_date = settings.get("end_date")
    for value in (start_date, end_date):
        if value is not None and re.fullmatch(r"\d{8}", str(value)) is None:
            raise ValueError("Time-series dates must use YYYYMMDD format.")
    if start_date is not None and end_date is not None and str(start_date) > str(end_date):
        raise ValueError("start_date must not be later than end_date.")
    return TimeSeriesSource(
        products=products,
        start_date=None if start_date is None else str(start_date),
        end_date=None if end_date is None else str(end_date),
    )


def selected_band_indexes(source: rasterio.DatasetReader, series: TimeSeriesSource) -> list[int]:
    indexes = []
    for index, description in enumerate(source.descriptions, start=1):
        date_text = parse_band_date(description, source.name, index)
        if series.start_date is not None and date_text < series.start_date:
            continue
        if series.end_date is not None and date_text > series.end_date:
            continue
        indexes.append(index)
    return indexes


def product_dates(series: TimeSeriesSource, products: Iterable[str]) -> dict[str, list[str]]:
    result = {}
    for product in products:
        dates = []
        for path in series.products[product]:
            with rasterio.open(path) as source:
                indexes = selected_band_indexes(source, series)
                dates.extend(parse_band_date(source.descriptions[index - 1], source.name, index) for index in indexes)
        if not dates:
            limits = f"{series.start_date or '*'}-{series.end_date or '*'}"
            raise MethodError(f"No {product} bands fall within {limits}.")
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise MethodError(f"{product} band dates are not unique and chronological.")
        result[product] = dates
    reference_dates = next(iter(result.values()))
    for product, dates in result.items():
        if dates != reference_dates:
            raise MethodError(f"{product} dates do not match the other products.")
    return result


def validate_source(series: TimeSeriesSource, config: Mapping[str, Any]) -> list[str]:
    products = ("EVI", "NDMI", *BAND_NAMES)
    dates_by_product = product_dates(series, products)
    dates = dates_by_product["EVI"]
    parsed = parse_date_sequence(dates)
    intervals = np.diff(parsed).astype("timedelta64[D]").astype(np.int64)
    expected_step = float(config.get("time_step_days", 8.0))
    median_step = float(np.median(intervals))
    minimum_years = float(config.get("minimum_years", 3.0))
    if series.start_date is not None and series.end_date is not None:
        configured_start = datetime.strptime(series.start_date, "%Y%m%d")
        configured_end = datetime.strptime(series.end_date, "%Y%m%d")
        support_days = (configured_end - configured_start).days + 1
    else:
        support_days = int((parsed[-1] - parsed[0]).astype("timedelta64[D]").astype(np.int64)) + median_step
    span_years = float(support_days) / 365.0
    errors = []
    if span_years < minimum_years:
        errors.append(f"The time series spans {span_years:.3f} years and must cover at least {minimum_years:g} years.")
    if not math.isclose(median_step, expected_step, abs_tol=0.5):
        errors.append(f"The median temporal step is {median_step:g} days instead of {expected_step:g} days.")
    tolerance = int(config.get("coverage_tolerance_days", 10))
    if series.start_date is not None:
        first = datetime.strptime(dates[0], "%Y%m%d")
        requested = datetime.strptime(series.start_date, "%Y%m%d")
        if (first - requested).days > tolerance:
            errors.append("The first observation is later than the configured start_date tolerance.")
    if series.end_date is not None:
        last = datetime.strptime(dates[-1], "%Y%m%d")
        requested = datetime.strptime(series.end_date, "%Y%m%d")
        if (requested - last).days > tolerance:
            errors.append("The last observation is earlier than the configured end_date tolerance.")
    if errors:
        raise MethodError("Input validation failed: " + " ".join(errors))
    print(
        f"Time series: {dates[0]} to {dates[-1]} "
        f"({len(dates)} observations; {span_years:.3f} years; median step {median_step:g} days)"
    )
    return dates


def select_target_sequence(dates: Sequence[str], settings: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    start_text = str(settings["start_date"])
    end_text = str(settings["end_date"])
    if re.fullmatch(r"\d{8}", start_text) is None or re.fullmatch(r"\d{8}", end_text) is None:
        raise ValueError("target_sequence dates must use YYYYMMDD format.")
    start_date = np.datetime64(datetime.strptime(start_text, "%Y%m%d").date())
    end_date = np.datetime64(datetime.strptime(end_text, "%Y%m%d").date())
    if start_date > end_date:
        raise ValueError("target_sequence start_date must not be later than end_date.")
    date_values = np.asarray(dates, dtype="datetime64[D]")
    selected = (date_values >= start_date) & (date_values <= end_date)
    if not np.any(selected):
        raise MethodError("No observations fall within target_sequence.")
    return date_values, selected


def scale_and_filter(
    values: np.ndarray,
    nodata_value: float | None,
    scale_factor: float,
    auto_scale: bool,
    valid_range: tuple[float, float],
) -> np.ndarray:
    data = values.astype(np.float64, copy=False)
    if nodata_value is not None:
        data[data == nodata_value] = np.nan
    finite = data[np.isfinite(data)]
    if auto_scale and finite.size and np.nanmax(np.abs(finite)) > 2.0:
        data = data / scale_factor
    low, high = valid_range
    data[(data < low) | (data > high)] = np.nan
    return data


def read_stack(
    paths: Sequence[Path],
    series: TimeSeriesSource,
    geometry: Any,
    expected_crs: Any,
    nodata_value: float | None,
    scale_factor: float,
    auto_scale: bool,
    valid_range: tuple[float, float],
    all_touched: bool,
    reference: RasterStack | None = None,
) -> RasterStack:
    arrays = []
    output_transform = None
    output_crs = None
    output_shape = None
    for path in paths:
        with rasterio.open(path) as source:
            if source.crs != expected_crs:
                raise MethodError(f"CRS mismatch in {path}: {source.crs} != {expected_crs}.")
            indexes = selected_band_indexes(source, series)
            if not indexes:
                continue
            cropped, transform = mask(
                source,
                [mapping(geometry)],
                indexes=indexes,
                crop=True,
                all_touched=all_touched,
                filled=False,
            )
            data = scale_and_filter(
                np.ma.filled(cropped.astype(np.float64), np.nan),
                nodata_value,
                scale_factor,
                auto_scale,
                valid_range,
            )
            if output_shape is None:
                output_shape = data.shape[1:]
                output_transform = transform
                output_crs = source.crs
            elif data.shape[1:] != output_shape or not transform.almost_equals(output_transform):
                raise MethodError(f"Raster grid mismatch within the stack at {path}.")
            arrays.extend(data[index] for index in range(data.shape[0]))
    if not arrays:
        raise MethodError("No raster bands fall within the configured time range.")
    stack = RasterStack(np.stack(arrays, axis=0), output_transform, output_crs)
    if reference is not None:
        if stack.values.shape != reference.values.shape or not stack.transform.almost_equals(reference.transform):
            raise MethodError("All products must share the same dates and pixel grid within each subregion.")
    return stack


def read_products_for_grid(
    series: TimeSeriesSource,
    geometry: Any,
    raster_crs: Any,
    config: Mapping[str, Any],
    products: Iterable[str],
    reference: RasterStack | None = None,
) -> dict[str, RasterStack]:
    ranges = config.get("valid_ranges", {})
    result = {}
    current_reference = reference
    for product in products:
        range_key = product if product in ("EVI", "NDMI") else "reflectance"
        valid_range = tuple(ranges.get(range_key, [-1.0, 1.0]))
        stack = read_stack(
            series.products[product],
            series,
            geometry,
            raster_crs,
            config.get("nodata_value", -9999),
            float(config.get("scale_factor", 10000.0)),
            bool(config.get("auto_scale", True)),
            valid_range,
            bool(config.get("all_touched", True)),
            current_reference,
        )
        result[product] = stack
        if current_reference is None:
            current_reference = stack
    return result


def read_candidate_reflectance(
    series: TimeSeriesSource,
    geometry: Any,
    raster_crs: Any,
    config: Mapping[str, Any],
    reference: RasterStack,
    positions: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    rows = np.asarray([position[0] for position in positions], dtype=np.int64)
    columns = np.asarray([position[1] for position in positions], dtype=np.int64)
    result = {}
    for band in BAND_NAMES:
        stack = read_products_for_grid(series, geometry, raster_crs, config, (band,), reference)[band]
        result[band] = stack.values[:, rows, columns]
    return result


def choose_candidate_pixels(
    dsint: np.ndarray,
    mtv: np.ndarray,
    count_per_class: int,
    trim_percentiles: tuple[float, float],
) -> tuple[dict[str, list[tuple[int, int, float, float]]], np.ndarray, dict[str, np.ndarray]]:
    valid = np.isfinite(dsint) & np.isfinite(mtv)
    if valid.sum() < 3:
        raise MethodError("Fewer than three pixels have valid DSINT and MTV values.")
    low, high = trim_percentiles
    dsint_limits = np.nanpercentile(dsint[valid], [low, high])
    mtv_limits = np.nanpercentile(mtv[valid], [low, high])
    retained = valid & (dsint >= dsint_limits[0]) & (dsint <= dsint_limits[1])
    retained &= (mtv >= mtv_limits[0]) & (mtv <= mtv_limits[1])
    row_indices, column_indices = np.where(retained)
    points = np.column_stack([dsint[retained], mtv[retained]])
    try:
        triangle, _ = maximum_area_triangle(points)
    except FeatureSpaceError as error:
        raise MethodError(str(error)) from error
    assigned = assign_vertices(triangle)
    dsint_50, dsint_80 = np.percentile(points[:, 0], [50.0, 80.0])
    mtv_50, mtv_80 = np.percentile(points[:, 1], [50.0, 80.0])
    eligible = {
        "woody_vegetation": (points[:, 0] > dsint_80) & (points[:, 1] < mtv_50),
        "herbaceous_vegetation": (points[:, 1] > mtv_80) & (points[:, 0] < dsint_50),
        "bare_land": (points[:, 0] < dsint_50) & (points[:, 1] < mtv_50),
    }
    selected = {}
    for component in CLASS_NAMES:
        pool = np.flatnonzero(eligible[component])
        if pool.size < count_per_class:
            raise MethodError(
                f"Only {pool.size} {component} pixels satisfy the percentile constraints; "
                f"{count_per_class} are required."
            )
        distances = np.linalg.norm(points[pool] - assigned[component][None, :], axis=1)
        nearest = pool[np.argsort(distances, kind="stable")[:count_per_class]]
        selected[component] = [
            (
                int(row_indices[index]),
                int(column_indices[index]),
                float(points[index, 0]),
                float(points[index, 1]),
            )
            for index in nearest
        ]
    return selected, points, assigned


def chronological_trajectory(
    products: Mapping[str, np.ndarray],
    candidate_index: int,
    target_sequence_mask: np.ndarray,
) -> np.ndarray:
    trajectories = [products[band][target_sequence_mask, candidate_index] for band in BAND_NAMES]
    return np.stack(trajectories, axis=-1)


def cycle_median_vector(
    products: Mapping[str, np.ndarray],
    candidate_index: int,
    cycle_mask: np.ndarray,
    row: int,
    column: int,
) -> np.ndarray:
    values = []
    pixel_mask = cycle_mask[:, row, column]
    for band in CVM_BANDS:
        series = products[band][:, candidate_index]
        values.append(np.nanmedian(np.where(pixel_mask, series, np.nan)))
    return np.asarray(values)


def change_vector_magnitude(
    products: Mapping[str, np.ndarray],
    candidate_index: int,
    cycle_model: Mapping[str, np.ndarray],
    row: int,
    column: int,
) -> float:
    target = cycle_median_vector(products, candidate_index, cycle_model["target_mask"], row, column)
    previous = [
        cycle_median_vector(products, candidate_index, cycle_model["previous_mask_1"], row, column),
        cycle_median_vector(products, candidate_index, cycle_model["previous_mask_2"], row, column),
    ]
    if not np.all(np.isfinite(target)) or not all(np.all(np.isfinite(item)) for item in previous):
        return math.nan
    return float(max(np.linalg.norm(target - item) for item in previous))


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 2:
        return -1.0
    left = first[valid]
    right = second[valid]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 1.0 if np.allclose(left, right, equal_nan=True) else -1.0
    return float(np.corrcoef(left, right)[0, 1])


def screen_candidates(
    candidates: Sequence[Candidate],
    cvm_percentile: float,
    neighbor_distance: float,
    trajectory_correlation: float,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    audit = []
    stable = []
    thresholds = {}
    for component in CLASS_NAMES:
        values = np.asarray([item.cvm for item in candidates if item.component == component and np.isfinite(item.cvm)])
        if values.size == 0:
            raise MethodError(f"No finite CVM values are available for {component} candidates.")
        thresholds[component] = float(np.percentile(values, cvm_percentile))
    for item in candidates:
        accepted = np.isfinite(item.cvm) and item.cvm <= thresholds[item.component]
        audit.append(
            {
                "grid_id": item.grid_id,
                "component": item.component,
                "x": item.x,
                "y": item.y,
                "dsint": item.dsint,
                "mtv": item.mtv,
                "cvm": item.cvm,
                "phenological_year_start_degrees": item.phenological_year_start_degrees,
                "cvm_threshold": thresholds[item.component],
                "retained_after_cvm": accepted,
                "retained_after_deduplication": False,
            }
        )
        if accepted:
            stable.append(item)
    retained = []
    for component in CLASS_NAMES:
        ordered = sorted((item for item in stable if item.component == component), key=lambda item: item.cvm)
        for item in ordered:
            redundant = False
            for kept in retained:
                if kept.component != component:
                    continue
                distance = math.hypot(item.x - kept.x, item.y - kept.y)
                similarity = correlation(item.reflectance.ravel(), kept.reflectance.ravel())
                if distance <= neighbor_distance and similarity >= trajectory_correlation:
                    redundant = True
                    break
            if not redundant:
                retained.append(item)
                for record in audit:
                    same_location = record["x"] == item.x and record["y"] == item.y
                    if record["grid_id"] == item.grid_id and record["component"] == component and same_location:
                        record["retained_after_deduplication"] = True
                        break
    return retained, audit


def save_feature_plot(
    points: np.ndarray,
    assigned: Mapping[str, np.ndarray],
    selected: Mapping[str, Sequence[tuple[int, int, float, float]]],
    grid_id: str,
    output_dir: Path,
) -> None:
    colors = {
        "woody_vegetation": "#18864b",
        "herbaceous_vegetation": "#2868c7",
        "bare_land": "#b65a32",
    }
    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    axis.scatter(points[:, 0], points[:, 1], s=3, color="0.65", alpha=0.35, linewidths=0)
    for component in CLASS_NAMES:
        vertex = assigned[component]
        candidates = np.asarray([[item[2], item[3]] for item in selected[component]])
        axis.scatter(vertex[0], vertex[1], marker="*", s=180, color=colors[component], label=f"{component} vertex")
        axis.scatter(candidates[:, 0], candidates[:, 1], s=44, facecolors="none", edgecolors=colors[component])
    axis.set_xlabel("Dormant Season Integral (DSINT)")
    axis.set_ylabel("Moisture Temporal Variability (MTV)")
    axis.set_title(f"Adaptive physio-phenological feature space: {grid_id}")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / f"{grid_id}_feature_space.png", dpi=200)
    plt.close(figure)


def regular_grid(bounds: rasterio.coords.BoundingBox, crs: Any, grid_size: float) -> gpd.GeoDataFrame:
    if grid_size <= 0.0:
        raise ValueError("grid_size must be positive.")
    records = []
    row = 0
    top = bounds.top
    while top > bounds.bottom:
        bottom = max(top - grid_size, bounds.bottom)
        column = 0
        left = bounds.left
        while left < bounds.right:
            right = min(left + grid_size, bounds.right)
            records.append({"grid_id": f"g{row}_{column}", "geometry": box(left, bottom, right, top)})
            left += grid_size
            column += 1
        top -= grid_size
        row += 1
    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)


def run(config: Mapping[str, Any], config_base: Path, overwrite: bool, validate_only: bool) -> None:
    output_dir = resolve_path(str(config["output_dir"]), config_base)
    series = configured_time_series(config["time_series"], config_base)
    required_products = {"EVI", "NDMI", *BAND_NAMES}
    missing = sorted(required_products - set(series.products))
    if missing:
        raise MethodError(f"The time series is missing products: {', '.join(missing)}")
    dates = validate_source(series, config)
    date_values, target_sequence_mask = select_target_sequence(dates, config["target_sequence"])
    if validate_only:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    final_npz = output_dir / "endmember_library.npz"
    if final_npz.exists() and not overwrite:
        raise FileExistsError(f"{final_npz} exists. Use --overwrite to replace generated outputs.")
    with rasterio.open(series.products["EVI"][0]) as reference_raster:
        raster_crs = reference_raster.crs
        raster_bounds = reference_raster.bounds
    grid_path = config.get("grid_shapefile")
    if grid_path:
        grids = gpd.read_file(resolve_path(str(grid_path), config_base))
        if grids.crs is None:
            raise MethodError("The grid layer has no CRS.")
        grids = grids.to_crs(raster_crs)
        id_field = str(config.get("grid_id_field", "grid_id"))
    else:
        id_field = "grid_id"
        grids = regular_grid(raster_bounds, raster_crs, float(config.get("grid_size", 5000.0)))
        grids.to_file(output_dir / "processing_grid.geojson", driver="GeoJSON")
    if id_field not in grids.columns:
        raise KeyError(f"Grid ID field {id_field!r} was not found.")
    requested_ids = config.get("grid_ids")
    if requested_ids is not None:
        grids = grids[grids[id_field].astype(str).isin({str(value) for value in requested_ids})]
    if grids.empty:
        raise MethodError("No subregions were selected.")
    time_step_days = float(config.get("time_step_days", 8.0))
    raw_candidates = []
    for position, (_, grid) in enumerate(grids.iterrows(), start=1):
        grid_id = str(grid[id_field])
        print(f"[{position}/{len(grids)}] Processing subregion {grid_id}")
        base_products = read_products_for_grid(series, grid.geometry, raster_crs, config, ("EVI", "NDMI"))
        evi = base_products["EVI"]
        ndmi = base_products["NDMI"]
        cycle_model = polar_phenological_cycles(
            evi.values,
            date_values,
            minimum_years=float(config.get("minimum_years", 3.0)),
            minimum_observations_per_cycle=int(config.get("minimum_observations_per_cycle", 30)),
            dormant_search_days=float(config.get("dormant_search_days", 91.0)),
            minimum_observations_per_boundary=int(config.get("minimum_observations_per_boundary", 3)),
        )
        target_evi = np.where(cycle_model["target_mask"], evi.values, np.nan)
        target_ndmi = np.where(cycle_model["target_mask"], ndmi.values, np.nan)
        dsint = compute_dsint(target_evi, time_step_days)
        mtv = compute_mtv(target_ndmi)
        selected, points, assigned = choose_candidate_pixels(
            dsint,
            mtv,
            int(config.get("candidate_count_per_class", 3)),
            tuple(config.get("trim_percentiles", [0.5, 99.5])),
        )
        candidate_records = [
            (component, row, column, dsint_value, mtv_value)
            for component, pixels in selected.items()
            for row, column, dsint_value, mtv_value in pixels
        ]
        positions = [(item[1], item[2]) for item in candidate_records]
        reflectance = read_candidate_reflectance(
            series,
            grid.geometry,
            raster_crs,
            config,
            evi,
            positions,
        )
        for candidate_index, record in enumerate(candidate_records):
            component, row, column, dsint_value, mtv_value = record
            x, y = rasterio.transform.xy(evi.transform, row, column, offset="center")
            start_degrees = math.degrees(cycle_model["phenological_year_start_angle"][row, column])
            raw_candidates.append(
                Candidate(
                    component=component,
                    grid_id=grid_id,
                    x=float(x),
                    y=float(y),
                    dsint=dsint_value,
                    mtv=mtv_value,
                    cvm=change_vector_magnitude(reflectance, candidate_index, cycle_model, row, column),
                    phenological_year_start_degrees=float(start_degrees),
                    reflectance=chronological_trajectory(
                        reflectance,
                        candidate_index,
                        target_sequence_mask,
                    ),
                )
            )
        save_feature_plot(points, assigned, selected, grid_id, output_dir)
    deduplication = config.get("deduplication", {})
    retained, audit = screen_candidates(
        raw_candidates,
        float(config.get("cvm_percentile", 95.0)),
        float(deduplication.get("neighbor_distance", 30.0)),
        float(deduplication.get("trajectory_correlation", 0.995)),
    )
    if not retained:
        raise MethodError("All candidates were removed during quality screening.")
    reflectance = np.stack([item.reflectance for item in retained]).astype(np.float32)
    indices = compute_indices(reflectance).astype(np.float32)
    features = np.concatenate([reflectance, indices], axis=-1)
    metadata = pd.DataFrame(
        [
            {
                "endmember_id": index,
                "component": item.component,
                "grid_id": item.grid_id,
                "x": item.x,
                "y": item.y,
                "dsint": item.dsint,
                "mtv": item.mtv,
                "cvm": item.cvm,
                "phenological_year_start_degrees": item.phenological_year_start_degrees,
            }
            for index, item in enumerate(retained)
        ]
    )
    metadata.to_csv(output_dir / "endmembers.csv", index=False)
    pd.DataFrame(audit).to_csv(output_dir / "candidate_screening_audit.csv", index=False)
    geometry = [Point(item.x, item.y) for item in retained]
    gpd.GeoDataFrame(metadata.copy(), geometry=geometry, crs=raster_crs).to_file(
        output_dir / "endmembers.geojson", driver="GeoJSON"
    )
    np.savez_compressed(
        final_npz,
        reflectance=reflectance,
        indices=indices,
        features=features,
        labels=metadata["component"].to_numpy(dtype="U24"),
        endmember_ids=metadata["endmember_id"].to_numpy(np.int64),
        grid_ids=metadata["grid_id"].to_numpy(dtype="U64"),
        coordinates=metadata[["x", "y"]].to_numpy(np.float64),
        phenological_year_start_degrees=metadata["phenological_year_start_degrees"].to_numpy(np.float64),
        observation_dates=np.asarray(dates, dtype="U8")[target_sequence_mask],
        band_names=np.asarray(BAND_NAMES, dtype="U8"),
        index_names=np.asarray(FEATURE_NAMES[len(BAND_NAMES) :], dtype="U8"),
        crs_wkt=np.asarray(raster_crs.to_wkt()),
    )
    print(f"Retained {len(retained)} endmembers. Library written to {final_npz}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        config_path = arguments.config.resolve()
        run(load_config(config_path), config_path.parent, arguments.overwrite, arguments.validate_only)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
