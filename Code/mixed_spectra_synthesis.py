from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from mixing_models import bilinear_mixing, intimate_like_mixing, linear_mixing, polynomial_nonlinear_mixing
from spectral_indices import BAND_NAMES, FEATURE_NAMES, INDEX_NAMES, compute_indices


CLASS_NAMES = ("herbaceous_vegetation", "woody_vegetation", "bare_land")
MODEL_NAMES = ("LMM", "BMM", "PNMM", "IMM")


def parse_probability_list(text: str, expected: int, label: str) -> np.ndarray:
    values = np.asarray([float(value.strip()) for value in text.split(",")], dtype=np.float64)
    if values.size != expected or np.any(values < 0.0) or not np.isfinite(values).all():
        raise argparse.ArgumentTypeError(f"{label} must contain {expected} finite non-negative values.")
    total = values.sum()
    if total <= 0.0:
        raise argparse.ArgumentTypeError(f"{label} must have a positive sum.")
    return values / total


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize LMM, BMM, PNMM, and IMM spectra.")
    parser.add_argument("--endmembers", required=True, type=Path, help="NPZ written by endmember_extraction.py.")
    parser.add_argument("--output", required=True, type=Path, help="Output HDF5 library.")
    parser.add_argument("--samples", type=int, default=150_000, help="Number of mixed spectra (default: 150000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Generation and HDF5 chunk size.")
    parser.add_argument(
        "--model-probabilities",
        default="0.70,0.10,0.10,0.10",
        help="Probabilities for LMM,BMM,PNMM,IMM (default: 0.70,0.10,0.10,0.10).",
    )
    parser.add_argument(
        "--class-count-probabilities",
        default="0.3333333333,0.3333333333,0.3333333334",
        help="Probabilities of selecting 1, 2, or 3 classes.",
    )
    parser.add_argument(
        "--nonlinearity-scale",
        type=float,
        default=1.0,
        help="Nonlinearity scale s (default: 1.0).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args(argv)


def load_endmembers(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        reflectance = np.asarray(archive["reflectance"], dtype=np.float64)
        labels = np.asarray(archive["labels"]).astype(str)
        stored_bands = tuple(np.asarray(archive["band_names"]).astype(str))
        observation_dates = np.asarray(archive["observation_dates"]).astype(str)
    if reflectance.ndim != 3 or reflectance.shape[-1] != len(BAND_NAMES):
        raise ValueError("Endmember reflectance must have shape (endmember, time, 10 bands).")
    if stored_bands != BAND_NAMES:
        raise ValueError(f"Band order must be {BAND_NAMES}; got {stored_bands}.")
    if observation_dates.ndim != 1 or observation_dates.size != reflectance.shape[1]:
        raise ValueError("Observation dates must match the temporal dimension of the endmember trajectories.")
    missing = [component for component in CLASS_NAMES if not np.any(labels == component)]
    if missing:
        raise ValueError(f"Endmember library is missing classes: {', '.join(missing)}")
    if not np.isfinite(reflectance).all():
        raise ValueError("Endmember reflectance contains NaN or infinite values.")
    return reflectance, labels, observation_dates


def sample_batch(
    reflectance: np.ndarray,
    labels: np.ndarray,
    sample_count: int,
    model_probabilities: np.ndarray,
    class_count_probabilities: np.ndarray,
    scale: float,
    design_rng: np.random.Generator,
    mixing_rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    selected_counts = design_rng.choice(np.arange(1, 4), size=sample_count, p=class_count_probabilities)
    selected_classes = np.zeros((sample_count, len(CLASS_NAMES)), dtype=bool)
    abundances = np.zeros((sample_count, len(CLASS_NAMES)), dtype=np.float64)
    selected_endmember_ids = np.full((sample_count, len(CLASS_NAMES)), -1, dtype=np.int64)

    for row, count in enumerate(selected_counts):
        class_indices = design_rng.choice(len(CLASS_NAMES), size=int(count), replace=False)
        selected_classes[row, class_indices] = True
        raw = design_rng.uniform(0.0, 1.0, size=int(count))
        while raw.sum() == 0.0:
            raw = design_rng.uniform(0.0, 1.0, size=int(count))
        abundances[row, class_indices] = raw / raw.sum()

    sampled = np.zeros(
        (sample_count, len(CLASS_NAMES), reflectance.shape[1], reflectance.shape[2]),
        dtype=np.float64,
    )
    for class_index, component in enumerate(CLASS_NAMES):
        pool = np.flatnonzero(labels == component)
        active = np.flatnonzero(selected_classes[:, class_index])
        choices = design_rng.choice(pool, size=active.size, replace=True)
        selected_endmember_ids[active, class_index] = choices
        sampled[active, class_index] = reflectance[choices]

    model_codes = mixing_rng.choice(len(MODEL_NAMES), size=sample_count, p=model_probabilities)
    mixed = np.empty((sample_count, reflectance.shape[1], reflectance.shape[2]), dtype=np.float64)
    nonlinear_parameters = np.full((sample_count, 2), np.nan, dtype=np.float64)

    for model_code, model_name in enumerate(MODEL_NAMES):
        rows = np.flatnonzero(model_codes == model_code)
        if rows.size == 0:
            continue
        members = sampled[rows]
        fractions = abundances[rows]
        if model_name == "LMM":
            mixed[rows] = linear_mixing(members, fractions)
        elif model_name == "BMM":
            gamma = mixing_rng.uniform(0.0, scale, size=rows.size)
            mixed[rows] = bilinear_mixing(members, fractions, gamma)
            nonlinear_parameters[rows, 0] = gamma
        elif model_name == "PNMM":
            beta2 = mixing_rng.uniform(-scale, scale, size=rows.size)
            mixed[rows] = polynomial_nonlinear_mixing(members, fractions, beta2)
            nonlinear_parameters[rows, 0] = beta2
        else:
            alpha = mixing_rng.uniform(0.3 * scale, 0.7 * scale, size=rows.size)
            beta = mixing_rng.uniform(0.1 * scale, 0.3 * scale, size=rows.size)
            if np.any(alpha <= 0.0):
                raise ValueError("IMM alpha must be positive; choose a positive nonlinearity scale.")
            mixed[rows] = intimate_like_mixing(members, fractions, alpha, beta)
            nonlinear_parameters[rows, 0] = alpha
            nonlinear_parameters[rows, 1] = beta

    indices = compute_indices(mixed)
    features = np.concatenate([mixed, indices], axis=-1)
    return {
        "features": features.astype(np.float32),
        "fractions": abundances.astype(np.float32),
        "model_code": model_codes.astype(np.uint8),
        "selected_class_count": selected_counts.astype(np.uint8),
        "endmember_ids": selected_endmember_ids,
        "nonlinear_parameters": nonlinear_parameters.astype(np.float32),
    }


def write_library(
    endmember_path: Path,
    output_path: Path,
    sample_count: int,
    chunk_size: int,
    seed: int,
    model_probabilities: np.ndarray,
    class_count_probabilities: np.ndarray,
    scale: float,
    overwrite: bool,
) -> None:
    if sample_count <= 0 or chunk_size <= 0:
        raise ValueError("Sample count and chunk size must be positive.")
    if scale <= 0.0:
        raise ValueError("Nonlinearity scale s must be positive.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reflectance, labels, observation_dates = load_endmembers(endmember_path)
    design_seed, mixing_seed = np.random.SeedSequence(seed).spawn(2)
    design_rng = np.random.default_rng(design_seed)
    mixing_rng = np.random.default_rng(mixing_seed)
    time_count = reflectance.shape[1]
    h5_chunk = min(chunk_size, sample_count)

    with h5py.File(output_path, "w") as output:
        output.attrs["description"] = "Knowledge-guided synthetic spectral-temporal library"
        output.attrs["random_seed"] = seed
        output.attrs["nonlinearity_scale_s"] = scale
        output.attrs["model_names"] = json.dumps(MODEL_NAMES)
        output.attrs["model_probabilities"] = model_probabilities
        output.attrs["class_names"] = json.dumps(CLASS_NAMES)
        output.attrs["class_count_probabilities"] = class_count_probabilities
        output.attrs["band_names"] = json.dumps(BAND_NAMES)
        output.attrs["index_names"] = json.dumps(INDEX_NAMES)
        output.attrs["feature_names"] = json.dumps(FEATURE_NAMES)
        output.create_dataset("observation_dates", data=np.asarray(observation_dates, dtype="S8"))

        datasets = {
            "features": output.create_dataset(
                "features", (sample_count, time_count, len(FEATURE_NAMES)), dtype="f4",
                chunks=(h5_chunk, time_count, len(FEATURE_NAMES)), compression="gzip", shuffle=True,
            ),
            "fractions": output.create_dataset("fractions", (sample_count, 3), dtype="f4", chunks=(h5_chunk, 3)),
            "model_code": output.create_dataset("model_code", (sample_count,), dtype="u1", chunks=(h5_chunk,)),
            "selected_class_count": output.create_dataset(
                "selected_class_count", (sample_count,), dtype="u1", chunks=(h5_chunk,)
            ),
            "endmember_ids": output.create_dataset(
                "endmember_ids", (sample_count, 3), dtype="i8", chunks=(h5_chunk, 3)
            ),
            "nonlinear_parameters": output.create_dataset(
                "nonlinear_parameters", (sample_count, 2), dtype="f4", chunks=(h5_chunk, 2)
            ),
        }

        for start in range(0, sample_count, chunk_size):
            stop = min(start + chunk_size, sample_count)
            batch = sample_batch(
                reflectance,
                labels,
                stop - start,
                model_probabilities,
                class_count_probabilities,
                scale,
                design_rng,
                mixing_rng,
            )
            for name, values in batch.items():
                datasets[name][start:stop] = values
            print(f"Generated {stop:,}/{sample_count:,} spectra", end="\r", flush=True)
    print(f"\nSynthetic library written to {output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        model_probabilities = parse_probability_list(arguments.model_probabilities, 4, "model probabilities")
        class_count_probabilities = parse_probability_list(
            arguments.class_count_probabilities, 3, "class-count probabilities"
        )
        write_library(
            arguments.endmembers.resolve(),
            arguments.output.resolve(),
            arguments.samples,
            arguments.chunk_size,
            arguments.seed,
            model_probabilities,
            class_count_probabilities,
            arguments.nonlinearity_scale,
            arguments.overwrite,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
