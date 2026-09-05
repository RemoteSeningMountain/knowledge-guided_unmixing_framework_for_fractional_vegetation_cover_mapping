from __future__ import annotations

import numpy as np


def linear_mixing(endmembers: np.ndarray, abundances: np.ndarray) -> np.ndarray:
    return np.sum(abundances[:, :, None, None] * endmembers, axis=1)


def bilinear_mixing(
    endmembers: np.ndarray,
    abundances: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    mixed = linear_mixing(endmembers, abundances)
    interaction = np.zeros_like(mixed)
    class_count = endmembers.shape[1]
    for first in range(class_count):
        for second in range(first + 1, class_count):
            weight = abundances[:, first] * abundances[:, second]
            interaction += weight[:, None, None] * endmembers[:, first] * endmembers[:, second]
    return mixed + gamma[:, None, None] * interaction


def polynomial_nonlinear_mixing(
    endmembers: np.ndarray,
    abundances: np.ndarray,
    beta2: np.ndarray,
) -> np.ndarray:
    mixed = linear_mixing(endmembers, abundances)
    return mixed + beta2[:, None, None] * np.square(mixed)


def intimate_like_mixing(
    endmembers: np.ndarray,
    abundances: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    if np.any(endmembers < 0.0):
        raise ValueError("IMM requires non-negative surface reflectance for fractional powers.")
    generalized = np.sum(
        abundances[:, :, None, None] * np.power(endmembers, alpha[:, None, None, None]),
        axis=1,
    )
    generalized = np.power(generalized, 1.0 / alpha[:, None, None])
    interaction = np.zeros_like(generalized)
    class_count = endmembers.shape[1]
    for first in range(class_count):
        for second in range(first + 1, class_count):
            weight = abundances[:, first] * abundances[:, second]
            interaction += weight[:, None, None] * np.sqrt(endmembers[:, first] * endmembers[:, second])
    return generalized + beta[:, None, None] * interaction
