from __future__ import annotations

import numpy as np


BAND_NAMES = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")
INDEX_NAMES = ("NDVI", "EVI", "NDWI", "NDMI", "MNDWI", "NBR", "TCB", "TCG", "TCW")
FEATURE_NAMES = BAND_NAMES + INDEX_NAMES

TASSELED_CAP = np.asarray(
    [
        [0.0822, 0.1360, 0.2611, 0.2964, 0.3338, 0.3877, 0.3895, 0.4750, 0.3882, 0.1366],
        [-0.1128, -0.1680, -0.3480, -0.3303, -0.2852, -0.2438, -0.1932, 0.0442, 0.2696, 0.6750],
        [0.1363, 0.2802, 0.3072, 0.5288, 0.1379, -0.0001, -0.0807, -0.1389, -0.4064, -0.5602],
    ],
    dtype=np.float64,
)


def _normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    return np.divide(
        a - b,
        denominator,
        out=np.full_like(denominator, np.nan, dtype=np.float64),
        where=np.abs(denominator) > np.finfo(np.float64).eps,
    )


def compute_indices(reflectance: np.ndarray) -> np.ndarray:
    values = np.asarray(reflectance, dtype=np.float64)
    if values.shape[-1] != len(BAND_NAMES):
        raise ValueError(f"Expected {len(BAND_NAMES)} reflectance bands; got {values.shape[-1]}.")

    blue = values[..., 0]
    green = values[..., 1]
    red = values[..., 2]
    nir = values[..., 6]
    swir1 = values[..., 8]
    swir2 = values[..., 9]

    ndvi = _normalized_difference(nir, red)
    evi_denominator = nir + 6.0 * red - 7.5 * blue + 1.0
    evi = np.divide(
        2.5 * (nir - red),
        evi_denominator,
        out=np.full_like(evi_denominator, np.nan, dtype=np.float64),
        where=np.abs(evi_denominator) > np.finfo(np.float64).eps,
    )
    ndwi = _normalized_difference(green, nir)
    ndmi = _normalized_difference(nir, swir1)
    mndwi = _normalized_difference(green, swir1)
    nbr = _normalized_difference(nir, swir2)
    tasseled_cap = values @ TASSELED_CAP.T

    return np.concatenate(
        [
            ndvi[..., None],
            evi[..., None],
            ndwi[..., None],
            ndmi[..., None],
            mndwi[..., None],
            nbr[..., None],
            tasseled_cap,
        ],
        axis=-1,
    )
