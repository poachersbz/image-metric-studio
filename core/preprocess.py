"""
Preprocessing, compatibility checking, resampling, normalization, and masking.

Provides tools to check and fix shape or range mismatches, and segment brain regions
for MRI-centric comparisons. No UI dependencies here.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndimage
from skimage.filters import threshold_otsu

from . import config
from .loaders import ImageMeta


def check_compatibility(meta_a: ImageMeta, meta_b: ImageMeta) -> dict:
    """
    Check if two volumes/images can be compared directly, and return flags + messages.
    """
    ndim_match = meta_a.ndim == meta_b.ndim
    shape_match = meta_a.shape == meta_b.shape
    
    # Calculate range match
    range_a = meta_a.intensity_max - meta_a.intensity_min
    range_b = meta_b.intensity_max - meta_b.intensity_min
    
    max_range = max(range_a, range_b)
    if max_range < 1e-9:
        range_match = True
    else:
        # Check if the difference in intensity range is within RANGE_MATCH_TOLERANCE fraction of the maximum range
        diff_fraction = abs(range_a - range_b) / max_range
        range_match = diff_fraction <= config.RANGE_MATCH_TOLERANCE

    messages = []
    if not ndim_match:
        messages.append(
            f"Dimensionality mismatch: Reference is {meta_a.ndim}D, "
            f"Test is {meta_b.ndim}D."
        )
    else:
        if not shape_match:
            messages.append(
                f"Shape mismatch: Reference shape is {meta_a.shape}, "
                f"Test shape is {meta_b.shape}."
            )
            
    if not range_match:
        messages.append(
            f"Intensity range mismatch: Reference range [{meta_a.intensity_min:.2f}, {meta_a.intensity_max:.2f}], "
            f"Test range [{meta_b.intensity_min:.2f}, {meta_b.intensity_max:.2f}]."
        )

    return {
        "ndim_match": ndim_match,
        "shape_match": shape_match,
        "range_match": range_match,
        "messages": messages
    }


def _validate_array(arr: np.ndarray, name: str) -> None:
    """Reject empty or non-finite input instead of silently producing garbage
    (e.g. NaN bounds collapsing a whole array to zeros)."""
    if arr is None or arr.size == 0:
        raise ValueError(f"{name} is empty: cannot normalize.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values: cannot normalize.")


def _bounds(arr: np.ndarray) -> tuple[float, float]:
    """Range used for normalization: raw min/max, or a percentile clip (robust
    to the handful of outlier-bright voxels -- e.g. fat/vessel spots -- that
    otherwise stretch the whole normalized range and wash out real tissue
    contrast, same issue as DICOM display windowing)."""
    _validate_array(arr, "array")
    if config.NORMALIZE_METHOD == "percentile":
        lo, hi = np.percentile(arr, [config.PERCENTILE_LOW, config.PERCENTILE_HIGH])
        return float(lo), float(hi)
    return float(arr.min()), float(arr.max())


def normalize_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize both arrays to [0, 1] depending on config.MATCH_MODE (independent
    vs shared range) and config.NORMALIZE_METHOD (percentile-clip vs raw min-max).
    Raises ValueError on empty or non-finite input; never silently zeroes an
    entire input array.
    """
    _validate_array(a, "a")
    _validate_array(b, "b")
    a_min, a_max = _bounds(a)
    b_min, b_max = _bounds(b)

    if config.MATCH_MODE == "independent":
        # Normalize independently
        denom_a = a_max - a_min
        if denom_a > 1e-9:
            a_norm = np.clip((a - a_min) / denom_a, 0.0, 1.0)
        else:
            a_norm = np.zeros_like(a)

        denom_b = b_max - b_min
        if denom_b > 1e-9:
            b_norm = np.clip((b - b_min) / denom_b, 0.0, 1.0)
        else:
            b_norm = np.zeros_like(b)

        return a_norm, b_norm

    elif config.MATCH_MODE == "shared":
        # Normalize to a shared range
        shared_min = min(a_min, b_min)
        shared_max = max(a_max, b_max)
        denom = shared_max - shared_min

        if denom > 1e-9:
            a_norm = np.clip((a - shared_min) / denom, 0.0, 1.0)
            b_norm = np.clip((b - shared_min) / denom, 0.0, 1.0)
        else:
            a_norm = np.zeros_like(a)
            b_norm = np.zeros_like(b)

        return a_norm, b_norm

    else:
        raise ValueError(f"Unknown MATCH_MODE: {config.MATCH_MODE}")


def resample_to_match(moving: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Resample with linear interpolation between the first and last voxel centers.
    Keeping both endpoints aligned avoids sampling beyond the source image.
    """
    if moving.shape == target_shape:
        return moving
        
    factors = [t / s for t, s in zip(target_shape, moving.shape)]
    return ndimage.zoom(moving, factors, order=1).astype(moving.dtype)


def brain_mask(slice_2d: np.ndarray) -> np.ndarray:
    """
    Generate a boolean mask for the brain region in a 2D slice.
    Uses Otsu's thresholding, hole filling, and keeps the largest component.
    """
    s_min, s_max = slice_2d.min(), slice_2d.max()
    if s_max - s_min < 1e-9:
        return np.ones_like(slice_2d, dtype=bool)
        
    # Normalize slice to [0, 1] for robust thresholding
    s_norm = (slice_2d - s_min) / (s_max - s_min)
    
    # Determine threshold using Otsu, bounded to avoid edge cases
    try:
        thresh = threshold_otsu(s_norm)
        thresh = np.clip(thresh, 0.05, 0.5)
    except Exception:
        thresh = 0.1
        
    binary = s_norm > thresh
    
    # Fill internal holes
    binary = ndimage.binary_fill_holes(binary)
    
    # Label components to find the largest one (e.g. brain, filtering out background noise)
    labeled, num_features = ndimage.label(binary)
    if num_features > 0:
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        mask = (labeled == largest_label)
        mask = ndimage.binary_fill_holes(mask)
    else:
        mask = binary
        
    return mask
