"""
Colorized difference / error map generation.

Returns plain uint8 RGB numpy arrays (H, W, 3) so the caller (an <img> tag
over the web API, or any other UI) can display them without any UI-specific
conversion living in this module. No UI imports here, ever.
"""
from __future__ import annotations

import numpy as np

from . import config


def compute_diff_map(
    ref: np.ndarray,
    test: np.ndarray,
    mode: str | None = None,
    colormap: str | None = None,
    amplify: float | None = None,
    percentile_clip: float | None = None,
    clip_hi: float | None = None,
) -> np.ndarray:
    """
    Build a colorized error map for a single 2D slice (pass a 2D array; for
    3D volumes the caller selects a slice first).

    mode: "absolute" (|A-B|) or "signed" (A-B, use a diverging cmap).
    clip_hi: explicit display ceiling for the error magnitude. Pass the same
    value to several maps (see diff_clip_value) so they share one color scale;
    None derives it from this map's own percentile.
    Returns a uint8 RGB array (H, W, 3).
    """
    import matplotlib as mpl

    mode = mode or config.DIFFMAP_MODE
    colormap = colormap or config.DIFFMAP_COLORMAP
    amplify = config.DIFFMAP_AMPLIFY if amplify is None else amplify
    percentile_clip = (
        config.DIFFMAP_PERCENTILE_CLIP if percentile_clip is None else percentile_clip
    )

    if ref.ndim != 2 or test.ndim != 2:
        raise ValueError(
            f"compute_diff_map expects 2D slices, got ref.ndim={ref.ndim}, "
            f"test.ndim={test.ndim}"
        )

    # Calculate difference. Using test - ref so that positive difference represents
    # test being higher than ref, and negative difference represents test being lower.
    diff = test.astype(np.float64) - ref.astype(np.float64)
    if mode == "absolute":
        diff = np.abs(diff)
    elif mode != "signed":
        raise ValueError(f"Unknown DIFFMAP_MODE: {mode!r}")

    diff = diff * amplify

    if clip_hi is None:
        clip_hi = np.percentile(np.abs(diff), percentile_clip) if diff.size else 0.0
    clip_hi = max(float(clip_hi), 1e-9)
    if mode == "absolute":
        normed = np.clip(diff / clip_hi, 0.0, 1.0)
    else:
        normed = np.clip(diff / clip_hi, -1.0, 1.0)
        normed = (normed + 1.0) / 2.0  # map [-1, 1] -> [0, 1] for the cmap

    if colormap == "blue_black_red":
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list("blue_black_red", ["blue", "black", "red"])
    else:
        try:
            cmap = mpl.colormaps[colormap]
        except Exception:
            cmap = mpl.cm.get_cmap(colormap)
    rgba = cmap(normed)  # (H, W, 4) floats in [0, 1]
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    return rgb


def diff_clip_value(
    ref: np.ndarray,
    test: np.ndarray,
    amplify: float | None = None,
    percentile_clip: float | None = None,
) -> float:
    """Display ceiling compute_diff_map would pick for this pair on its own.
    Take the max over several pairs to put their error maps on one scale."""
    amplify = config.DIFFMAP_AMPLIFY if amplify is None else amplify
    percentile_clip = (
        config.DIFFMAP_PERCENTILE_CLIP if percentile_clip is None else percentile_clip
    )
    diff = np.abs(test.astype(np.float64) - ref.astype(np.float64)) * amplify
    return float(np.percentile(diff, percentile_clip)) if diff.size else 0.0
