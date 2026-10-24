"""
Similarity / error metrics between two aligned arrays (2D slices or 3D
volumes -- every metric operates on whatever ndarray it's given).

Every metric has the signature:

    fn(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float

`data_range` must be supplied explicitly by the caller (see core/preprocess.py
and api_main.py) -- never let a metric infer it implicitly, since that
silently changes PSNR/SSIM results.

`mask`, when given, is a boolean array the same shape as ref/test (e.g. from
core/preprocess.py::brain_mask). Pixel-independent metrics (MSE, MAE, NCC,
MI, ...) simply restrict their statistics to the masked pixels. Spatial
metrics that need neighborhood context to compute a per-pixel map (SSIM,
MS-SSIM, HFEN, GMSD) still filter/convolve over the *full* array first --
masking before filtering would introduce artificial edges at the mask
boundary -- and only average/aggregate the result over the masked region.

No UI imports here, ever.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_laplace, prewitt
from skimage.metrics import (
    mean_squared_error as _sk_mse,
    peak_signal_noise_ratio as _sk_psnr,
    structural_similarity as _sk_ssim,
)

from . import config

_EPS = 1e-12


def _safe_win_size(arr: np.ndarray, requested: int) -> int:
    """SSIM's win_size must be odd, >= 3, and <= the smallest array dimension.
    Arrays too small for any valid window raise a controlled ValueError (captured
    as None by compute_metrics)."""
    max_allowed = min(arr.shape)
    if max_allowed < 3:
        raise ValueError(
            f"Array too small for SSIM: smallest dimension {max_allowed} < 3."
        )
    win = min(requested, max_allowed)
    if win % 2 == 0:
        win -= 1
    return win


def _validate_mask(mask, ref: np.ndarray, test: np.ndarray) -> bool:
    """Raise ValueError if a mask doesn't match the array shape; return whether
    the mask has any foreground pixels."""
    if mask is None:
        return False
    if mask.shape != ref.shape or mask.shape != test.shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match array shape {ref.shape}."
        )
    return bool(mask.any())


def _seg_threshold(ref: np.ndarray, test: np.ndarray, data_range: float) -> float:
    """Shared midpoint threshold for segmentation metrics: actual finite combined
    minimum + data_range/2. Handles negative-valued arrays, where the old
    0.5*data_range would sit above the whole range (e.g. data_range=1 with values
    in [-0.5, 0.5])."""
    vals = np.concatenate([np.ravel(ref), np.ravel(test)])
    vals = vals[np.isfinite(vals)]
    lo = float(vals.min()) if vals.size else 0.0
    return lo + data_range / 2.0


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def psnr(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    if mask is None:
        if np.array_equal(ref, test):
            return float("inf")
        return float(_sk_psnr(ref, test, data_range=data_range))

    r = ref[mask].astype(np.float64)
    t = test[mask].astype(np.float64)
    mse_val = np.mean((r - t) ** 2)
    if mse_val < _EPS:
        return float("inf")
    return float(10 * np.log10((data_range ** 2) / mse_val))


def _auto_background_mask(ref: np.ndarray, data_range: float) -> np.ndarray:
    """Foreground-only mask (excludes background) so a large, trivially-matching
    background doesn't inflate SSIM. Threshold is ref.min() plus a fraction of
    data_range, so data whose minimum is not 0 (e.g. CT starting at -1024) is
    handled correctly."""
    return ref > (ref.min() + config.SSIM_AUTO_BACKGROUND_THRESHOLD * data_range)


def ssim(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    # Shallow 3D (any axis < 3): no valid 3D window exists, so average the 2D
    # SSIM over slices along the shallow axis.
    if ref.ndim == 3 and min(ref.shape) < 3:
        axis = int(np.argmin(ref.shape))
        per_slice = [
            ssim(np.take(ref, i, axis=axis), np.take(test, i, axis=axis),
                 data_range, None if mask is None else np.take(mask, i, axis=axis))
            for i in range(ref.shape[axis])
        ]
        return float(np.mean(per_slice))

    win = _safe_win_size(ref, config.SSIM_WIN_SIZE)
    kw = dict(data_range=data_range, win_size=win, gaussian_weights=True,
              sigma=config.SSIM_GAUSSIAN_SIGMA, use_sample_covariance=False)
    if mask is None:
        mask = _auto_background_mask(ref, data_range)

    _, smap = _sk_ssim(ref, test, full=True, **kw)
    return float(smap[mask].mean()) if mask.any() else float(smap.mean())


def ms_ssim(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    """
    Standard 5-scale multi-scale SSIM, implemented directly with
    scipy/skimage so no extra heavy dependency is required. If the array is
    too small for all 5 scales, uses as many scales as fit (re-normalizing
    the weights) rather than crashing.
    """
    from skimage.measure import block_reduce

    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    n_scales = len(weights)

    r, t = ref.astype(np.float64), test.astype(np.float64)
    m = mask if mask is not None else _auto_background_mask(r, data_range)
    values = []  # mean SSIM at each scale (luminance*contrast*structure)
    for scale in range(n_scales):
        if min(r.shape) < 4:
            break
        win = _safe_win_size(r, config.SSIM_WIN_SIZE)
        kw = dict(data_range=data_range, win_size=win, gaussian_weights=True,
                  sigma=config.SSIM_GAUSSIAN_SIGMA, use_sample_covariance=False)
        _, smap = _sk_ssim(r, t, full=True, **kw)
        value = smap[m].mean() if m.any() else smap.mean()
        values.append(float(value))
        if scale < n_scales - 1:
            block = tuple([2] * r.ndim)
            r = block_reduce(r, block, func=np.mean)
            t = block_reduce(t, block, func=np.mean)
            if m is not None:
                m = block_reduce(m.astype(np.float64), block, func=np.mean) > 0.5

    if not values:
        raise ValueError("Array too small to compute MS-SSIM.")

    values_arr = np.clip(np.array(values), _EPS, None)
    w = weights[: len(values)]
    w = w / w.sum()
    return float(np.prod(values_arr ** w))


def mse(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    if mask is None:
        return float(_sk_mse(ref, test))
    r = ref[mask].astype(np.float64)
    t = test[mask].astype(np.float64)
    return float(np.mean((r - t) ** 2))


def rmse(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    return float(np.sqrt(mse(ref, test, data_range, mask=mask)))


def mae(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    r = ref.astype(np.float64)
    t = test.astype(np.float64)
    if mask is not None:
        r, t = r[mask], t[mask]
    return float(np.mean(np.abs(r - t)))


def _relative_l2_error(ref: np.ndarray, test: np.ndarray) -> float:
    error = float(np.linalg.norm(ref - test))
    denom = float(np.linalg.norm(ref))
    if denom == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / denom


def nrmse(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    r = (ref[mask] if mask is not None else ref).astype(np.float64)
    t = (test[mask] if mask is not None else test).astype(np.float64)
    return _relative_l2_error(r, t)


def ncc(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    r = ref[mask] if mask is not None else ref.ravel()
    t = test[mask] if mask is not None else test.ravel()
    r = r.astype(np.float64) - r.astype(np.float64).mean()
    t = t.astype(np.float64) - t.astype(np.float64).mean()
    denom = np.linalg.norm(r) * np.linalg.norm(t)
    if denom < _EPS:
        return 0.0
    return float(np.dot(r, t) / denom)


def mutual_information(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    from sklearn.metrics import mutual_info_score

    bins = config.MI_BINS
    r = ref[mask] if mask is not None else ref.ravel()
    t = test[mask] if mask is not None else test.ravel()
    r_min, r_max = r.min(), r.max()
    t_min, t_max = t.min(), t.max()
    if r_max - r_min < _EPS or t_max - t_min < _EPS:
        return 0.0
    r_binned = np.digitize(r, np.linspace(r_min, r_max, bins))
    t_binned = np.digitize(t, np.linspace(t_min, t_max, bins))
    return float(mutual_info_score(r_binned, t_binned))


def normalized_mutual_information(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    """Joint-entropy-normalized mutual information. One implementation for both
    masked and unmasked calls. Degenerate inputs are handled explicitly: both
    images constant and equal -> 1.0, constant and different -> 0.0, either
    constant alone -> 0.0. Never returns NaN."""
    bins = config.MI_BINS
    r = ref[mask].ravel() if mask is not None else ref.ravel()
    t = test[mask].ravel() if mask is not None else test.ravel()
    if r.size == 0 or t.size == 0:
        return 0.0
    # Constant inputs collapse to one histogram cell with zero entropy, so
    # distinguish them at the data level (equal constants = perfect agreement).
    r_const = r.min() == r.max()
    t_const = t.min() == t.max()
    if r_const or t_const:
        if r_const and t_const:
            return 1.0 if r[0] == t[0] else 0.0
        return 0.0

    hist_2d, _, _ = np.histogram2d(r, t, bins=bins)
    total = hist_2d.sum()
    if total < _EPS:
        return 0.0
    pxy = hist_2d / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    def _entropy(p):
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    hx, hy, hxy = _entropy(px), _entropy(py), _entropy(pxy.ravel())
    if hxy < _EPS:
        return 0.0
    return (hx + hy) / hxy


def hfen(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    sigma = config.HFEN_SIGMA
    log_ref = gaussian_laplace(ref.astype(np.float64), sigma=sigma)
    log_test = gaussian_laplace(test.astype(np.float64), sigma=sigma)
    if mask is not None:
        log_ref = log_ref[mask]
        log_test = log_test[mask]
    return _relative_l2_error(log_ref, log_test)


def gmsd(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    r = ref.astype(np.float64)
    t = test.astype(np.float64)

    def _grad_mag(arr):
        grads = [prewitt(arr, axis=ax) for ax in range(arr.ndim)]
        return np.sqrt(sum(g ** 2 for g in grads))

    gr = _grad_mag(r)
    gt = _grad_mag(t)
    c = 0.0026 * (data_range ** 2) if data_range else 1e-6
    gms = (2 * gr * gt + c) / (gr ** 2 + gt ** 2 + c)
    if mask is not None:
        gms = gms[mask]
    return float(np.std(gms))


# ---------------------------------------------------------------------------
# Mask & Frequency Domain Metrics
# ---------------------------------------------------------------------------

def _seg_masks(ref: np.ndarray, test: np.ndarray, data_range: float, mask):
    """Binarize ref/test with the shared segmentation threshold and restrict to
    `mask` if given."""
    _validate_mask(mask, ref, test)
    thresh = _seg_threshold(ref, test, data_range)
    r_bool = ref > thresh
    t_bool = test > thresh
    if mask is not None:
        r_bool, t_bool = r_bool & mask, t_bool & mask
    return r_bool, t_bool


def dice(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    """
    Dice Similarity Coefficient (DSC) between binarized Reference and Test masks.
    Both empty -> 1.0 (identical empty segmentations).
    """
    r_bool, t_bool = _seg_masks(ref, test, data_range, mask)
    intersection = np.logical_and(r_bool, t_bool).sum()
    total = r_bool.sum() + t_bool.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def jaccard(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    """
    Jaccard Index / Intersection over Union (IoU) between binarized Reference and Test masks.
    Both empty -> 1.0 (identical empty segmentations).
    """
    r_bool, t_bool = _seg_masks(ref, test, data_range, mask)
    intersection = np.logical_and(r_bool, t_bool).sum()
    union = np.logical_or(r_bool, t_bool).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def hd95(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None, spacing=None) -> float:
    """
    95th Percentile Hausdorff Distance, true symmetric: P95 of the combined
    directed distances, computed via EDT. Both segmentations empty -> 0; one
    empty -> inf. Units are pixels/voxels, or the units of `spacing` (e.g. mm)
    when a per-axis `spacing` (one entry per array axis) is given.
    """
    r_bool, t_bool = _seg_masks(ref, test, data_range, mask)
    if not r_bool.any():
        return 0.0 if not t_bool.any() else float("inf")
    if not t_bool.any():
        return float("inf")

    dist_t = distance_transform_edt(~t_bool, sampling=spacing)  # distance from each r-voxel to t
    dist_r = distance_transform_edt(~r_bool, sampling=spacing)  # distance from each t-voxel to r
    d = np.concatenate([dist_t[r_bool], dist_r[t_bool]])
    if d.size == 0:
        return 0.0
    return float(np.percentile(d, 95))


def fft_error(ref: np.ndarray, test: np.ndarray, data_range: float, mask=None) -> float:
    """
    K-Space / Frequency Domain Relative Log-Power Spectrum Error.
    `mask` restricts the comparison to the masked region by zeroing the arrays
    outside it before the FFT (no wrap-around artifacts, unlike cropping).
    """
    _validate_mask(mask, ref, test)
    r = ref.astype(np.float64)
    t = test.astype(np.float64)
    if mask is not None:
        r = r * mask
        t = t * mask

    fft_ref = np.abs(np.fft.fftn(r))
    fft_test = np.abs(np.fft.fftn(t))

    log_ref = np.log1p(fft_ref)
    log_test = np.log1p(fft_test)

    return _relative_l2_error(log_ref, log_test)


METRICS = {
    "PSNR": psnr,
    "SSIM": ssim,
    "MS-SSIM": ms_ssim,
    "MSE": mse,
    "RMSE": rmse,
    "MAE": mae,
    "NRMSE": nrmse,
    "NCC": ncc,
    "MI": mutual_information,
    "NMI": normalized_mutual_information,
    "HFEN": hfen,
    "GMSD": gmsd,
    "Dice": dice,
    "Jaccard (IoU)": jaccard,
    "HD95": hd95,
    "FFT Error": fft_error,
}


def compute_metrics(ref: np.ndarray, test: np.ndarray, data_range: float, metric_names=None, mask=None, spacing=None):
    """
    Run the requested metrics (default: all registered) and return
    {name: value_or_None}. A metric that raises is recorded as None so one
    bad metric (e.g. MS-SSIM on a too-small array) never crashes the batch.

    `mask`, if given, is a boolean array the same shape as ref/test (e.g.
    core/preprocess.py::brain_mask) restricting every metric to that region
    instead of the whole frame. A shape-mismatched mask raises ValueError; an
    empty (all-False) mask yields None for every requested metric -- no NaN or
    misleading values.

    `spacing` (optional, per-axis, e.g. mm) is forwarded to HD95 only.
    """
    if mask is not None and (mask.shape != ref.shape or mask.shape != test.shape):
        raise ValueError(
            f"Mask shape {mask.shape} does not match array shape {ref.shape}."
        )
    names = metric_names if metric_names is not None else list(METRICS.keys())
    results = {}
    if mask is not None and not mask.any():
        for name in names:
            if name in METRICS:
                results[name] = None
        return results
    for name in names:
        fn = METRICS.get(name)
        if fn is None:
            continue
        try:
            kw = {"spacing": spacing} if name == "HD95" else {}
            results[name] = fn(ref, test, data_range, mask=mask, **kw)
        except Exception:
            results[name] = None
    return results
