"""Self-check for core/metrics.py. Run with: python test_metrics.py"""
import numpy as np

from core.metrics import compute_metrics, dice, hd95, jaccard, mse, psnr, ssim
from core.preprocess import resample_to_match


def test_identical_arrays():
    rng = np.random.default_rng(0)
    ref = rng.random((64, 64)).astype(np.float32)
    test = ref.copy()
    assert psnr(ref, test, data_range=1.0) == float("inf")
    assert abs(ssim(ref, test, data_range=1.0) - 1.0) < 1e-6
    assert mse(ref, test, data_range=1.0) == 0
    assert dice(ref, test, data_range=1.0) == 1.0


def test_ct_like_range_ssim_finite():
    rng = np.random.default_rng(1)
    ref = rng.uniform(-1024, 1000, size=(64, 64)).astype(np.float32)
    test = (ref + rng.normal(0, 10, size=ref.shape)).astype(np.float32)
    data_range = float(ref.max() - ref.min())
    value = ssim(ref, test, data_range=data_range)
    assert np.isfinite(value), value
    assert value > 0.5, value


def test_empty_mask_returns_none():
    rng = np.random.default_rng(2)
    ref = rng.random((64, 64)).astype(np.float32)
    test = rng.random((64, 64)).astype(np.float32)
    empty = np.zeros((64, 64), dtype=bool)
    results = compute_metrics(ref, test, data_range=1.0, mask=empty)
    assert results, "expected at least one metric"
    assert all(v is None for v in results.values()), results


def test_hd95_spacing_scales():
    ref = np.zeros((64, 64), dtype=np.float32)
    test = np.zeros((64, 64), dtype=np.float32)
    ref[10:30, 10:30] = 1.0
    test[14:34, 10:30] = 1.0
    px = hd95(ref, test, data_range=1.0)
    assert px > 0
    assert hd95(ref, test, data_range=1.0, spacing=(2.0, 2.0)) == 2 * px


def test_masked_hd95_keeps_spatial_coordinates():
    for shape, spacing in [((16, 16), (2.0, 3.0)), ((8, 16, 16), (4.0, 2.0, 3.0))]:
        ref = np.zeros(shape, dtype=np.float32)
        test = ref.copy()
        p = tuple(3 for _ in shape)
        q = (*p[:-2], p[-2] + 1, p[-1])
        ref[p], test[q] = 1.0, 1.0
        mask = np.ones(shape, dtype=bool)
        # Exclude an unrelated point: an irregular ROI must preserve distances too.
        mask[(0,) * len(shape)] = False
        assert hd95(ref, test, 1.0, mask=mask) == 1.0
        result = compute_metrics(ref, test, 1.0, ["HD95"], mask=mask, spacing=spacing)
        assert result["HD95"] == spacing[-2], result
        assert dice(ref, test, 1.0, mask=mask) == 0.0
        assert jaccard(ref, test, 1.0, mask=mask) == 0.0
        mask[q] = False
        assert hd95(ref, test, 1.0, mask=mask, spacing=spacing) == float("inf")


def test_relative_errors_with_zero_reference():
    ref = np.zeros((16, 16), dtype=np.float32)
    test = np.ones_like(ref)
    names = ["NRMSE", "HFEN", "FFT Error"]
    for mask in (None, np.ones_like(ref, dtype=bool)):
        different = compute_metrics(ref, test, 1.0, names, mask=mask)
        identical = compute_metrics(ref, ref, 1.0, names, mask=mask)
        assert all(value == float("inf") for value in different.values()), different
        assert all(value == 0.0 for value in identical.values()), identical
        # Scaling a nonzero reference by two has relative L2 error one.
        scaled = compute_metrics(test, 2 * test, 1.0, ["NRMSE", "HFEN"], mask=mask)
        assert all(abs(value - 1.0) < 1e-12 for value in scaled.values()), scaled


def test_resampling_preserves_constants_and_endpoints():
    for source, target in [((8, 8), (16, 16)), ((8, 8), (5, 6)),
                           ((3, 8, 8), (7, 16, 16)), ((1, 8, 8), (4, 16, 16))]:
        arr = np.ones(source, dtype=np.float32)
        out = resample_to_match(arr, target)
        assert out.shape == target and out.dtype == arr.dtype
        np.testing.assert_array_equal(out, np.ones(target, dtype=arr.dtype))
    ramp = np.tile(np.arange(8, dtype=np.float32), (8, 1))
    expected = np.tile(np.linspace(0, 7, 16, dtype=np.float32), (16, 1))
    np.testing.assert_allclose(resample_to_match(ramp, (16, 16)), expected, atol=1e-6)
    assert resample_to_match(ramp, ramp.shape) is ramp


if __name__ == "__main__":
    test_hd95_spacing_scales()
    test_identical_arrays()
    test_ct_like_range_ssim_finite()
    test_empty_mask_returns_none()
    test_masked_hd95_keeps_spatial_coordinates()
    test_relative_errors_with_zero_reference()
    test_resampling_preserves_constants_and_endpoints()
    print("OK")
