"""
2D rigid/affine slice registration via SimpleITK.

Uses Mattes mutual information + regular-step gradient descent with a
multi-resolution pyramid, scoped to run interactively per comparison
request instead of as an offline batch job.

No UI imports here, ever.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk

MATTES_MI_BINS = 50
SAMPLING_PERCENTAGE = 0.20
OPT_LEARNING_RATE = 2.0
OPT_MIN_STEP = 1e-4
OPT_ITERATIONS = 200
OPT_GRADIENT_TOLERANCE = 1e-6
SHRINK_FACTORS = [4, 2, 1]
SMOOTHING_SIGMAS = [2, 1, 0]
INTERPOLATOR = sitk.sitkLinear


@dataclass
class RegistrationInfo:
    transform_type: str
    translation: tuple    # (tx, ty) or (tx, ty, tz); pixels for 2D (spacing 1.0), physical units for 3D
    rotation_deg: float   # 2D: in-plane angle; 3D: max abs rotation angle
    final_metric: float   # Mattes MI is negative-of-MI convention: more negative = better match
    iterations_used: int
    stop_condition: str


def _array_to_image(arr: np.ndarray) -> sitk.Image:
    return sitk.GetImageFromArray(np.ascontiguousarray(arr.astype(np.float32)))


def _build_method(transform_type: str) -> sitk.ImageRegistrationMethod:
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=MATTES_MI_BINS)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(SAMPLING_PERCENTAGE)
    R.SetInterpolator(INTERPOLATOR)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=OPT_LEARNING_RATE,
        minStep=OPT_MIN_STEP,
        numberOfIterations=OPT_ITERATIONS,
        gradientMagnitudeTolerance=OPT_GRADIENT_TOLERANCE,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=SHRINK_FACTORS)
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=SMOOTHING_SIGMAS)
    R.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return R


def _extract_params(final_tx: sitk.Transform, transform_type: str):
    params = final_tx.GetParameters()
    if transform_type == "rigid":
        angle_rad, tx, ty = params
        return (tx, ty), float(np.degrees(angle_rad))
    a11, a12, a21, a22, tx, ty = params
    return (tx, ty), float(np.degrees(np.arctan2(a21, a11)))


def register_2d(fixed_arr: np.ndarray, moving_arr: np.ndarray, transform_type: str = "rigid"):
    """
    Register `moving_arr` onto `fixed_arr`'s grid. Returns (sitk.Transform, RegistrationInfo).
    `transform_type`: "rigid" (rotation+translation) or "affine" (+scale/shear).
    """
    if transform_type not in ("rigid", "affine"):
        raise ValueError(f"Unknown transform_type: {transform_type!r}")

    fixed = _array_to_image(fixed_arr)
    moving = _array_to_image(moving_arr)

    tx0 = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler2DTransform() if transform_type == "rigid" else sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = _build_method(transform_type)
    R.SetInitialTransform(tx0, inPlace=False)
    final_tx = R.Execute(fixed, moving)

    translation, rotation_deg = _extract_params(final_tx, transform_type)
    info = RegistrationInfo(
        transform_type=transform_type,
        translation=translation,
        rotation_deg=rotation_deg,
        final_metric=float(R.GetMetricValue()),
        iterations_used=int(R.GetOptimizerIteration()),
        stop_condition=R.GetOptimizerStopConditionDescription(),
    )
    return final_tx, info


def apply_transform_2d(reference_arr: np.ndarray, moving_arr: np.ndarray, tx: sitk.Transform) -> np.ndarray:
    """Resample `moving_arr` onto `reference_arr`'s grid using `tx`. Output dtype matches moving_arr."""
    reference = _array_to_image(reference_arr)
    moving = _array_to_image(moving_arr)
    resampled = sitk.Resample(moving, reference, tx, INTERPOLATOR, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled).astype(moving_arr.dtype)


def _array_to_image_3d(arr: np.ndarray, spacing=None) -> sitk.Image:
    img = _array_to_image(arr)
    if spacing is not None:
        img.SetSpacing(tuple(float(v) for v in reversed(spacing)))  # (z,y,x) -> sitk (x,y,z)
    return img


def register_3d(fixed_arr: np.ndarray, moving_arr: np.ndarray, transform_type: str = "rigid", spacing=None):
    """
    3D counterpart of register_2d for (Z,Y,X) volumes. `spacing` is (z,y,x) or None (1.0).
    Returns (sitk.Transform, RegistrationInfo); translation is (tx, ty, tz) in physical units.
    """
    if transform_type not in ("rigid", "affine"):
        raise ValueError(f"Unknown transform_type: {transform_type!r}")

    fixed = _array_to_image_3d(fixed_arr, spacing)
    moving = _array_to_image_3d(moving_arr, spacing)

    tx0 = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform() if transform_type == "rigid" else sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = _build_method(transform_type)
    if min(fixed_arr.shape) < 16:  # a short axis can't survive the default 4x/2x pyramid
        R.SetShrinkFactorsPerLevel(shrinkFactors=[1])
        R.SetSmoothingSigmasPerLevel(smoothingSigmas=[0])
    R.SetInitialTransform(tx0, inPlace=False)
    final_tx = R.Execute(fixed, moving)

    params = final_tx.GetParameters()
    if transform_type == "rigid":
        translation = tuple(params[3:6])
        rotation_deg = float(np.degrees(np.max(np.abs(params[:3]))))
    else:
        translation = tuple(params[9:12])
        u, _, vt = np.linalg.svd(np.array(params[:9]).reshape(3, 3))
        cos = (np.trace(u @ vt) - 1.0) / 2.0
        rotation_deg = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    info = RegistrationInfo(
        transform_type=transform_type,
        translation=translation,
        rotation_deg=rotation_deg,
        final_metric=float(R.GetMetricValue()),
        iterations_used=int(R.GetOptimizerIteration()),
        stop_condition=R.GetOptimizerStopConditionDescription(),
    )
    return final_tx, info


def apply_transform_3d(reference_arr: np.ndarray, moving_arr: np.ndarray, tx: sitk.Transform, spacing=None) -> np.ndarray:
    """Resample `moving_arr` onto `reference_arr`'s grid using `tx`. Output dtype matches moving_arr."""
    reference = _array_to_image_3d(reference_arr, spacing)
    moving = _array_to_image_3d(moving_arr, spacing)
    resampled = sitk.Resample(moving, reference, tx, INTERPOLATOR, 0.0, moving.GetPixelID())
    return sitk.GetArrayFromImage(resampled).astype(moving_arr.dtype)
