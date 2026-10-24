"""
Format-agnostic image/volume loading.

Public API: `load_image(file_input) -> (np.ndarray, ImageMeta)`

Design rules (do not violate):
- Pure loading only. No normalization, no resampling. Just faithful reading,
  a cast to float32, and metadata extraction.
- 3D arrays are returned in (Z, Y, X) order. 2D arrays are (Y, X).
- No UI imports here, ever.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ImageMeta:
    source_path: str
    kind: str            # "jpg" | "png" | "dicom_single" | "dicom_series" | "nifti" | "npy"
    ndim: int             # 2 or 3
    shape: tuple
    dtype: str
    intensity_min: float
    intensity_max: float
    spacing: tuple | None = None  # voxel spacing if available (DICOM/NIfTI), else None
    warnings: list | None = None  # non-fatal notes generated while loading (e.g. RGB->gray)
    window_center: float | None = None  # DICOM WindowCenter (0028,1050), for display windowing only
    window_width: float | None = None   # DICOM WindowWidth (0028,1051), for display windowing only
    window_centers: list | None = None  # per-slice WindowCenter for dicom_series (slices can vary in scale)
    window_widths: list | None = None   # per-slice WindowWidth for dicom_series
    sequence: str | None = None          # DICOM SeriesDescription / ProtocolName (e.g. "t2_tse_fs_sag"), else None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DICOM_EXTENSIONS = {".dcm"}
NIFTI_EXTENSIONS = {".nii", ".nii.gz"}
NPY_EXTENSIONS = {".npy"}


def _ext_of(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(lower)[1]


def _make_meta(source_path, kind, arr, spacing=None, warnings=None, window_center=None, window_width=None,
                window_centers=None, window_widths=None, sequence=None) -> ImageMeta:
    return ImageMeta(
        source_path=source_path,
        kind=kind,
        ndim=arr.ndim,
        shape=tuple(arr.shape),
        dtype=str(arr.dtype),
        intensity_min=float(np.min(arr)) if arr.size else float("nan"),
        intensity_max=float(np.max(arr)) if arr.size else float("nan"),
        spacing=spacing,
        warnings=warnings or [],
        window_center=window_center,
        window_width=window_width,
        window_centers=window_centers,
        window_widths=window_widths,
        sequence=sequence,
    )


def _validate_array(arr: np.ndarray) -> np.ndarray:
    """Enforce the loader contract on every returned array: float32, 2D/3D,
    nonempty with no zero-length axis, and all finite. Raises a clear ValueError
    otherwise."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Image array must be 2D or 3D, got ndim={arr.ndim} (shape={arr.shape}).")
    if arr.size == 0 or any(d == 0 for d in arr.shape):
        raise ValueError(f"Image array is empty or has a zero-length axis (shape={arr.shape}).")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Image array contains NaN or infinite values.")
    return arr


def _canonicalize_dicom_pixels(arr: np.ndarray, ds) -> tuple[np.ndarray, list[str]]:
    """Canonicalize DICOM pixels: RGB/RGBA color channels or multi-frame color to grayscale
    using luminosity weights; MONOCHROME1 invert with (min+max)-arr. Returns (arr, warnings)."""
    warnings = []
    photo = str(getattr(ds, "PhotometricInterpretation", "")).upper()
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)

    is_color = samples in (3, 4) or photo in (
        "RGB", "RGBA", "YBR_FULL", "YBR_FULL_422", "YBR_PARTIAL_422", "YBR_RCT", "YBR_ICT"
    )
    if arr.ndim in (3, 4) and arr.shape[-1] in (3, 4) and (is_color or photo not in ("MONOCHROME1", "MONOCHROME2")):
        rgb = arr[..., :3].astype(np.float32)
        arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        warnings.append(
            "Input DICOM was RGB/RGBA color; converted to grayscale using luminosity weights "
            "(0.2126R + 0.7152G + 0.0722B)."
        )
    elif photo == "MONOCHROME1":
        min_v = float(np.min(arr)) if arr.size else 0.0
        max_v = float(np.max(arr)) if arr.size else 0.0
        arr = (min_v + max_v) - arr
        warnings.append("Inverted MONOCHROME1 DICOM to MONOCHROME2 display scale.")

    return arr, warnings


def _apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
    """Apply DICOM RescaleSlope/RescaleIntercept (0028,1053 / 0028,1052) so raw
    stored pixel values match the modality LUT units that WindowCenter/WindowWidth
    (and real viewers like RadiAnt) assume. Most MR is slope=1/intercept=0, but not
    all -- skipping this made pydicom-loaded previews too bright/washed."""
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope == 1.0 and intercept == 0.0:
        return arr
    return arr * slope + intercept


def _extract_window(ds) -> tuple[float | None, float | None]:
    """Pull DICOM WindowCenter/WindowWidth (0028,1050 / 0028,1051) for display only.
    Both tags can be multi-valued (DS list); take the first value."""
    try:
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if wc is None or ww is None:
            return None, None
        if hasattr(wc, "__iter__") and not isinstance(wc, str):
            wc = wc[0]
        if hasattr(ww, "__iter__") and not isinstance(ww, str):
            ww = ww[0]
        return float(wc), float(ww)
    except Exception:
        return None, None


def _extract_sequence(ds) -> str | None:
    """Human-readable MRI sequence name. Prefer SeriesDescription (0008,103E)
    (e.g. "t2_tse_fs_sag"); fall back to ProtocolName (0018,1030). None if neither."""
    for attr in ("SeriesDescription", "ProtocolName"):
        try:
            val = getattr(ds, attr, None)
            if val:
                s = str(val).strip()
                if s:
                    return s
        except Exception:
            continue
    return None


def _repair_missing_geometry(ds) -> list:
    """Some DICOM writers omit Rows (0028,0010) or Columns (0028,0011). pydicom
    then refuses to decode PixelData at all. When exactly one of the two is
    missing it is unambiguously recoverable from the pixel buffer's length, so
    fill it in rather than discarding an otherwise-valid slice. Returns a list of
    human-readable notes describing any repair performed."""
    notes = []
    if "PixelData" not in ds:
        return notes

    rows = ds.get("Rows")
    cols = ds.get("Columns")
    if rows is not None and cols is not None:
        return notes

    # Compressed/encapsulated PixelData is a compressed byte stream, not raw
    # samples, so its length cannot infer Rows/Columns. Never guess from it.
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", None)
    if ts is not None and (getattr(ts, "is_compressed", False) or getattr(ts, "is_encapsulated", False)):
        return notes
    if getattr(ds, "is_compressed", False) or getattr(ds, "is_encapsulated", False):
        return notes
    if not isinstance(ds.PixelData, (bytes, bytearray)):
        return notes

    bits_allocated = ds.get("BitsAllocated")
    samples = ds.get("SamplesPerPixel", 1) or 1
    if not bits_allocated:
        return notes

    bytes_per_sample = bits_allocated // 8
    if bytes_per_sample <= 0:
        return notes

    total_samples = len(ds.PixelData) // (bytes_per_sample * samples)
    if total_samples <= 0:
        return notes

    if rows is not None and cols is None and total_samples % int(rows) == 0:
        ds.Columns = total_samples // int(rows)
        notes.append(
            f"DICOM was missing Columns (0028,0011); inferred Columns={ds.Columns} "
            f"from Rows={int(rows)} and the pixel buffer length."
        )
    elif cols is not None and rows is None and total_samples % int(cols) == 0:
        ds.Rows = total_samples // int(cols)
        notes.append(
            f"DICOM was missing Rows (0028,0010); inferred Rows={ds.Rows} "
            f"from Columns={int(cols)} and the pixel buffer length."
        )
    return notes


def _load_jpg_png(path: str):
    import imageio.v3 as iio

    arr = iio.imread(path)
    warnings = []
    if arr.ndim == 3:
        # RGB(A) -> grayscale via luminosity weights. Metrics assume single channel.
        rgb = arr[..., :3].astype(np.float32)
        arr = (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )
        warnings.append(
            "Input was RGB; converted to grayscale using luminosity weights "
            "(0.2126R + 0.7152G + 0.0722B) before metrics."
        )
    arr = _validate_array(arr)
    ext = _ext_of(path).lstrip(".")
    kind = "jpg" if ext in ("jpg", "jpeg") else "png"
    meta = _make_meta(path, kind, arr, spacing=None, warnings=warnings)
    return arr, meta


def _load_dicom_single(path: str):
    import pydicom

    ds = pydicom.dcmread(path, force=True)
    geometry_notes = _repair_missing_geometry(ds)
    arr = ds.pixel_array.astype(np.float32)
    arr = _apply_rescale(arr, ds)
    arr, canon_notes = _canonicalize_dicom_pixels(arr, ds)
    geometry_notes.extend(canon_notes)
    window_center, window_width = _extract_window(ds)
    sequence = _extract_sequence(ds)

    if arr.ndim == 3:
        # Multi-frame single-file DICOM: keep the whole (Z, Y, X) volume.
        arr = _validate_array(arr)
        spacing = None
        try:
            px_spacing = ds.PixelSpacing  # (row, col) mm
            z_spacing = float(getattr(ds, "SpacingBetweenSlices", getattr(ds, "SliceThickness", 1.0)))
            spacing = (z_spacing, float(px_spacing[0]), float(px_spacing[1]))
        except Exception:
            spacing = None
        meta = _make_meta(path, "dicom_series", arr, spacing=spacing, warnings=geometry_notes,
                           window_center=window_center, window_width=window_width, sequence=sequence)
        return arr, meta

    arr = _validate_array(arr)
    spacing = None
    try:
        px_spacing = ds.PixelSpacing  # (row, col) mm -> (Y, X)
        spacing = (float(px_spacing[0]), float(px_spacing[1]))
    except Exception:
        spacing = None
    meta = _make_meta(path, "dicom_single", arr, spacing=spacing, warnings=geometry_notes,
                       window_center=window_center, window_width=window_width, sequence=sequence)
    return arr, meta


def _load_dicom_series(paths: list[str]):
    # Resolve file list
    if len(paths) == 1 and os.path.isdir(paths[0]):
        folder = paths[0]
        file_names = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    else:
        file_names = list(paths)
        
    # Read headers of all files to extract sort keys (physical projection, Instance number, or filename)
    import pydicom
    import re

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    slice_infos = []
    for path in file_names:
        try:
            # Fast check: skip known non-DICOM OS files
            base = os.path.basename(path).lower()
            if base in ['.ds_store', 'thumbs.db', 'desktop.ini'] or base.endswith('.txt') or base.endswith('.tmp'):
                continue

            # Read header only
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

            # Must look like a DICOM image slice. Accept EITHER dimension tag on
            # its own: writers do sometimes omit one (see _repair_missing_geometry),
            # and requiring both here silently dropped every slice of such a series.
            has_any_geometry = ds.get('Rows') is not None or ds.get('Columns') is not None
            has_pixel_data = 'PixelData' in ds
            if not (has_any_geometry or has_pixel_data):
                continue

            # Physical slice position = dot(IPP, normal) where normal =
            # cross(row, col) of ImageOrientationPatient. Works for axial, sagittal,
            # coronal, and oblique series alike.
            proj_pos = None
            iop = ds.get('ImageOrientationPatient')
            ipp = ds.get('ImagePositionPatient')
            if iop is not None and len(iop) >= 6 and ipp is not None and len(ipp) >= 3:
                try:
                    row = np.array([float(x) for x in iop[:3]], dtype=np.float64)
                    col = np.array([float(x) for x in iop[3:6]], dtype=np.float64)
                    normal = np.cross(row, col)
                    norm = float(np.linalg.norm(normal))
                    if norm > 0:
                        normal = normal / norm
                    proj_pos = float(np.dot(np.array([float(x) for x in ipp[:3]], dtype=np.float64), normal))
                except Exception:
                    proj_pos = None

            # Extract Instance Number
            inst_num = None
            if hasattr(ds, 'InstanceNumber') and ds.InstanceNumber is not None:
                try: inst_num = float(ds.InstanceNumber)
                except Exception: pass

            wc, ww = _extract_window(ds)
            slice_infos.append({
                "path": path,
                "proj_pos": proj_pos,
                "inst_num": inst_num,
                "filename_key": natural_sort_key(base),
                "series_uid": str(ds.get('SeriesInstanceUID', '')),
                "wc": wc,
                "ww": ww,
                "seq": _extract_sequence(ds),
            })
        except Exception:
            # Ignore files that cannot be parsed as DICOM
            continue

    # Keep only the largest series when the upload mixes several SeriesInstanceUIDs
    series_warnings = []
    distinct_uids = {s["series_uid"] for s in slice_infos if s["series_uid"]}
    if len(distinct_uids) > 1:
        counts = {}
        for s in slice_infos:
            counts[s["series_uid"]] = counts.get(s["series_uid"], 0) + 1
        best_uid = max(counts, key=lambda u: counts[u])
        best_n = counts[best_uid]
        slice_infos = [s for s in slice_infos if s["series_uid"] == best_uid or not s["series_uid"]]
        series_warnings.append(
            f"Upload contained {len(distinct_uids)} DICOM series; kept the largest "
            f"({best_n} slices) and ignored the rest."
        )

    # Sort hierarchy:
    # 1. Physical projection (dot(IPP, normal)); fallback Instance Number; then filename
    # 2. Instance Number (if available for all slices)
    # 3. Natural filename sort (default fallback)
    has_proj = len(slice_infos) > 0 and all(s["proj_pos"] is not None for s in slice_infos)
    has_inst = len(slice_infos) > 0 and all(s["inst_num"] is not None for s in slice_infos)

    if has_proj:
        slice_infos.sort(key=lambda x: x["proj_pos"])
        print(f"[DICOM Load] Sorted {len(slice_infos)} slices physically along the slice normal.")
    elif has_inst:
        slice_infos.sort(key=lambda x: x["inst_num"])
        print(f"[DICOM Load] Sorted {len(slice_infos)} slices by Instance number.")
    else:
        slice_infos.sort(key=lambda x: x["filename_key"])
        print(f"[DICOM Load] Sorted {len(slice_infos)} slices by filename (natural sort).")

    sorted_paths = [s["path"] for s in slice_infos]

    if not sorted_paths:
        raise ValueError("No valid DICOM image slices found in the uploaded files.")

    # Fallback z-spacing: median of adjacent projected positions (robust to gaps).
    proj_z_spacing = None
    if has_proj and len(slice_infos) > 1:
        diffs = [abs(slice_infos[i + 1]["proj_pos"] - slice_infos[i]["proj_pos"]) for i in range(len(slice_infos) - 1)]
        diffs = [d for d in diffs if d > 1e-6]
        if diffs:
            proj_z_spacing = float(np.median(diffs))

    try:
        import SimpleITK as sitk
        
        # SimpleITK needs gdcm names resolver if a single directory was passed originally,
        # but since we want the exact sorted file names, we pass sorted_paths directly!
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(sorted_paths)
        image = reader.Execute()

        arr = sitk.GetArrayFromImage(image).astype(np.float32)  # already (Z, Y, X)
        spacing_xyz = image.GetSpacing()  # (x, y, z)
        z_sp = proj_z_spacing if proj_z_spacing is not None else spacing_xyz[2]
        spacing = (float(z_sp), float(spacing_xyz[1]), float(spacing_xyz[0]))  # -> (z, y, x)

        # SimpleITK's reader drops WindowCenter/WindowWidth, so reuse the per-slice
        # values already read in the header pass (display-only, not used for metrics).
        # Slices in a series can carry different WC/WW, so keep one entry per slice.
        window_centers = [s["wc"] for s in slice_infos]
        window_widths = [s["ww"] for s in slice_infos]
        sequence = next((s["seq"] for s in slice_infos if s["seq"]), None)
        window_center = window_centers[0] if window_centers else None
        window_width = window_widths[0] if window_widths else None

        arr = _validate_array(arr)
        meta = _make_meta(",".join(sorted_paths), "dicom_series", arr, spacing=spacing,
                           warnings=list(series_warnings),
                           window_center=window_center, window_width=window_width,
                           window_centers=window_centers, window_widths=window_widths,
                           sequence=sequence)
        return arr, meta

    except (ImportError, Exception) as e:
        # Fallback using pydicom (direct pixels load)
        import pydicom

        arrays = []
        valid_paths = []
        window_centers, window_widths = [], []
        sequence = None
        geometry_notes = []
        skipped = []
        for path in sorted_paths:
            try:
                ds = pydicom.dcmread(path, force=True)
                # Repair before touching pixel_array: it's a property that raises
                # on missing Rows/Columns, so a plain hasattr() check would report
                # False and silently drop the slice.
                for note in _repair_missing_geometry(ds):
                    if note not in geometry_notes:
                        geometry_notes.append(note)
                slice_arr = _apply_rescale(ds.pixel_array.astype(np.float32), ds)
                slice_arr, c_notes = _canonicalize_dicom_pixels(slice_arr, ds)
                for note in c_notes:
                    if note not in geometry_notes:
                        geometry_notes.append(note)
                arrays.append(slice_arr)
                valid_paths.append(path)
                wc, ww = _extract_window(ds)
                window_centers.append(wc)
                window_widths.append(ww)
                if sequence is None:
                    sequence = _extract_sequence(ds)
            except Exception as slice_err:
                skipped.append(f"{os.path.basename(path)}: {slice_err}")
                continue

        if not arrays:
            detail = f" First failure: {skipped[0]}" if skipped else ""
            raise ValueError(
                f"No valid DICOM slices could be loaded from {len(sorted_paths)} file(s).{detail}"
            )
        if skipped:
            geometry_notes.append(f"Skipped {len(skipped)} unreadable slice(s); loaded {len(arrays)}.")
        window_center = window_centers[0] if window_centers else None
        window_width = window_widths[0] if window_widths else None

        arr = np.stack(arrays, axis=0)  # Shape: (Z, Y, X)

        # Spacing: use projected z-spacing if available, else SliceThickness
        spacing = None
        try:
            first_ds = pydicom.dcmread(valid_paths[0], force=True)
            px_spacing = first_ds.PixelSpacing  # (row, col) -> (y, x)
            z_spacing = proj_z_spacing
            if z_spacing is None:
                z_spacing = float(getattr(first_ds, "SliceThickness", 1.0))
            spacing = (z_spacing, float(px_spacing[0]), float(px_spacing[1]))
        except Exception:
            spacing = None

        arr = _validate_array(arr)
        meta = _make_meta(",".join(valid_paths), "dicom_series", arr, spacing=spacing,
                           warnings=series_warnings + geometry_notes,
                           window_center=window_center, window_width=window_width,
                           window_centers=window_centers, window_widths=window_widths,
                           sequence=sequence)
        return arr, meta


def _load_nifti(path: str):
    import nibabel as nib

    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    warnings = []

    if data.ndim > 3:
        warnings.append(
            f"NIfTI volume had {data.ndim} dimensions; taking the first volume "
            "(index 0 along trailing axes)."
        )
        # Collapse trailing dims by taking index 0 repeatedly.
        while data.ndim > 3:
            data = data[..., 0]

    spacing = None
    try:
        zooms = img.header.get_zooms()
        if data.ndim == 3 and len(zooms) >= 3:
            spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # (z, y, x)
        elif data.ndim == 2 and len(zooms) >= 2:
            spacing = (float(zooms[1]), float(zooms[0]))  # (y, x)
    except Exception:
        spacing = None

    # nibabel gives (X, Y, Z) or (X, Y); reorder to (Z, Y, X) or (Y, X) to match project convention.
    if data.ndim == 3:
        data = np.transpose(data, (2, 1, 0))
    elif data.ndim == 2:
        data = np.transpose(data, (1, 0))

    data = _validate_array(data)
    meta = _make_meta(path, "nifti", data, spacing=spacing, warnings=warnings)
    return data, meta


def _load_npy(path: str):
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float32)
    warnings = []

    if arr.ndim == 4:
        c = arr.shape[-1]
        if c == 1:
            arr = arr[..., 0]
            warnings.append("NPY 4D channel-last single channel array squeezed to 3D (Z, Y, X).")
        elif c in (3, 4):
            rgb = arr[..., :3]
            arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            warnings.append("NPY 4D channel-last color array converted to 3D grayscale (Z, Y, X).")
        else:
            warnings.append(
                f"NPY array had {arr.ndim} dimensions; taking index 0 along trailing axes."
            )
            while arr.ndim > 3:
                arr = arr[..., 0]
    elif arr.ndim > 4:
        warnings.append(
            f"NPY array had {arr.ndim} dimensions; taking index 0 along trailing axes."
        )
        while arr.ndim > 3:
            arr = arr[..., 0]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3]
            arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            warnings.append("NPY 3D channel-last color array converted to 2D grayscale (Y, X).")
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
            warnings.append("NPY 3D channel-last single channel array squeezed to 2D (Y, X).")

    arr = _validate_array(arr)
    meta = _make_meta(path, "npy", arr, spacing=None, warnings=warnings)
    return arr, meta


def load_image(file_input) -> tuple[np.ndarray, ImageMeta]:
    """
    file_input: a single path (str), or a list of paths (DICOM series), or a
    directory path (DICOM series folder).

    Dispatches on extension / input type. Returns the pixel data as a float32
    numpy array plus an ImageMeta. Always returns float32 (values are NOT
    rescaled here -- just cast). Orientation: (Z, Y, X) for 3D, (Y, X) for 2D.
    """
    # List of paths (or a bare directory) -> DICOM series.
    if isinstance(file_input, (list, tuple)):
        paths = list(file_input)
        if len(paths) == 1 and not os.path.isdir(paths[0]):
            return _dispatch_single(paths[0])
        return _load_dicom_series(paths)

    if isinstance(file_input, str) and os.path.isdir(file_input):
        return _load_dicom_series([file_input])

    return _dispatch_single(file_input)


def _looks_like_dicom(path: str) -> bool:
    """
    DICOM Part 10 files have a 128-byte preamble followed by the magic bytes
    b"DICM". Scanner exports (e.g. numbered files with no extension) often
    lack a ".dcm" suffix but still follow this layout. If that check fails,
    we try reading the file using pydicom with force=True to handle preamble-less DICOMs.
    """
    try:
        with open(path, "rb") as f:
            f.seek(128)
            if f.read(4) == b"DICM":
                return True
        # Try reading with pydicom (force=True) to handle preamble-less DICOMs
        import pydicom
        try:
            pydicom.dcmread(path, force=True)
            return True
        except Exception:
            return False
    except OSError:
        return False


def _dispatch_single(path: str) -> tuple[np.ndarray, ImageMeta]:
    ext = _ext_of(path)
    if ext in IMAGE_EXTENSIONS:
        return _load_jpg_png(path)
    if ext in DICOM_EXTENSIONS:
        return _load_dicom_single(path)
    if ext in NIFTI_EXTENSIONS:
        return _load_nifti(path)
    if ext in NPY_EXTENSIONS:
        return _load_npy(path)
    if _looks_like_dicom(path):
        return _load_dicom_single(path)
    raise ValueError(f"Unsupported file extension: {ext!r} (path={path!r})")
