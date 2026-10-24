import io
import base64
import sys
import time
import uuid
import os
import shutil
import pickle
import re
import tempfile
import threading
from collections import OrderedDict
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse, Response

# Import core logic
from core.loaders import load_image, ImageMeta
from core.preprocess import check_compatibility, normalize_pair, resample_to_match, brain_mask
from core.metrics import compute_metrics, METRICS
from core.diffmap import compute_diff_map
from core.registration import register_2d, apply_transform_2d, register_3d, apply_transform_3d

app = FastAPI(title="Image & Volume Metric Studio")


class PipelineError(Exception):
    """Raised when a comparison can't proceed (bad/mismatched inputs). Carries an
    HTTP status so endpoints can turn it into a JSONResponse verbatim."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@app.exception_handler(PipelineError)
async def _pipeline_error_handler(request, exc: PipelineError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message}
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request, exc):
    """Log full exception details server-side while keeping the generic 500 client response
    safe from exposing internal file paths or exception details."""
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "An internal server error occurred."}
    )


# Serve HTML/JS. Resolve the directory absolutely rather than relying on the
# process CWD: when frozen by PyInstaller the app runs from a temp extraction
# dir (sys._MEIPASS) and a relative "static" would not exist.
if getattr(sys, "frozen", False):
    BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")

# ponytail: disk-backed temp session cache -> skipped: Redis / distributed session store, add when running on multi-node cluster.
CACHE_DIR = os.path.join(tempfile.gettempdir(), "image_compare_cache")
CACHE_TTL_SECONDS = 3600
CACHE_MAX_BYTES = 500 * 1024 * 1024
SESSION_ID_RE = re.compile(r"[0-9a-f]{32}")

# Bounded in-memory LRUs so repeated requests skip the disk and RAM stays capped.
MEM_CACHE_MAX = 4
_MEM_LOCK = threading.Lock()
_MEM_CACHE = OrderedDict()  # compare sessions: session_id -> loaded tuple


def _lru_put(cache: OrderedDict, key, value):
    with _MEM_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > MEM_CACHE_MAX:
            cache.popitem(last=False)


def _mem_put(key, value):
    _lru_put(_MEM_CACHE, key, value)


_RESAMPLE_CACHE = OrderedDict()  # (session_id, target_shape) -> resampled B (read-only; never mutate)


def _resampled_b(session_id, arr_b, target_shape):
    if arr_b.shape == tuple(target_shape):
        return arr_b
    key = (session_id, tuple(target_shape))
    with _MEM_LOCK:
        hit = _RESAMPLE_CACHE.get(key)
        if hit is not None:
            _RESAMPLE_CACHE.move_to_end(key)
            return hit
    out = resample_to_match(arr_b, tuple(target_shape))
    out.flags.writeable = False
    _lru_put(_RESAMPLE_CACHE, key, out)
    return out


def _hd95_spacing(spacing, ndim, is_slice_of_3d=False):
    """Spacing for HD95: (y,x) for 2D slices, (z,y,x) for volumes; None -> pixels."""
    if spacing is None:
        return None
    if is_slice_of_3d:
        return tuple(spacing[1:]) if len(spacing) == 3 else None
    return tuple(spacing) if len(spacing) == ndim else None


def _get_dir_size(path: str) -> int:
    total = 0
    try:
        for entry in os.scandir(path):
            if entry.is_file(follow_symlinks=False):
                total += entry.stat().st_size
    except Exception:
        pass
    return total


def _prune_disk_cache():
    if not os.path.exists(CACHE_DIR):
        return
    now = time.time()
    sessions = []
    try:
        entries = list(os.scandir(CACHE_DIR))
    except Exception:
        return

    for entry in entries:
        if not entry.is_dir(follow_symlinks=False):
            continue
        name = entry.name
        path = entry.path

        if name.startswith(".tmp_"):
            try:
                if now - entry.stat().st_mtime > 300:
                    shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
            continue

        meta_path = os.path.join(path, "meta.pkl")
        mtime = entry.stat().st_mtime
        if os.path.exists(meta_path):
            try:
                mtime = os.path.getmtime(meta_path)
            except Exception:
                pass

        if now - mtime > CACHE_TTL_SECONDS:
            shutil.rmtree(path, ignore_errors=True)
            continue

        size = _get_dir_size(path)
        sessions.append({"path": path, "mtime": mtime, "size": size})

    total_bytes = sum(s["size"] for s in sessions)
    if total_bytes > CACHE_MAX_BYTES:
        sessions.sort(key=lambda s: s["mtime"])
        for s in sessions:
            if total_bytes <= CACHE_MAX_BYTES:
                break
            shutil.rmtree(s["path"], ignore_errors=True)
            total_bytes -= s["size"]


def _save_session_to_disk(arr_a: np.ndarray, meta_a: ImageMeta, arr_b: np.ndarray, meta_b: ImageMeta, warnings: list) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    session_id = uuid.uuid4().hex
    staging_dir = os.path.join(CACHE_DIR, f".tmp_{session_id}_{os.getpid()}")
    target_dir = os.path.join(CACHE_DIR, session_id)

    os.makedirs(staging_dir, exist_ok=True)
    try:
        np.save(os.path.join(staging_dir, "arr_a.npy"), arr_a)
        np.save(os.path.join(staging_dir, "arr_b.npy"), arr_b)
        meta_payload = {
            "meta_a": meta_a,
            "meta_b": meta_b,
            "warnings": warnings,
            "timestamp": time.time()
        }
        with open(os.path.join(staging_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta_payload, f)

        os.replace(staging_dir, target_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    _prune_disk_cache()
    _mem_put(session_id, (arr_a, meta_a, arr_b, meta_b, warnings, session_id))
    return session_id


def _valid_session_id(session_id) -> bool:
    # Session ids are uuid4().hex; anything else could path-traverse out of
    # CACHE_DIR and reach pickle.load on an attacker-controlled file.
    return bool(session_id) and bool(SESSION_ID_RE.fullmatch(session_id))


def _load_session_from_disk(session_id: str):
    if not _valid_session_id(session_id):
        return None
    with _MEM_LOCK:
        if session_id in _MEM_CACHE:
            _MEM_CACHE.move_to_end(session_id)
            return _MEM_CACHE[session_id]
    session_dir = os.path.join(CACHE_DIR, session_id)
    if not os.path.isdir(session_dir):
        return None

    arr_a_path = os.path.join(session_dir, "arr_a.npy")
    arr_b_path = os.path.join(session_dir, "arr_b.npy")
    meta_path = os.path.join(session_dir, "meta.pkl")

    if not (os.path.isfile(arr_a_path) and os.path.isfile(arr_b_path) and os.path.isfile(meta_path)):
        return None

    try:
        arr_a = np.load(arr_a_path)
        arr_b = np.load(arr_b_path)
        with open(meta_path, "rb") as f:
            meta_payload = pickle.load(f)

        try:
            os.utime(meta_path, None)
        except Exception:
            pass

        meta_a = meta_payload["meta_a"]
        meta_b = meta_payload["meta_b"]
        warnings = meta_payload["warnings"]
        result = (arr_a, meta_a, arr_b, meta_b, warnings, session_id)
        _mem_put(session_id, result)
        return result
    except Exception:
        return None


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


def _extract_upload_ext(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(lower)[1]


def _save_upload_bytes(data: bytes, filename: str) -> str:
    ext = _extract_upload_ext(filename)
    if not ext:
        # Sniff for preamble-less DICOM
        is_dicom = len(data) > 132 and data[128:132] == b"DICM"
        if not is_dicom:
            import pydicom
            try:
                pydicom.dcmread(io.BytesIO(data), force=True)
                is_dicom = True
            except Exception:
                is_dicom = False
        ext = ".dcm" if is_dicom else ".png"
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
    return path


def _save_uploads(upload_files: list[UploadFile]) -> str | list[str]:
    """
    Saves a list of UploadFiles.
    If it's a single file, returns a single temp file path.
    If it's multiple files, creates a temp folder, saves them with ordinal prefixes, and returns list of paths.
    """
    if not upload_files:
        raise PipelineError("No files uploaded.", 400)
    if len(upload_files) == 1:
        data = upload_files[0].file.read()
        return _save_upload_bytes(data, upload_files[0].filename or "file")

    temp_dir = tempfile.mkdtemp()
    saved_paths = []
    try:
        for idx, f in enumerate(upload_files):
            data = f.file.read()
            filename = f.filename or f"file_{idx}"
            ext = _extract_upload_ext(filename)

            if not ext:
                is_dicom = len(data) > 132 and data[128:132] == b"DICM"
                if not is_dicom:
                    try:
                        import pydicom
                        pydicom.dcmread(io.BytesIO(data), force=True)
                        is_dicom = True
                    except Exception:
                        is_dicom = False
                ext = ".dcm" if is_dicom else ""

            safe_name = filename.replace("/", "_").replace("\\", "_")
            if ext and not safe_name.lower().endswith(ext):
                safe_name += ext

            unique_name = f"{idx:05d}_{safe_name}"
            path = os.path.join(temp_dir, unique_name)
            with open(path, 'wb') as out_f:
                out_f.write(data)
            saved_paths.append(path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    saved_paths.sort()
    return saved_paths


def _cleanup_paths(paths):
    """Safely cleans up files and temporary folders."""
    if not paths:
        return
    if isinstance(paths, list):
        for p in paths:
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
        if paths:
            parent = os.path.dirname(paths[0])
            if os.path.isdir(parent):
                try:
                    shutil.rmtree(parent, ignore_errors=True)
                except Exception:
                    pass
    else:
        if os.path.isfile(paths):
            try:
                os.remove(paths)
            except Exception:
                pass
        elif os.path.isdir(paths):
            try:
                shutil.rmtree(paths, ignore_errors=True)
            except Exception:
                pass


def _validate_loaded_array(arr: np.ndarray, label: str):
    if arr is None or arr.size == 0:
        raise PipelineError(f"Could not read the {label} dataset: image data is empty (zero size).", 400)
    if not np.isfinite(arr).all():
        raise PipelineError(f"Could not read the {label} dataset: image data contains non-finite values (NaN/Inf).", 400)


def _validate_roi_and_window(roi_x, roi_y, roi_w, roi_h, window_center, window_width):
    for name, val in [("roi_x", roi_x), ("roi_y", roi_y), ("roi_w", roi_w), ("roi_h", roi_h)]:
        if val is not None:
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise PipelineError(f"Invalid ROI parameter '{name}': value must be a finite number.", 400)
    for name, val in [("window_center", window_center), ("window_width", window_width)]:
        if val is not None:
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise PipelineError(f"Invalid window parameter '{name}': value must be a finite number.", 400)


def _slice_window(meta, idx, current_len):
    """Per-slice WindowCenter/WindowWidth if available and still aligned with the
    current (possibly resampled) slice count; otherwise falls back to the series-wide
    scalar window (still None if the DICOM didn't carry one -> caller uses percentile)."""
    wcs, wws = meta.window_centers, meta.window_widths
    if wcs and wws and len(wcs) == current_len and 0 <= idx < len(wcs):
        return wcs[idx], wws[idx]
    return meta.window_center, meta.window_width


def _array_to_b64(arr: np.ndarray) -> str:
    """Convert uint8 RGB numpy array to base64 image."""
    img = Image.fromarray(arr)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _json_safe_metrics(metrics: dict) -> dict:
    """JSON has no literal for infinity/NaN, and json.dumps' non-standard
    `Infinity` output is rejected by JSON.parse in the browser. PSNR is
    legitimately +inf when two images are pixel-identical, so encode non-finite
    values as sentinel strings ("inf"/"-inf") that the frontend revives back
    into numbers, and NaN as null."""
    safe = {}
    for name, val in metrics.items():
        if isinstance(val, float):
            if val == float("inf"):
                safe[name] = "inf"
                continue
            if val == float("-inf"):
                safe[name] = "-inf"
                continue
            if val != val:  # NaN
                safe[name] = None
                continue
        safe[name] = val
    return safe


def _window_to_float01(slc: np.ndarray, window_center=None, window_width=None) -> np.ndarray:
    """Map a raw slice into [0,1] for display using an explicit WindowCenter/
    WindowWidth (what PACS viewers render). Falls back to a percentile stretch
    when no window is supplied, which is still robust to the outlier-bright
    voxels that a raw min/max stretch would let wash out the whole image."""
    if window_center is not None and window_width and window_width > 1e-6:
        low = window_center - window_width / 2.0
        high = window_center + window_width / 2.0
    else:
        low, high = np.percentile(slc, [0.0, 99.9])
    span = high - low
    if span <= 1e-9:
        return np.zeros_like(slc, dtype=np.float32)
    return np.clip((slc - low) / span, 0.0, 1.0).astype(np.float32)


def _window_to_uint8(slc: np.ndarray, window_center=None, window_width=None) -> np.ndarray:
    return (_window_to_float01(slc, window_center, window_width) * 255).astype(np.uint8)


def _roi_mask(shape, roi_x, roi_y, roi_w, roi_h):
    """Boolean mask for a rectangular region of interest.

    The ROI arrives as fractions of the image size (0..1) rather than pixels so
    it survives the browser scaling the preview to fit its panel. Returns None
    when no usable rectangle was supplied.
    """
    if None in (roi_x, roi_y, roi_w, roi_h):
        return None
    if roi_w <= 0 or roi_h <= 0:
        return None

    height, width = shape[-2], shape[-1]
    x0 = int(np.floor(roi_x * width))
    y0 = int(np.floor(roi_y * height))
    x1 = int(np.ceil((roi_x + roi_w) * width))
    y1 = int(np.ceil((roi_y + roi_h) * height))

    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))

    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _combine_masks(roi, brain):
    """ROI and brain mask are both restrictions, so applying both means the
    intersection -- metrics over the brain tissue *inside* the drawn box."""
    if roi is None:
        return brain
    if brain is None:
        return roi
    return roi & brain


def _load_pair(file1, file2, session_id):
    """Resolve a (arr_a, meta_a, arr_b, meta_b, warnings, session_id) tuple from
    either the volume disk store or freshly uploaded files, caching the latter."""
    if session_id:
        cached = _load_session_from_disk(session_id)
        if cached is not None:
            return cached

    if not file1 or not file2:
        raise PipelineError("Session expired or files missing. Please upload files again.", 400)

    path1, path2 = None, None
    try:
        path1 = _save_uploads(file1)
        path2 = _save_uploads(file2)

        try:
            arr_a, meta_a = load_image(path1)
            _validate_loaded_array(arr_a, "Reference (A)")
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Could not read the Reference (A) dataset: {e}", 400)

        try:
            arr_b, meta_b = load_image(path2)
            _validate_loaded_array(arr_b, "Test (B)")
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(f"Could not read the Test (B) dataset: {e}", 400)
    finally:
        _cleanup_paths(path1)
        _cleanup_paths(path2)

    # Surface loader-level notes (e.g. a repaired/omitted DICOM geometry tag)
    # alongside the compatibility messages, tagged so it's clear which side they came from.
    warnings = []
    for label, meta in (("Reference · A", meta_a), ("Test · B", meta_b)):
        for note in (meta.warnings or []):
            warnings.append(f"{label}: {note}")
    warnings.extend(check_compatibility(meta_a, meta_b)["messages"])

    session_id = _save_session_to_disk(arr_a, meta_a, arr_b, meta_b, warnings)
    return arr_a, meta_a, arr_b, meta_b, warnings, session_id


def _prepare_slice(file1, file2, session_id, slice_idx, normalize, resample,
                    brain_only, do_register, register_type,
                    roi_x=None, roi_y=None, roi_w=None, roi_h=None,
                    window_center=None, window_width=None):
    """Shared front half of the comparison pipeline: load -> resample -> normalize
    -> extract the requested slice -> optionally register B onto A. Returns a dict
    of everything the response builders need, so /api/compare and the plot
    endpoint always render the exact same data."""
    _validate_roi_and_window(roi_x, roi_y, roi_w, roi_h, window_center, window_width)
    arr_a, meta_a, arr_b, meta_b, warnings, session_id = _load_pair(file1, file2, session_id)

    # Preprocess cached volumes without mutating the raw cached data
    arr_b_proc = arr_b
    meta_b_shape = meta_b.shape
    if resample and meta_a.shape != meta_b.shape:
        arr_b_proc = _resampled_b(session_id, arr_b, meta_a.shape)
        meta_b_shape = arr_b_proc.shape

    if arr_a.ndim != arr_b_proc.ndim:
        raise PipelineError(f"NDIM mismatch: {arr_a.ndim} vs {arr_b_proc.ndim}", 400)
    if arr_a.shape != arr_b_proc.shape:
        raise PipelineError(
            f"Shape mismatch: {arr_a.shape} vs {arr_b_proc.shape}. Try checking 'Resample'.",
            400
        )

    if normalize:
        arr_a_norm, arr_b_norm = normalize_pair(arr_a, arr_b_proc)
        data_range = 1.0
    else:
        arr_a_norm, arr_b_norm = arr_a, arr_b_proc
        range_a = meta_a.intensity_max - meta_a.intensity_min
        range_b = meta_b.intensity_max - meta_b.intensity_min
        data_range = max(range_a, range_b, 1e-9)

    # Slice extraction (for 3D)
    max_slices = 1
    is_3d = arr_a.ndim == 3
    if is_3d:
        max_slices = arr_a.shape[0]
        slice_idx = max(0, min(slice_idx, max_slices - 1))
        slice_a = arr_a_norm[slice_idx]
        slice_b = arr_b_norm[slice_idx]
        raw_slice_a = arr_a[slice_idx]
        raw_slice_b = arr_b_proc[slice_idx]
        wc_a, ww_a = _slice_window(meta_a, slice_idx, arr_a.shape[0])
        # arr_b_proc may have been resampled, so its slice count/index no longer
        # lines up with meta_b's per-slice window list -- only trust it unresampled.
        wc_b, ww_b = _slice_window(meta_b, slice_idx, arr_b_proc.shape[0])
    else:
        slice_idx = 0
        slice_a = arr_a_norm
        slice_b = arr_b_norm
        raw_slice_a = arr_a
        raw_slice_b = arr_b_proc
        wc_a, ww_a = meta_a.window_center, meta_a.window_width
        wc_b, ww_b = meta_b.window_center, meta_b.window_width

    # Optionally register slice B onto slice A (rigid/affine) so pixel-wise
    # metrics aren't biased by misalignment between the two acquisitions.
    # The same transform is applied to the raw slice so the preview reflects
    # what the metrics actually saw.
    reg_info = None
    fov = None
    if do_register:
        try:
            tx, info = register_2d(slice_a, slice_b, transform_type=register_type)
            # Pixels that fall outside B's field of view are zero-filled by the
            # warp; exclude them from scoring.
            fov_b = apply_transform_2d(slice_a, np.ones_like(slice_b, dtype=np.float32), tx) > 0.5
            slice_b = apply_transform_2d(slice_a, slice_b, tx)
            raw_slice_b = apply_transform_2d(raw_slice_a, raw_slice_b, tx)
            fov = fov_b
            reg_info = {
                "transform_type": info.transform_type,
                "translation_x": info.translation[0],
                "translation_y": info.translation[1],
                "rotation_deg": info.rotation_deg,
                "final_metric": info.final_metric,
                "iterations_used": info.iterations_used,
                "stop_condition": info.stop_condition,
            }
        except Exception as e:
            reg_info = {"error": f"{type(e).__name__}: {e}"}

    roi = _roi_mask(slice_a.shape, roi_x, roi_y, roi_w, roi_h)
    mask = _combine_masks(roi, brain_mask(slice_a) if brain_only else None)
    if fov is not None and not fov.all():
        mask = fov if mask is None else (mask & fov)
    if mask is not None and not mask.any():
        raise PipelineError(
            "The selected region contains no pixels to measure"
            + (" (the ROI and the brain mask don't overlap)." if roi is not None and brain_only
               else ". Draw a larger region."),
            400
        )

    return {
        "session_id": session_id, "warnings": warnings, "roi": roi,
        "meta_a": meta_a, "meta_b": meta_b, "meta_b_shape": meta_b_shape,
        "slice_a": slice_a, "slice_b": slice_b,
        "raw_slice_a": raw_slice_a, "raw_slice_b": raw_slice_b,
        "wc_a": wc_a, "ww_a": ww_a, "wc_b": wc_b, "ww_b": ww_b,
        "data_range": data_range, "mask": mask, "reg_info": reg_info,
        "slice_idx": slice_idx, "max_slices": max_slices, "is_3d": is_3d,
        "hd95_spacing": _hd95_spacing(meta_a.spacing, arr_a.ndim, is_3d),
    }


@app.post("/api/compare")
def api_compare(
    file1: list[UploadFile] = File(None),
    file2: list[UploadFile] = File(None),
    session_id: str = Form(None),
    slice_idx: int = Form(0),
    normalize: bool = Form(False),
    resample: bool = Form(False),
    brain_only: bool = Form(False),
    do_register: bool = Form(False, alias="register"),
    register_type: str = Form("rigid"),
    diff_mode: str = Form("absolute"),
    diff_cmap: str = Form("gray"),
    window_center: float = Form(None),
    window_width: float = Form(None),
    roi_x: float = Form(None),
    roi_y: float = Form(None),
    roi_w: float = Form(None),
    roi_h: float = Form(None)
):
    try:
        ctx = _prepare_slice(file1, file2, session_id, slice_idx, normalize,
                              resample, brain_only, do_register, register_type,
                              roi_x, roi_y, roi_w, roi_h, window_center, window_width)
    except PipelineError as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.message})

    session_id = ctx["session_id"]
    warnings = ctx["warnings"]
    meta_a, meta_b, meta_b_shape = ctx["meta_a"], ctx["meta_b"], ctx["meta_b_shape"]
    slice_a, slice_b = ctx["slice_a"], ctx["slice_b"]
    raw_slice_a, raw_slice_b = ctx["raw_slice_a"], ctx["raw_slice_b"]
    wc_a, ww_a, wc_b, ww_b = ctx["wc_a"], ctx["ww_a"], ctx["wc_b"], ctx["ww_b"]
    data_range, mask, reg_info = ctx["data_range"], ctx["mask"], ctx["reg_info"]
    slice_idx, max_slices, is_3d = ctx["slice_idx"], ctx["max_slices"], ctx["is_3d"]

    metrics_res = compute_metrics(slice_a, slice_b, data_range=data_range, mask=mask,
                                  spacing=ctx["hd95_spacing"])
    diff_arr = compute_diff_map(slice_a, slice_b, mode=diff_mode, colormap=diff_cmap)
    diff_b64 = _array_to_b64(diff_arr)

    manual_window = window_center is not None and window_width is not None
    if manual_window:
        wc_a = wc_b = window_center
        ww_a = ww_b = window_width

    scored_mask = ctx["mask"]
    measured = {
        "roi_active": ctx["roi"] is not None,
        "brain_only": bool(brain_only),
        "pixels": int(scored_mask.sum()) if scored_mask is not None else int(slice_a.size),
        "total_pixels": int(slice_a.size),
    }

    return {
        "session_id": session_id,
        "metrics": _json_safe_metrics(metrics_res),
        "hd95_unit": "mm" if ctx["hd95_spacing"] else "px",
        "measured": measured,
        "diff_map": f"data:image/png;base64,{diff_b64}",
        "preview_a": f"data:image/png;base64,{_array_to_b64(_window_to_uint8(raw_slice_a, wc_a, ww_a))}",
        "preview_b": f"data:image/png;base64,{_array_to_b64(_window_to_uint8(raw_slice_b, wc_b, ww_b))}",
        "window": {
            "center": wc_a,
            "width": ww_a,
            "manual": manual_window,
            "data_min": float(min(meta_a.intensity_min, meta_b.intensity_min)),
            "data_max": float(max(meta_a.intensity_max, meta_b.intensity_max))
        },
        "warnings": warnings,
        "is_3d": is_3d,
        "max_slices": max_slices,
        "current_slice": slice_idx,
        "registration": reg_info,
        "meta_a": {
            "kind": meta_a.kind,
            "shape": meta_a.shape,
            "dtype": meta_a.dtype,
            "min": meta_a.intensity_min,
            "max": meta_a.intensity_max,
            "sequence": meta_a.sequence
        },
        "meta_b": {
            "kind": meta_b.kind,
            "shape": meta_b_shape,
            "dtype": meta_b.dtype,
            "min": meta_b.intensity_min,
            "max": meta_b.intensity_max,
            "sequence": meta_b.sequence
        }
    }


@app.post("/api/combined_plot")
def api_combined_plot(
    file1: list[UploadFile] = File(None),
    file2: list[UploadFile] = File(None),
    session_id: str = Form(None),
    slice_idx: int = Form(0),
    normalize: bool = Form(False),
    resample: bool = Form(False),
    brain_only: bool = Form(False),
    do_register: bool = Form(False, alias="register"),
    register_type: str = Form("rigid"),
    diff_mode: str = Form("absolute"),
    diff_cmap: str = Form("gray"),
    window_center: float = Form(None),
    window_width: float = Form(None),
    roi_x: float = Form(None),
    roi_y: float = Form(None),
    roi_w: float = Form(None),
    roi_h: float = Form(None),
    dpi: int = Form(150)
):
    if dpi is None or not isinstance(dpi, (int, float)) or not np.isfinite(dpi) or not (1 <= dpi <= 600):
        return JSONResponse(status_code=400, content={"error": "DPI must be a number between 1 and 600."})
    dpi = int(dpi)

    from matplotlib.figure import Figure

    try:
        ctx = _prepare_slice(file1, file2, session_id, slice_idx, normalize,
                              resample, brain_only, do_register, register_type,
                              roi_x, roi_y, roi_w, roi_h, window_center, window_width)
    except PipelineError as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.message})

    meta_a, meta_b = ctx["meta_a"], ctx["meta_b"]
    raw_slice_a, raw_slice_b = ctx["raw_slice_a"], ctx["raw_slice_b"]
    wc_a, ww_a, wc_b, ww_b = ctx["wc_a"], ctx["ww_a"], ctx["wc_b"], ctx["ww_b"]

    if window_center is not None and window_width is not None:
        wc_a = wc_b = window_center
        ww_a = ww_b = window_width

    metrics_res = compute_metrics(ctx["slice_a"], ctx["slice_b"],
                                   data_range=ctx["data_range"], mask=ctx["mask"])
    diff_arr = compute_diff_map(ctx["slice_a"], ctx["slice_b"],
                                 mode=diff_mode, colormap=diff_cmap)

    img_a = _window_to_float01(raw_slice_a, wc_a, ww_a)
    img_b = _window_to_float01(raw_slice_b, wc_b, ww_b)

    fig = Figure(figsize=(13.5, 5))
    axes = fig.subplots(1, 3)
    fig.patch.set_facecolor("white")

    axes[0].imshow(img_a, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Reference · A{chr(10) + meta_a.sequence if meta_a.sequence else ''}",
                       fontsize=11, fontweight="bold")
    axes[1].imshow(img_b, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Test · B{chr(10) + meta_b.sequence if meta_b.sequence else ''}",
                       fontsize=11, fontweight="bold")
    axes[2].imshow(diff_arr)
    axes[2].set_title(f"Δ Error Map · {diff_mode}", fontsize=11, fontweight="bold")

    for ax in axes:
        ax.axis("off")

    if ctx["roi"] is not None:
        from matplotlib.patches import Rectangle
        ys, xs = np.where(ctx["roi"])
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        for ax in axes:
            ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1, y1 - y0 + 1,
                                    fill=False, edgecolor="#35c6e0", linewidth=1.5,
                                    linestyle="--"))

    summary_keys = ["PSNR", "SSIM", "MSE", "NCC", "HFEN"]
    parts = []
    for k in summary_keys:
        v = metrics_res.get(k)
        if v is None:
            continue
        parts.append(f"{k}={'inf' if v == float('inf') else f'{v:.4f}'}")
    subtitle = f"slice {ctx['slice_idx']}/{max(ctx['max_slices'] - 1, 0)}"
    region_bits = []
    if ctx["roi"] is not None:
        region_bits.append("ROI")
    if brain_only:
        region_bits.append("brain mask")
    if region_bits:
        subtitle += f"   |   measured in: {' ∩ '.join(region_bits)}"
    if parts:
        subtitle += "   |   " + "   ".join(parts)
    if ctx["reg_info"] and not ctx["reg_info"].get("error"):
        r = ctx["reg_info"]
        subtitle += (f"   |   registered ({r['transform_type']}): "
                     f"dx={r['translation_x']:.2f} dy={r['translation_y']:.2f} "
                     f"rot={r['rotation_deg']:.2f}°")
    fig.suptitle(subtitle, fontsize=9, y=0.06)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/compare_volume")
def api_compare_volume(
    file1: list[UploadFile] = File(None),
    file2: list[UploadFile] = File(None),
    session_id: str = Form(None),
    normalize: bool = Form(False),
    resample: bool = Form(False),
    brain_only: bool = Form(False),
    do_register: bool = Form(False, alias="register"),
    register_type: str = Form("rigid"),
    roi_x: float = Form(None),
    roi_y: float = Form(None),
    roi_w: float = Form(None),
    roi_h: float = Form(None)
):
    try:
        _validate_roi_and_window(roi_x, roi_y, roi_w, roi_h, None, None)
        arr_a, meta_a, arr_b, meta_b, warnings, session_id = _load_pair(file1, file2, session_id)
    except PipelineError as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.message})

    arr_b_proc = arr_b
    if resample and meta_a.shape != meta_b.shape:
        arr_b_proc = _resampled_b(session_id, arr_b, meta_a.shape)

    if arr_a.ndim != arr_b_proc.ndim:
        return JSONResponse(status_code=400, content={"error": f"NDIM mismatch: {arr_a.ndim} vs {arr_b_proc.ndim}"})

    if arr_a.shape != arr_b_proc.shape:
        return JSONResponse(status_code=400, content={"error": f"Shape mismatch: {arr_a.shape} vs {arr_b_proc.shape}. Try checking 'Resample'."})

    if normalize:
        arr_a_norm, arr_b_norm = normalize_pair(arr_a, arr_b_proc)
        data_range = 1.0
    else:
        arr_a_norm, arr_b_norm = arr_a, arr_b_proc
        range_a = meta_a.intensity_max - meta_a.intensity_min
        range_b = meta_b.intensity_max - meta_b.intensity_min
        data_range = max(range_a, range_b, 1e-9)

    is_3d = arr_a.ndim == 3
    if not is_3d:
        return JSONResponse(status_code=400, content={"error": "Volume comparison requires 3D inputs."})

    max_slices = arr_a.shape[0]

    fovs = [None] * max_slices  # per-slice B field-of-view (None = whole slice)
    sp = meta_a.spacing if meta_a.spacing is not None and len(meta_a.spacing) == 3 else None
    if do_register:
        try:
            tx, _ = register_3d(arr_a_norm, arr_b_norm, transform_type=register_type, spacing=sp)
            fov_vol = apply_transform_3d(
                arr_a_norm, np.ones_like(arr_b_norm, dtype=np.float32), tx, spacing=sp) > 0.5
            arr_b_norm = apply_transform_3d(arr_a_norm, arr_b_norm, tx, spacing=sp)
            fovs = list(fov_vol)
        except Exception as e:
            warnings = list(warnings) + [
                f"3D registration failed ({type(e).__name__}: {e}); volume was scored unregistered."]

    roi = _roi_mask(arr_a_norm.shape, roi_x, roi_y, roi_w, roi_h)
    if roi is not None and not roi.any():
        return JSONResponse(status_code=400,
                            content={"error": "The selected region contains no pixels to measure."})

    slice_metrics_list = []
    slice_masks = []  # exact mask scored per slice (None = whole slice)
    empty_slices = 0
    for slice_idx in range(max_slices):
        slice_a = arr_a_norm[slice_idx]
        slice_b = arr_b_norm[slice_idx]
        mask = _combine_masks(roi, brain_mask(slice_a) if brain_only else None)
        fov = fovs[slice_idx]
        if fov is not None and not fov.all():
            mask = fov if mask is None else (mask & fov)
        if mask is not None and not mask.any():
            empty_slices += 1
            slice_masks.append(np.zeros(slice_a.shape, dtype=bool))
            slice_metrics_list.append({"slice": slice_idx,
                                       "metrics": {name: None for name in METRICS}})
            continue
        slice_masks.append(mask)
        metrics_res = compute_metrics(slice_a, slice_b, data_range=data_range, mask=mask,
                                      spacing=_hd95_spacing(meta_a.spacing, 3, True))
        slice_metrics_list.append({
            "slice": slice_idx,
            "metrics": metrics_res
        })
    if empty_slices:
        warnings = list(warnings) + [
            f"{empty_slices}/{max_slices} slices had no pixels inside the selected region and were skipped."
        ]

    avg_metrics = {}
    metric_names = list(METRICS.keys())
    for name in metric_names:
        valid_vals = [s["metrics"][name] for s in slice_metrics_list if s["metrics"].get(name) is not None]
        if valid_vals:
            avg_metrics[name] = float(np.mean(valid_vals))
        else:
            avg_metrics[name] = None

    mask_3d = None
    if any(m is not None for m in slice_masks):
        mask_3d = np.stack([np.ones(arr_a.shape[1:], dtype=bool) if m is None else m
                            for m in slice_masks])

    volume_metrics = compute_metrics(arr_a_norm, arr_b_norm, data_range=data_range, mask=mask_3d,
                                     spacing=_hd95_spacing(meta_a.spacing, 3))

    return {
        "session_id": session_id,
        "slice_metrics": [{"slice": s["slice"], "metrics": _json_safe_metrics(s["metrics"])}
                          for s in slice_metrics_list],
        "average_metrics": _json_safe_metrics(avg_metrics),
        "volume_metrics": _json_safe_metrics(volume_metrics),
        "hd95_unit": "mm" if sp else "px",
        "warnings": warnings,
        "max_slices": max_slices
    }


# ==========================================
# Single Volume Viewer Integration Endpoints
# ==========================================
VIEWER_SESSIONS = OrderedDict()  # session_id -> (arr, meta)


def _viewer_entry(session_id):
    """Return the (arr, meta) for a viewer or compare session, or None if the id is
    invalid/expired."""
    if not _valid_session_id(session_id):
        return None
    with _MEM_LOCK:
        entry = VIEWER_SESSIONS.get(session_id)
        if entry is not None:
            VIEWER_SESSIONS.move_to_end(session_id)
            return entry
    # Fallback check disk cache from compare session
    cached = _load_session_from_disk(session_id)
    if cached is None:
        return None
    entry = (cached[0], cached[1])
    _lru_put(VIEWER_SESSIONS, session_id, entry)
    return entry


def _viewer_render(arr: np.ndarray, meta: ImageMeta, idx: int, wc: float = None, ww: float = None):
    is_3d = arr.ndim == 3
    idx = max(0, min(idx, (arr.shape[0] - 1) if is_3d else 0))
    slc = arr[idx] if is_3d else arr
    dwc, dww = _slice_window(meta, idx, arr.shape[0] if is_3d else 1)
    uwc = wc if wc is not None else dwc
    uww = ww if ww is not None else dww
    if uwc is None or uww is None or uww <= 1e-6:
        low, high = np.percentile(slc, [0.0, 99.9])
        uwc = float((low + high) / 2.0)
        uww = float(max(high - low, 1.0))
    return idx, uwc, uww, _array_to_b64(_window_to_uint8(slc, uwc, uww))


@app.post("/api/view_open")
def api_view_open(files: list[UploadFile] = File(...)):
    saved = _save_uploads(files)
    if not saved:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    try:
        arr, meta = load_image(saved)
        _validate_loaded_array(arr, "Viewer Dataset")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Could not read dataset: {e}"})
    finally:
        _cleanup_paths(saved)

    sid = uuid.uuid4().hex
    _lru_put(VIEWER_SESSIONS, sid, (arr, meta))
    is_3d = arr.ndim == 3
    max_slices = arr.shape[0] if is_3d else 1
    idx, uwc, uww, b64 = _viewer_render(arr, meta, 0, None, None)
    return {
        "session_id": sid,
        "is_3d": is_3d,
        "max_slices": int(max_slices),
        "current_slice": idx,
        "kind": meta.kind,
        "shape": list(meta.shape),
        "data_min": float(meta.intensity_min),
        "data_max": float(meta.intensity_max),
        "window": {"center": uwc, "width": uww},
        "preview": f"data:image/png;base64,{b64}",
    }


@app.post("/api/view_slice")
def api_view_slice(
    session_id: str = Form(...),
    slice_idx: int = Form(0),
    window_center: float = Form(None),
    window_width: float = Form(None),
):
    entry = _viewer_entry(session_id)
    if entry is None:
        return JSONResponse(status_code=400, content={"error": "Session expired. Upload again."})

    arr, meta = entry
    idx, uwc, uww, b64 = _viewer_render(arr, meta, slice_idx, window_center, window_width)
    return {
        "current_slice": idx,
        "max_slices": int(arr.shape[0]) if arr.ndim == 3 else 1,
        "window": {"center": uwc, "width": uww},
        "preview": f"data:image/png;base64,{b64}",
    }


@app.post("/api/view_pixel")
def api_view_pixel(
    session_id: str = Form(...),
    slice_idx: int = Form(0),
    x: float = Form(...),
    y: float = Form(...),
):
    entry = _viewer_entry(session_id)
    if entry is None:
        return JSONResponse(status_code=400, content={"error": "Session expired."})

    arr, meta = entry
    is_3d = arr.ndim == 3
    idx = max(0, min(slice_idx, (arr.shape[0] - 1) if is_3d else 0))
    slc = arr[idx] if is_3d else arr
    h, w = slc.shape[-2], slc.shape[-1]

    # Convert normalized (0..1) or pixel coordinates
    px = int(np.clip(round(x * w if 0.0 <= x <= 1.0 else x), 0, w - 1))
    py = int(np.clip(round(y * h if 0.0 <= y <= 1.0 else y), 0, h - 1))
    val = float(slc[py, px])

    return {
        "x": px,
        "y": py,
        "value": val,
        "shape": [h, w],
    }


@app.post("/api/view_mpr")
def api_view_mpr(
    session_id: str = Form(...),
    z: int = Form(0),
    x: float = Form(0.5),
    y: float = Form(0.5),
    window_center: float = Form(None),
    window_width: float = Form(None),
):
    entry = _viewer_entry(session_id)
    if entry is None:
        return JSONResponse(status_code=400, content={"error": "Session expired."})

    arr, meta = entry
    if arr.ndim != 3:
        return JSONResponse(status_code=400, content={"error": "MPR orthogonal views require a 3D volume."})

    z_dim, y_dim, x_dim = arr.shape
    cz = int(np.clip(z, 0, z_dim - 1))
    cx = int(np.clip(round(x * x_dim if 0.0 <= x <= 1.0 else x), 0, x_dim - 1))
    cy = int(np.clip(round(y * y_dim if 0.0 <= y <= 1.0 else y), 0, y_dim - 1))

    # 1. Axial slice: (Y, X) at fixed Z
    slc_axial = arr[cz, :, :]

    # 2. Coronal slice: (Z, X) at fixed Y -> flip vertically so Z increases upward (standard radiological display)
    slc_coronal = np.flipud(arr[:, cy, :])

    # 3. Sagittal slice: (Z, Y) at fixed X -> flip vertically so Z increases upward
    slc_sagittal = np.flipud(arr[:, :, cx])

    # Windowing: use explicit window or compute robustly from the volume
    uwc = window_center
    uww = window_width
    if uwc is None or uww is None or uww <= 1e-6:
        low, high = np.percentile(slc_axial, [0.0, 99.9])
        uwc = float((low + high) / 2.0)
        uww = float(max(high - low, 1.0))

    return {
        "is_3d": True,
        "coords": {"z": cz, "y": cy, "x": cx},
        "dims": {"z": z_dim, "y": y_dim, "x": x_dim},
        "value": float(arr[cz, cy, cx]),
        "window": {"center": uwc, "width": uww},
        "axial": f"data:image/png;base64,{_array_to_b64(_window_to_uint8(slc_axial, uwc, uww))}",
        "coronal": f"data:image/png;base64,{_array_to_b64(_window_to_uint8(slc_coronal, uwc, uww))}",
        "sagittal": f"data:image/png;base64,{_array_to_b64(_window_to_uint8(slc_sagittal, uwc, uww))}",
    }



