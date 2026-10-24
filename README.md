# Image Metric Studio 🔬📊

An interactive, high-performance image and medical volume comparison studio with real-time multi-metric evaluation, slice-by-slice cine viewer, difference mapping, and automated 2D/3D registration.

Designed for biomedical imaging researchers, computer vision engineers, and radiology AI developers working with **DICOM series**, **NIfTI volumes** (`.nii`, `.nii.gz`), and standard 2D images (**PNG**, **JPG**, **TIFF**, **BMP**).

---

## Key Features

- **16 Quantitative Metrics**:
  - **Fidelity & Error**: PSNR, SSIM, MS-SSIM, MSE, RMSE, MAE, NRMSE, FFT Error.
  - **Correlation & Information**: NCC, Mutual Information (MI), Normalized Mutual Information (NMI).
  - **Perceptual & Structure**: High-Frequency Error Norm (HFEN), Gradient Magnitude Similarity Deviation (GMSD).
  - **Segmentation & Overlap**: Dice Coefficient, Jaccard Index (IoU), 95th Percentile Hausdorff Distance (HD95 in physical millimeters or voxels).
- **Synchronized Multi-Slice Viewer**:
  - Side-by-side synchronized pan, zoom, and crosshair probing.
  - Cine playback with adjustable FPS and keyboard navigation (`←` / `→`).
  - Interactive difference error map with diverging colormaps (`blue-black-red` or absolute error).
- **Voxel Preprocessing & Alignment**:
  - Automatic spatial resampling (Nearest, Linear, BSpline).
  - Robust intensity normalization (percentile-based clipping or min-max).
  - Interactive rigid/affine slice registration powered by **SimpleITK** (Mattes Mutual Information + multi-resolution pyramid).
- **Lightweight & Portable**:
  - Self-contained modern web UI (zero build step / no node_modules required).
  - Can run as a local web app or desktop GUI window via **PyWebView**.
  - Ready for single-binary packaging with **PyInstaller**.

---

## Installation

### 1. Prerequisites
- Python 3.9+ (Python 3.10 or 3.11 recommended)

### 2. Clone Repository
```bash
git clone https://github.com/poachersbz/image-metric-studio.git
cd image-metric-studio
```

### 3. Install Dependencies
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

### Option A: Desktop Window (Recommended)
Launches the integrated native GUI window via `pywebview`:
```bash
python run_gui.py
```

### Option B: Local Web Server
Runs the FastAPI server directly in your browser:
```bash
python -m uvicorn api_main:app --host 127.0.0.1 --port 8002
```
Or simply double-click `run.bat` on Windows. Then navigate to:
```
http://127.0.0.1:8002/static/index.html
```

---

## Project Structure

```
image-metric-studio/
├── api_main.py             # FastAPI backend with comparison endpoints & session cache
├── run_gui.py              # PyWebView desktop launcher
├── run.bat                 # Windows one-click start script
├── requirements.txt        # Python package dependencies
├── ImageMetricStudio.spec  # PyInstaller packaging configuration
├── test_metrics.py         # Self-contained verification & metric test suite
├── test_ui.js              # UI interaction tests
├── core/
│   ├── config.py           # Central tunable parameters & defaults
│   ├── diffmap.py          # Difference map generator & diverging colormaps
│   ├── loaders.py          # Unified DICOM, NIfTI, and 2D image reader
│   ├── metrics.py          # 16 standard image & volume metric implementations
│   ├── preprocess.py       # Intensity normalization & spatial resampling
│   └── registration.py     # SimpleITK Mattes MI rigid/affine slice registration
└── static/
    └── index.html          # Responsive single-page application & slice viewer
```

---

## Testing

Run the included verification suite to validate all metric computations:
```bash
python test_metrics.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
