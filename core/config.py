"""
Central configuration for the image-comparison tool.

All tunable parameters live here so the FastAPI/web UI (and any future
alternate UI) can share a single source of truth. Nothing in this
module imports any UI library.
"""

# ---- Difference map ----
DIFFMAP_COLORMAP = "blue_black_red"  # custom colormap (blue if lower, red if higher, black as base)
DIFFMAP_MODE = "signed"        # "absolute" (|A-B|) or "signed" (A-B, use a diverging cmap)
DIFFMAP_AMPLIFY = 1.0          # multiply the error before colormapping (visual boost)
DIFFMAP_PERCENTILE_CLIP = 99   # clip error map at this percentile for display contrast

# ---- Metrics ----
DEFAULT_METRICS = ["PSNR", "SSIM", "MSE", "NCC", "HFEN"]  # pre-ticked in UI
MI_BINS = 256                  # histogram bins for MI / NMI
HFEN_SIGMA = 1.5               # LoG sigma for HFEN
SSIM_WIN_SIZE = 11             # odd window size for SSIM (11 + gaussian_weights/sigma=1.5 matches the standard Wang et al. formulation)
SSIM_GAUSSIAN_SIGMA = 1.5      # stddev of the Gaussian SSIM window

# ---- Normalization / matching ----
MATCH_MODE = "independent"     # "independent" or "shared" range when normalizing
RANGE_MATCH_TOLERANCE = 0.05   # ranges within this fraction are treated as matching
NORMALIZE_METHOD = "percentile"  # "percentile" (clip to PERCENTILE_LOW/HIGH, robust to outlier voxels) or "minmax" (raw min/max)
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0

# ---- SSIM background handling ----
# When no explicit mask is passed to ssim()/ms_ssim(), restrict the score to
# pixels above this fraction of data_range (foreground only) so a large,
# near-identical black background doesn't inflate the score.
SSIM_AUTO_BACKGROUND_THRESHOLD = 0.05
