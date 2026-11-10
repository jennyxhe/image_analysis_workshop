"""Compare scikit-image's automatic threshold methods on the DAPI channel.

Each method looks at the image's histogram (how many pixels have each
intensity) and picks a threshold using a different rule.  This script prints
the threshold each method picks, and shows:
  - the histogram with every method's threshold drawn as a vertical line
  - the resulting nucleus masks, side by side

Run it from the top folder of the workshop repository with:

    uv run src/compare_auto_thresholds.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from skimage import filters
from skimage.measure import label

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT = (
    REPO
    / "testImages/3Channel_FF_Corrected_1"
    / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
)
CHANNEL_NAME = "DAPI"
MANUAL_THRESHOLD = 2000  # the value we picked by hand, for comparison

# ---------------------------------------------------------------------------
# 1. Load the DAPI channel
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)

# ---------------------------------------------------------------------------
# 2. Ask every global method for a threshold.  Each function takes the image
#    and returns one number.
# ---------------------------------------------------------------------------
methods = {
    "Otsu": filters.threshold_otsu,
    "Li": filters.threshold_li,
    "Yen": filters.threshold_yen,
    "Isodata": filters.threshold_isodata,
    "Triangle": filters.threshold_triangle,
    "Mean": filters.threshold_mean,
    "Minimum": filters.threshold_minimum,
}

thresholds = {"Manual": MANUAL_THRESHOLD}
for name, method in methods.items():
    try:
        thresholds[name] = float(method(dapi))
    except RuntimeError as err:  # "Minimum" fails if the histogram has no clear valley
        print(f"{name:9s}: failed ({err})")

print(f"\n{'Method':9s} {'threshold':>10s} {'% pixels above':>15s} {'# objects':>10s}")
for name, t in thresholds.items():
    mask = dapi > t
    print(f"{name:9s} {t:10.0f} {mask.mean() * 100:14.1f}% {label(mask).max():10d}")

# ---------------------------------------------------------------------------
# 3. Plot the histogram with all thresholds.  The y-axis is logarithmic,
#    because the background has far more pixels than the nuclei.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(dapi.ravel(), bins=512, color="gray", log=True)
colors = plt.cm.tab10(np.arange(len(thresholds)))
for (name, t), c in zip(thresholds.items(), colors):
    ax.axvline(t, color=c, label=f"{name} ({t:.0f})")
ax.set_xlabel("DAPI intensity")
ax.set_ylabel("Number of pixels (log scale)")
ax.set_title("DAPI histogram and automatic thresholds")
ax.legend()
fig.tight_layout()

# ---------------------------------------------------------------------------
# 4. Show the mask produced by each threshold
# ---------------------------------------------------------------------------
n = len(thresholds) + 1
cols = 4
rows = int(np.ceil(n / cols))
fig2, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
axes = axes.ravel()
axes[0].imshow(dapi, cmap="gray", vmax=np.percentile(dapi, 99.5))
axes[0].set_title("DAPI (original)")
for ax, (name, t) in zip(axes[1:], thresholds.items()):
    ax.imshow(dapi > t, cmap="gray")
    ax.set_title(f"{name}: > {t:.0f}")
for ax in axes:
    ax.axis("off")
fig2.tight_layout()

plt.show()
