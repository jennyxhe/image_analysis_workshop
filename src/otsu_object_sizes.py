"""Threshold the DAPI channel with Otsu's method, count the objects, and plot
a histogram of their sizes (in pixels).

Run it from the top folder of the workshop repository with:

    uv run src/otsu_object_sizes.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bioio import BioImage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

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
N_BINS = 50  # number of bars in the histogram

# ---------------------------------------------------------------------------
# 1. Load the DAPI channel
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y

# ---------------------------------------------------------------------------
# 2. Let Otsu's method pick the threshold, then make the True/False mask
# ---------------------------------------------------------------------------
threshold = threshold_otsu(dapi)
mask = dapi > threshold
print(f"Otsu threshold: {threshold:.0f}")

# ---------------------------------------------------------------------------
# 3. Label the objects (each touching blob gets its own number) and count them
# ---------------------------------------------------------------------------
labels = label(mask)
objects = regionprops(labels)  # one entry per object, with its measurements
print(f"Total number of objects (any size): {len(objects)}")

# ---------------------------------------------------------------------------
# 4. Size of each object = its number of pixels ("area" in regionprops)
# ---------------------------------------------------------------------------
sizes = np.array([obj.area for obj in objects])
pixel_area_um2 = pixel_size.X * pixel_size.Y
print(f"Smallest object: {sizes.min():.0f} pixels")
print(f"Largest object:  {sizes.max():.0f} pixels")
print(f"Median size:     {np.median(sizes):.0f} pixels "
      f"(= {np.median(sizes) * pixel_area_um2:.1f} µm²)")
print(f"Objects of only 1 pixel: {(sizes == 1).sum()}")

# ---------------------------------------------------------------------------
# 5. Histogram of the object sizes
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sizes, bins=N_BINS, color="steelblue", edgecolor="black")
ax.set_xlabel("Object size (number of pixels)")
ax.set_ylabel("Number of objects")
ax.set_title(f"DAPI objects after Otsu threshold ({threshold:.0f}): "
             f"{len(objects)} objects")
fig.tight_layout()
plt.show()
