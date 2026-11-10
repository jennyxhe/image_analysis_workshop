"""Clean up an Otsu-thresholded DAPI mask with 2x erosion followed by
2x dilation, then plot a histogram of the sizes of the remaining objects.

Erosion  : every object loses one layer of pixels around its edge.
           Objects that are very small (or thin bridges between objects)
           disappear completely.
Dilation : every object gains one layer of pixels around its edge.
           The surviving objects grow back to roughly their original size,
           but whatever was erased stays erased.
(Erosion followed by the same amount of dilation is called an "opening".)

Run it from the top folder of the workshop repository with:

    uv run src/otsu_erode_dilate_sizes.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")  # let matplotlib and napari share the same window system
import matplotlib.pyplot as plt
import napari
import numpy as np
from bioio import BioImage
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, erosion

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
N_ROUNDS = 2          # number of erosions, and then the same number of dilations
FOOTPRINT = disk(1)   # the neighbourhood used: a pixel plus its 4 direct neighbours
N_BINS = 50           # number of bars in the histogram

# ---------------------------------------------------------------------------
# 1. Load the DAPI channel and threshold it with Otsu's method
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y

threshold = threshold_otsu(dapi)
mask = dapi > threshold
print(f"Otsu threshold: {threshold:.0f}")
print(f"Objects before erosion/dilation: {label(mask).max()}")

# ---------------------------------------------------------------------------
# 2. Erode N_ROUNDS times, then dilate N_ROUNDS times
# ---------------------------------------------------------------------------
cleaned = mask
for i in range(N_ROUNDS):
    cleaned = erosion(cleaned, FOOTPRINT)
for i in range(N_ROUNDS):
    cleaned = dilation(cleaned, FOOTPRINT)

# ---------------------------------------------------------------------------
# 3. Label the remaining objects and measure their sizes (pixels per object)
# ---------------------------------------------------------------------------
labels = label(cleaned)
sizes = np.array([obj.area for obj in regionprops(labels)])
print(f"Objects after {N_ROUNDS}x erosion + {N_ROUNDS}x dilation: {len(sizes)}")
print(f"Smallest object: {sizes.min():.0f} pixels")
print(f"Largest object:  {sizes.max():.0f} pixels")
print(f"Median size:     {np.median(sizes):.0f} pixels")

# ---------------------------------------------------------------------------
# 4. Histogram of the sizes of the remaining objects
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sizes, bins=N_BINS, color="steelblue", edgecolor="black")
ax.set_xlabel("Object size (number of pixels)")
ax.set_ylabel("Number of objects")
ax.set_title(f"After Otsu + {N_ROUNDS}x erosion + {N_ROUNDS}x dilation: "
             f"{len(sizes)} objects")
fig.tight_layout()
plt.show(block=False)  # don't wait here; continue on to napari

# ---------------------------------------------------------------------------
# 5. Show the cleaned mask in napari on top of the original DAPI image.
#    "scale" tells napari the pixel size, so everything is drawn in µm.
# ---------------------------------------------------------------------------
scale = (pixel_size.Y, pixel_size.X)
viewer = napari.Viewer()
viewer.add_image(dapi, name=CHANNEL_NAME, colormap="gray", scale=scale)
# The Otsu mask before clean-up, hidden at first: tick its eye icon to compare.
viewer.add_labels(label(mask), name="Otsu mask (before)", scale=scale,
                  opacity=0.5, visible=False)
# The cleaned mask: each remaining object in its own colour.
viewer.add_labels(labels, name=f"Otsu + {N_ROUNDS}x erode/dilate", scale=scale,
                  opacity=0.5)
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "µm"
napari.run()
