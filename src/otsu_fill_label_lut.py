"""Segment the DAPI channel and save a colour-coded label image.

Steps:
  1. Threshold the DAPI channel with Otsu's method (True = nucleus).
  2. Fill holes: dark spots fully enclosed by a nucleus become part of it.
  3. Label: every separate object gets its own number (1, 2, 3, ...).
  4. Watershed: split objects that consist of touching nuclei, and give
     every resulting piece its own number.
     Then remove objects that touch the image border (they are cut off, so
     their size and shape would be wrong), and objects smaller than
     MIN_AREA_PIXELS (noise specks and debris, not real nuclei).
  5. Save the watershed label image as a TIFF with a LUT (look-up table)
     that gives every number its own distinct colour.  The pixel size is
     copied, so Fiji shows correct µm measurements and scale bars.
  6. Show the result in napari, using the same colours.

Run it from the top folder of the workshop repository with:

    uv run src/otsu_fill_label_lut.py

The output is saved next to the original image, with "_DAPI_labels_LUT"
added to the name.  Open it in Fiji to see the colours of the saved LUT.
"""

import colorsys
from pathlib import Path

import napari
import numpy as np
import tifffile
from bioio import BioImage
from napari.utils.colormaps import DirectLabelColormap
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label
from skimage.segmentation import clear_border, relabel_sequential, watershed

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
# Watershed: two nucleus centres must be at least this many pixels apart.
# Smaller -> more splitting (risk: one nucleus cut into pieces).
# Larger  -> less splitting (risk: touching nuclei stay merged).
MIN_DISTANCE = 10
# Objects with fewer pixels than this are removed (not counted as nuclei).
MIN_AREA_PIXELS = 50
OUTPUT = INPUT.with_name(INPUT.name.replace(".ome.tif", "_DAPI_watershed_LUT.tif"))

# ---------------------------------------------------------------------------
# 1. Load the DAPI channel and threshold it with Otsu's method
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y

threshold = threshold_otsu(dapi)
mask = dapi > threshold
print(f"Otsu threshold: {threshold:.0f}")

# ---------------------------------------------------------------------------
# 2. Close holes: any background region completely surrounded by nucleus
#    pixels is switched to nucleus.
# ---------------------------------------------------------------------------
filled = binary_fill_holes(mask)
print(f"Pixels added by filling holes: {filled.sum() - mask.sum()}")

# ---------------------------------------------------------------------------
# 3. Label each object with its own number (background stays 0)
# ---------------------------------------------------------------------------
labels_simple = label(filled)
print(f"Number of objects before watershed: {labels_simple.max()}")

# ---------------------------------------------------------------------------
# 4. Watershed, in three steps:
#    a) Distance transform: every nucleus pixel gets a value equal to its
#       distance (in pixels) to the nearest background pixel.  The centre of
#       a nucleus is the "highest" point; two touching nuclei look like two
#       hills with a valley at the place where they touch.
#    b) Markers: find the hill tops (local maxima), at least MIN_DISTANCE
#       pixels apart, and number them.  Each hill top is the seed of one
#       nucleus.
#    c) Watershed: imagine the image upside down (-distance), so hills become
#       basins.  Water rises from every seed; where water from two seeds
#       meets, a dividing line is drawn.  Only nucleus pixels (mask) are
#       flooded.  connectivity=2 counts diagonal neighbours as touching,
#       the same as label() above.
# ---------------------------------------------------------------------------
distance = distance_transform_edt(filled)

peaks = peak_local_max(
    distance, min_distance=MIN_DISTANCE, labels=labels_simple, exclude_border=False
)
markers = np.zeros_like(labels_simple)
markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)  # number the seeds 1, 2, 3...

labels_watershed = watershed(-distance, markers, mask=filled, connectivity=2)
print(f"Number of objects after watershed:  {labels_watershed.max()}")

# ---------------------------------------------------------------------------
# 4b. Remove objects touching the image border.
#     clear_border() sets every object that has at least one pixel on the
#     outer edge of the image to 0 (background).  That leaves gaps in the
#     numbering (e.g. 1, 2, 4, 7...), so relabel_sequential() renumbers the
#     remaining objects 1, 2, 3, ... without gaps.
# ---------------------------------------------------------------------------
labels_cleared = clear_border(labels_watershed)
n_after_border = len(np.unique(labels_cleared)) - 1  # -1: don't count background
print(f"Objects touching the border removed: {labels_watershed.max() - n_after_border}")

# ---------------------------------------------------------------------------
# 4c. Remove objects smaller than MIN_AREA_PIXELS.
#     np.bincount counts how many pixels carry each label number, so
#     sizes[5] is the area of object 5.  Every pixel that belongs to a too
#     small object is set to 0 (background).
# ---------------------------------------------------------------------------
#     (Numbers of objects already removed at the border have size 0; they
#     are skipped by requiring sizes > 0.)
sizes = np.bincount(labels_cleared.ravel())
too_small = (sizes > 0) & (sizes < MIN_AREA_PIXELS)
too_small[0] = False  # label 0 is the background, leave it alone
labels_cleared[too_small[labels_cleared]] = 0
print(f"Objects under {MIN_AREA_PIXELS} pixels removed:    {too_small.sum()}")

labels, _, _ = relabel_sequential(labels_cleared)
n_objects = labels.max()
print(f"Number of objects remaining:        {n_objects}")

# ---------------------------------------------------------------------------
# 5. Build a LUT with a distinct colour for every label number.
#    Hues are spaced with the "golden ratio" trick, so labels with
#    neighbouring numbers get very different colours.  Label 0
#    (background) is black.
# ---------------------------------------------------------------------------
GOLDEN = 0.618033988749895
n_colors = 256  # an 8-bit LUT, as Fiji expects
lut = np.zeros((n_colors, 3), dtype=np.uint8)  # rows: label number; columns: R, G, B
for i in range(1, n_colors):
    hue = (i * GOLDEN) % 1.0
    value = 1.0 if i % 2 else 0.75  # alternate brightness for extra contrast
    lut[i] = np.array(colorsys.hsv_to_rgb(hue, 0.85, value)) * 255

# A LUT has 256 entries, so the labels must fit in 8 bits (0-255).
if n_objects > 255:
    raise SystemExit(f"{n_objects} objects do not fit in an 8-bit LUT (max 255).")
labels_8bit = labels.astype(np.uint8)

# ---------------------------------------------------------------------------
# 6. Save as an ImageJ/Fiji TIFF with the LUT and the pixel size.
#    ImageJ stores pixel size as "resolution" = pixels per µm = 1 / pixel size.
#    min/max 0-255 make Fiji map label number i exactly to LUT entry i.
# ---------------------------------------------------------------------------
#    The LUT is stored as a TIFF "palette": 3 rows (R, G, B) x 256 entries,
#    as 16-bit numbers (0-65535), so we multiply the 0-255 values by 257.
tifffile.imwrite(
    OUTPUT,
    labels_8bit,
    imagej=True,
    photometric="palette",
    colormap=lut.T.astype(np.uint16) * 257,
    resolution=(1 / pixel_size.X, 1 / pixel_size.Y),
    metadata={"axes": "YX", "unit": "um", "min": 0, "max": 255},
)
print("Saved:", OUTPUT)

# ---------------------------------------------------------------------------
# 7. Show in napari with the same colours as the saved LUT.
#    The labels before watershed and the distance map are added as hidden
#    layers: tick their eye icons to compare.
# ---------------------------------------------------------------------------
colors = {0: "transparent", None: "transparent"}
for i in range(1, labels_watershed.max() + 1):
    colors[i] = np.append(lut[i] / 255, 1.0)  # RGB in 0-1, plus opacity 1

scale = (pixel_size.Y, pixel_size.X)  # µm per pixel, (rows, columns)
units = ("µm", "µm")
viewer = napari.Viewer()
viewer.add_image(dapi, name=CHANNEL_NAME, colormap="gray", scale=scale, units=units)
viewer.add_image(distance, name="Distance transform", colormap="magma",
                 scale=scale, units=units, visible=False)
viewer.add_labels(
    labels_simple,
    name="Labels before watershed",
    colormap=DirectLabelColormap(color_dict=colors),
    scale=scale,
    units=units,
    opacity=0.6,
    visible=False,
)
viewer.add_labels(
    labels_watershed,
    name="Watershed labels (incl. border objects)",
    colormap=DirectLabelColormap(color_dict=colors),
    scale=scale,
    units=units,
    opacity=0.6,
    visible=False,
)
viewer.add_labels(
    labels,
    name="Watershed labels, border objects removed",
    colormap=DirectLabelColormap(color_dict=colors),
    scale=scale,
    units=units,
    opacity=0.6,
)
viewer.scale_bar.visible = True
napari.run()
