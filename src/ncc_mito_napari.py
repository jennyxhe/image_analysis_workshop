"""Normalised cross-correlation of a mitochondria image with a template,
then find the best matches and show them in napari.

Run it from the top folder of the workshop repository with:

    uv run src/ncc_mito_napari.py

It writes testImages/Mito_Cropped_ncc.tif, prints the peaks it found, and
opens napari.  Close the napari window to end the script.

Normalised cross-correlation (NCC): like plain cross-correlation, the
template slides over the image.  But at every position, both the template
and the piece of image under it are first adjusted to have average 0 and
the same "spread" of values.  So only the *shape* of the brightness pattern
counts, not how bright it is.  The result is always between -1 and +1:

    +1  the image looks exactly like the template there
     0  no resemblance
    -1  the exact opposite (bright where the template is dark)

This fixes the problem of plain cross-correlation, where bright regions win
just because they are bright.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from skimage.feature import match_template, peak_local_max

REPO = Path(__file__).parent.parent
IMAGE_FILE = REPO / "testImages/Mito_Cropped.tif"
TEMPLATE_FILE = REPO / "testImages/Mito_template.tif"
OUTPUT = IMAGE_FILE.with_name(f"{IMAGE_FILE.stem}_ncc.tif")

# Peak-finding settings - try changing these
THRESHOLD = 0.9     # only keep matches at least this good (max is 1.0)
MIN_DISTANCE = 4    # peaks must be at least this many pixels apart

# ---------------------------------------------------------------------------
# 1. Read the image, the template, and the pixel size
# ---------------------------------------------------------------------------
with tifffile.TiffFile(IMAGE_FILE) as tif:
    image = tif.asarray()
    # ImageJ stores "pixels per unit" in the resolution tags as a fraction,
    # and the unit (e.g. micron) in its own metadata.
    x_res = tif.pages[0].tags["XResolution"].value  # (numerator, denominator)
    y_res = tif.pages[0].tags["YResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")

template = tifffile.imread(TEMPLATE_FILE)

pixel_size_x = x_res[1] / x_res[0]
pixel_size_y = y_res[1] / y_res[0]
print(f"Image:    {IMAGE_FILE.name}  shape {image.shape}")
print(f"Template: {TEMPLATE_FILE.name}  shape {template.shape}")
print(f"Pixel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} {unit}")

# ---------------------------------------------------------------------------
# 2. Normalised cross-correlation
# ---------------------------------------------------------------------------
# pad_input=True makes the result the same size as the image, with each
# value placed at the centre of the template - so it lines up with the image.
ncc = match_template(image.astype(np.float32), template.astype(np.float32),
                     pad_input=True)
print(f"NCC values range from {ncc.min():.2f} to {ncc.max():.2f}")

tifffile.imwrite(
    OUTPUT,
    ncc.astype(np.float32),
    imagej=True,
    resolution=(1 / pixel_size_x, 1 / pixel_size_y),  # copy pixel size
    metadata={"unit": unit, "axes": "YX"},
)
print(f"Saved {OUTPUT.name}")

# ---------------------------------------------------------------------------
# 3. Find the peaks
# ---------------------------------------------------------------------------
# peak_local_max finds pixels that are higher than everything around them,
# keeps only those above THRESHOLD, and if two peaks are closer than
# MIN_DISTANCE it keeps only the higher one.  Result: (row, column) pairs.
peaks = peak_local_max(ncc, min_distance=MIN_DISTANCE, threshold_abs=THRESHOLD)

print(f"\nFound {len(peaks)} peaks with NCC > {THRESHOLD}, "
      f"at least {MIN_DISTANCE} pixels apart:")
for row, col in peaks:
    print(f"  row {row:3d}, column {col:3d}   NCC = {ncc[row, col]:.3f}")

# ---------------------------------------------------------------------------
# 4. Show everything in napari
# ---------------------------------------------------------------------------
# scale= tells napari the pixel size, so the ruler/coordinates are in microns.
# The points are given in pixels, so they get the same scale to line up.
scale = (pixel_size_y, pixel_size_x)
units = ("um", "um")
viewer = napari.Viewer()
viewer.add_image(image, name="Mito_Cropped", scale=scale, units=units,
                 colormap="gray")
viewer.add_image(ncc, name="NCC", scale=scale, units=units, colormap="magma",
                 visible=False)  # tick the eye icon in napari to show it
viewer.add_points(
    peaks,
    name=f"matches (NCC > {THRESHOLD})",
    scale=scale,
    units=units,
    size=template.shape[0],   # circle as wide as the template
    face_color="transparent",
    border_color="yellow",
    border_width=0.1,         # fraction of the circle size
)
viewer.scale_bar.visible = True
napari.run()
