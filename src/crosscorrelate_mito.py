"""Cross-correlate a mitochondria image with a small template (no normalisation).

Run it from the top folder of the workshop repository with:

    uv run src/crosscorrelate_mito.py

It writes testImages/Mito_Cropped_xcorr.tif and prints where the highest
value is.

What cross-correlation does: slide the small template over the image.  At
every position, multiply each template pixel with the image pixel under it
and add everything up.  That sum is the value of the result at that
position (the position of the template's centre).  Where the image looks
like the template, bright template pixels land on bright image pixels and
the sum is large.

Without normalisation, though, the sum also simply gets large wherever the
image is bright - whether or not it has the right shape.  Check whether the
highest value is really at a mitochondrion that looks like the template!
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
IMAGE_FILE = REPO / "testImages/Mito_Cropped.tif"
TEMPLATE_FILE = REPO / "testImages/Mito_template.tif"
OUTPUT = IMAGE_FILE.with_name(f"{IMAGE_FILE.stem}_xcorr.tif")

# Where the template was cut from (centre row, column) - see
# src/make_mito_template.py.  Used only for comparison at the end.
TEMPLATE_ORIGIN = (225, 172)

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
# 2. Cross-correlate
# ---------------------------------------------------------------------------
# The sums get very large (millions to billions), far beyond what 16-bit
# can hold, so we calculate in 64-bit floating point.
# ndimage.correlate places the template centre on each pixel in turn, so
# the result has the same size as the image and lines up with it.
xcorr = ndimage.correlate(image.astype(np.float64),
                          template.astype(np.float64),
                          mode="reflect")

# ---------------------------------------------------------------------------
# 3. Find the highest value
# ---------------------------------------------------------------------------
# argmax finds the position in the flattened array; unravel_index turns it
# back into (row, column).
peak_row, peak_col = np.unravel_index(np.argmax(xcorr), xcorr.shape)
print(f"\nHighest value: {xcorr[peak_row, peak_col]:.4g}")
print(f"  at row {peak_row}, column {peak_col}  (= x {peak_col}, y {peak_row})")
print(f"  = x {peak_col * pixel_size_x:.2f} {unit}, "
      f"y {peak_row * pixel_size_y:.2f} {unit}")

r, c = TEMPLATE_ORIGIN
print(f"\nFor comparison, at the spot the template was cut from "
      f"(row {r}, column {c}): {xcorr[r, c]:.4g}")

# ---------------------------------------------------------------------------
# 4. Save as 32-bit float, copying the pixel size
# ---------------------------------------------------------------------------
tifffile.imwrite(
    OUTPUT,
    xcorr.astype(np.float32),
    imagej=True,
    resolution=(1 / pixel_size_x, 1 / pixel_size_y),
    metadata={"unit": unit, "axes": "YX"},
)
print(f"\nSaved {OUTPUT.name}")
