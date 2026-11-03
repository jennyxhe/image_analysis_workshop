"""Sharpen an actin image by convolving it with a sharpening kernel.

Run it from the top folder of the workshop repository with:

    uv run src/sharpen_actin.py

It writes testImages/Actin_cropped_sharpen.tif.

How sharpening works: the kernel gives the centre pixel a large positive
weight (12) and subtracts its 8 neighbours (-1 each).  In a flat region the
neighbours look like the centre, so not much changes.  At an edge or a thin
filament the centre differs from its neighbours, and that difference gets
amplified - so edges and fine structures stand out more.  (Noise is also a
small "difference", so sharpening makes noise more visible, too.)

The kernel adds up to 12 - 8 = 4.  We divide it by 4 so the weights sum to 1
and the overall brightness of the image stays the same (this is the same
thing ImageJ's Process > Filters > Sharpen does).
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Actin_cropped.tif"
SUFFIX = "sharpen"

KERNEL = np.array([[-1, -1, -1],
                   [-1, 12, -1],
                   [-1, -1, -1]], dtype=np.float64)

# ---------------------------------------------------------------------------
# 1. Read the image and its pixel size
# ---------------------------------------------------------------------------
with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # ImageJ stores "pixels per unit" in the resolution tags as a fraction,
    # and the unit (e.g. micron) in its own metadata.
    x_res = tif.pages[0].tags["XResolution"].value  # (numerator, denominator)
    y_res = tif.pages[0].tags["YResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")

pixel_size_x = x_res[1] / x_res[0]
pixel_size_y = y_res[1] / y_res[0]
print(f"Input: {INPUT.name}  shape {image.shape}  type {image.dtype}")
print(f"Pixel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} {unit}")

# ---------------------------------------------------------------------------
# 2. Normalise the kernel and convolve
# ---------------------------------------------------------------------------
kernel = KERNEL / KERNEL.sum()  # sum is 4 -> weights now add up to 1
print("Normalised kernel:")
print(kernel)

# Do the maths in floating point.  Sharpening can produce values below 0 or
# above the 16-bit maximum next to strong edges, so we clip them back into
# the range the image type can hold before saving.
sharpened = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")
print(f"Value range before clipping: {sharpened.min():.0f} to {sharpened.max():.0f}")
sharpened = np.clip(np.round(sharpened), 0, np.iinfo(image.dtype).max)

# ---------------------------------------------------------------------------
# 3. Save, copying the pixel size
# ---------------------------------------------------------------------------
output = INPUT.with_name(f"{INPUT.stem}_{SUFFIX}.tif")
tifffile.imwrite(
    output,
    sharpened.astype(image.dtype),
    imagej=True,
    resolution=(1 / pixel_size_x, 1 / pixel_size_y),  # copy pixel size
    metadata={"unit": unit, "axes": "YX"},
)
print(f"Saved {output.name}")
