"""Detect edges in an actin image with the Sobel x and Sobel y kernels.

Run it from the top folder of the workshop repository with:

    uv run src/sobel_actin.py

It writes three new images into testImages/:

    Actin_cropped_sobelx.tif      change in brightness from left to right
    Actin_cropped_sobely.tif      change in brightness from top to bottom
    Actin_cropped_sobelmag.tif    gradient magnitude (edge strength)

How it works: the Sobel x kernel subtracts the pixels on the left from the
pixels on the right.  In a flat region they cancel out (result ~0); at a
vertical edge the result is large - positive where the image gets brighter
to the right, negative where it gets darker.  Sobel y does the same from
top to bottom, so it finds horizontal edges.

The gradient magnitude combines both:  sqrt(sobel_x**2 + sobel_y**2).
It is large at any edge, whatever its direction, and never negative.

Sobel results can be negative and are not limited to the 16-bit range, so
all three images are saved as 32-bit floating point.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Actin_cropped.tif"

# Left column -1, right column +1; the middle row counts double.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
# Same idea, top row -1, bottom row +1 (it is SOBEL_X turned on its side).
SOBEL_Y = SOBEL_X.T

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


def save(data, suffix):
    """Save an image next to the input, copying the pixel size."""
    output = INPUT.with_name(f"{INPUT.stem}_{suffix}.tif")
    tifffile.imwrite(
        output,
        data.astype(np.float32),
        imagej=True,
        resolution=(1 / pixel_size_x, 1 / pixel_size_y),
        metadata={"unit": unit, "axes": "YX"},
    )
    print(f"  saved {output.name}  ({data.min():.0f} to {data.max():.0f})")


# ---------------------------------------------------------------------------
# 2. Apply both kernels and combine them
# ---------------------------------------------------------------------------
original = image.astype(np.float32)  # decimals, and negatives allowed

# ndimage.correlate places the kernel on the image exactly as written above
# (ndimage.convolve would flip it first, which swaps the + and - signs).
sobel_x = ndimage.correlate(original, SOBEL_X, mode="reflect")
sobel_y = ndimage.correlate(original, SOBEL_Y, mode="reflect")
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

save(sobel_x, "sobelx")
save(sobel_y, "sobely")
save(magnitude, "sobelmag")
