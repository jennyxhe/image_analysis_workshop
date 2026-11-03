"""Apply the Laplacian filter to an actin image.

Run it from the top folder of the workshop repository with:

    uv run src/laplacian_actin.py

It writes testImages/Actin_cropped_laplacian.tif.

What the Laplacian does: it compares each pixel with the average of its
4 direct neighbours (up, down, left, right):

    result = (sum of the 4 neighbours) - 4 x (centre pixel)

    - flat region or a smooth, even slope  -> about 0
    - centre brighter than its surroundings (e.g. on a filament) -> negative
    - centre darker than its surroundings (just beside a filament) -> positive

So where Sobel measures how steeply brightness changes (the first
derivative), the Laplacian measures how that change itself changes (the
second derivative).  It highlights thin lines and spots, and the edge of
an object shows up as a "zero crossing" where the sign flips.

Because it compares single neighbouring pixels it is very sensitive to
noise.  The results can be negative, so the image is saved as 32-bit
floating point.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Actin_cropped.tif"
SUFFIX = "laplacian"

# The 4-neighbour Laplacian kernel.  It adds up to 0, which is why flat
# regions give 0.
LAPLACIAN = np.array([[-1,  -1, -1],
                      [-1, 8, -1],
                      [-1,  -1, -1]], dtype=np.float32)

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
# 2. Apply the Laplacian kernel
# ---------------------------------------------------------------------------
original = image.astype(np.float32)  # decimals, and negatives allowed
laplacian = ndimage.correlate(original, LAPLACIAN, mode="reflect")
print(f"Result range: {laplacian.min():.0f} to {laplacian.max():.0f}")

# ---------------------------------------------------------------------------
# 3. Save as 32-bit float, copying the pixel size
# ---------------------------------------------------------------------------
output = INPUT.with_name(f"{INPUT.stem}_{SUFFIX}.tif")
tifffile.imwrite(
    output,
    laplacian,
    imagej=True,
    resolution=(1 / pixel_size_x, 1 / pixel_size_y),
    metadata={"unit": unit, "axes": "YX"},
)
print(f"Saved {output.name}")
