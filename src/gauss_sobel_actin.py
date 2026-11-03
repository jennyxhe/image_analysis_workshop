"""Blur an actin image with a 5x5 Gaussian, then detect edges with Sobel.

Run it from the top folder of the workshop repository with:

    uv run src/gauss_sobel_actin.py

It writes three new images into testImages/:

    Actin_cropped_gauss5x5_sobelx.tif    Sobel x of the blurred image
    Actin_cropped_gauss5x5_sobely.tif    Sobel y of the blurred image
    Actin_cropped_gauss5x5_sobelmag.tif  gradient magnitude of the blurred image

Why blur first?  Sobel looks at differences between neighbouring pixels.
Noise is also a difference between neighbouring pixels, so Sobel on its own
turns every bit of noise into a little "edge".  A light Gaussian blur first
smooths away the pixel-to-pixel noise while keeping the real structures, so
the edge image becomes cleaner.  Compare the result with
Actin_cropped_sobelmag.tif from src/sobel_actin.py.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Actin_cropped.tif"

# Same 5x5 Gaussian as in src/gaussian_unsharp_actin.py
GAUSS_RADIUS = 2  # kernel is 2*2 + 1 = 5 pixels across
GAUSS_SIGMA = 1.0

# Left column -1, right column +1; the middle row counts double.
SOBEL_X = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = SOBEL_X.T  # the same kernel turned on its side

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
# 2. Blur, then apply both Sobel kernels and combine them
# ---------------------------------------------------------------------------
original = image.astype(np.float32)  # decimals, and negatives allowed

# Keep the blurred image as decimals (no rounding) before taking edges.
blurred = ndimage.gaussian_filter(original, sigma=GAUSS_SIGMA,
                                  radius=GAUSS_RADIUS, mode="reflect")

# ndimage.correlate places the kernel on the image exactly as written above.
sobel_x = ndimage.correlate(blurred, SOBEL_X, mode="reflect")
sobel_y = ndimage.correlate(blurred, SOBEL_Y, mode="reflect")
magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

save(sobel_x, "gauss5x5_sobelx")
save(sobel_y, "gauss5x5_sobely")
save(magnitude, "gauss5x5_sobelmag")
