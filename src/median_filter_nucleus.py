"""Apply three different median filters to a noisy nucleus image.

Run it from the top folder of the workshop repository with:

    uv run src/median_filter_nucleus.py

It writes three new images into testImages/:

    Nucleus_cropped_noise_median3x3cross.tif   (3x3 kernel without corners)
    Nucleus_cropped_noise_median3x3.tif        (full 3x3 kernel)
    Nucleus_cropped_noise_median5x5.tif        (full 5x5 kernel)

What a median filter does: for every pixel, it looks at the pixels in a small
neighbourhood (the "kernel"), sorts their values, and replaces the centre
pixel with the middle value (the median).  A single very bright or very dark
noise pixel is an outlier, so it never ends up in the middle and disappears.
A bigger kernel removes more noise but also blurs away more fine detail.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Nucleus_cropped_noise.tif"

# The kernels ("footprints").  1 = this neighbour is included, 0 = ignored.
KERNELS = {
    # 3x3 without the corners: only the centre plus its 4 direct neighbours
    "median3x3cross": np.array([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]),
    # full 3x3: centre plus all 8 neighbours
    "median3x3": np.ones((3, 3)),
    # full 5x5: centre plus 24 neighbours
    "median5x5": np.ones((5, 5)),
}

# ---------------------------------------------------------------------------
# 1. Read the image and its pixel size
# ---------------------------------------------------------------------------
with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray()
    # ImageJ stores the pixel size as "pixels per unit" in the resolution
    # tags, and the unit (e.g. micron) in its own metadata.
    x_res = tif.pages[0].tags["XResolution"].value  # (numerator, denominator)
    y_res = tif.pages[0].tags["YResolution"].value
    unit = (tif.imagej_metadata or {}).get("unit", "micron")

pixel_size_x = x_res[1] / x_res[0]
pixel_size_y = y_res[1] / y_res[0]
print(f"Input: {INPUT.name}  shape {image.shape}  type {image.dtype}")
print(f"Pixel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} {unit}")

# ---------------------------------------------------------------------------
# 2. Filter with each kernel and save the result
# ---------------------------------------------------------------------------
for suffix, kernel in KERNELS.items():
    filtered = ndimage.median_filter(image, footprint=kernel)

    output = INPUT.with_name(f"{INPUT.stem}_{suffix}.tif")
    tifffile.imwrite(
        output,
        filtered.astype(image.dtype),
        imagej=True,
        resolution=(1 / pixel_size_x, 1 / pixel_size_y),  # copy pixel size
        metadata={"unit": unit, "axes": "YX"},
    )
    print(f"Saved {output.name}  (kernel uses {int(kernel.sum())} pixels)")
