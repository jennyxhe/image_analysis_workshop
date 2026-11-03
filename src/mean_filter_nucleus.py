"""Smooth a noisy nucleus image by convolving it with three averaging kernels.

Run it from the top folder of the workshop repository with:

    uv run src/mean_filter_nucleus.py

It writes three new images into testImages/:

    Nucleus_cropped_noise_mean3x3.tif         (3x3, all ones)
    Nucleus_cropped_noise_mean3x3center4.tif  (3x3, ones with a 4 in the centre)
    Nucleus_cropped_noise_mean7x7.tif         (7x7, all ones)

What convolution does: for every pixel, multiply each neighbour by the
matching number in the kernel, add everything up, and use that as the new
pixel value.  Because each kernel is divided by its own sum, the numbers add
up to 1, so the result is a (weighted) average and the overall brightness of
the image stays the same.
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Nucleus_cropped_noise.tif"

# The kernels before normalisation
center4 = np.ones((3, 3))
center4[1, 1] = 4  # the centre pixel counts 4x as much as each neighbour

KERNELS = {
    "mean3x3": np.ones((3, 3)),
    "mean3x3center4": center4,
    "mean7x7": np.ones((7, 7)),
}

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
# 2. Normalise each kernel, convolve, and save
# ---------------------------------------------------------------------------
for suffix, kernel in KERNELS.items():
    kernel = kernel / kernel.sum()  # weights now add up to 1
    print(f"\n{suffix} kernel (normalised, sum = {kernel.sum():.2f}):")
    print(np.round(kernel, 3))

    # Do the maths in floating point so no precision is lost, then round
    # back to the original 16-bit integer type for saving.
    # mode="reflect" mirrors the image at its border so edge pixels have
    # neighbours to average with.
    smoothed = ndimage.convolve(image.astype(np.float64), kernel, mode="reflect")
    smoothed = np.clip(np.round(smoothed), 0, np.iinfo(image.dtype).max)

    output = INPUT.with_name(f"{INPUT.stem}_{suffix}.tif")
    tifffile.imwrite(
        output,
        smoothed.astype(image.dtype),
        imagej=True,
        resolution=(1 / pixel_size_x, 1 / pixel_size_y),  # copy pixel size
        metadata={"unit": unit, "axes": "YX"},
    )
    print(f"Saved {output.name}")
