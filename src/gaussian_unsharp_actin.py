"""Gaussian blur and unsharp masking of an actin image.

Run it from the top folder of the workshop repository with:

    uv run src/gaussian_unsharp_actin.py

It writes six new images into testImages/:

    Actin_cropped_gauss5x5.tif     blurred with a 5x5 Gaussian kernel
    Actin_cropped_gauss21x21.tif   blurred with a 21x21 Gaussian kernel
    Actin_cropped_unsharp5x5.tif   original minus the 5x5 blur (the "mask")
    Actin_cropped_unsharp21x21.tif original minus the 21x21 blur
    Actin_cropped_sharpened5x5_amount1.0.tif    original + AMOUNT x 5x5 mask
    Actin_cropped_sharpened21x21_amount1.0.tif  original + AMOUNT x 21x21 mask

(The number after "amount" in the file name is the AMOUNT setting below, so
runs with different amounts don't overwrite each other.)

Gaussian blur: a weighted average where close neighbours count a lot and
far-away neighbours count little, following a bell-shaped (Gaussian) curve.
Its width is set by "sigma"; the kernel size just says how far out we bother
to compute it.

Unsharp mask: blurring removes fine detail.  So "original - blurred" is
exactly the fine detail that the blur took away (edges, thin filaments).
With a small blur you keep only the finest detail; with a big blur you also
keep larger structures and mainly remove the smooth background.
The result can be negative (just next to edges), so these images are saved
as 32-bit floating point instead of 16-bit integers.

Sharpening: adding the mask back onto the original boosts exactly that
detail:  sharpened = original + AMOUNT * mask.
AMOUNT = 0 gives back the original; larger values sharpen more strongly
(but also make noise and dark "halos" next to edges more visible).
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Actin_cropped.tif"

# Kernel sizes to use.  For each one, sigma is chosen so that the kernel
# reaches out 2 sigma from the centre (which covers ~95% of the bell curve):
#   5x5   -> radius 2  -> sigma 1
#   21x21 -> radius 10 -> sigma 5
KERNEL_SIZES = [5, 21]

# How much of the mask to add back for the sharpened images.
# Try e.g. 0.5 (gentle), 1.0 (moderate), 2.0 or 3.0 (strong).
AMOUNT = 3.0

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
        data,
        imagej=True,
        resolution=(1 / pixel_size_x, 1 / pixel_size_y),
        metadata={"unit": unit, "axes": "YX"},
    )
    print(f"  saved {output.name}  ({data.dtype}, {data.min():.0f} to {data.max():.0f})")


# ---------------------------------------------------------------------------
# 2. Blur with each kernel size, then subtract the blur from the original
# ---------------------------------------------------------------------------
original = image.astype(np.float32)  # decimals, and negatives allowed

for size in KERNEL_SIZES:
    radius = size // 2
    sigma = radius / 2
    print(f"\n{size}x{size} Gaussian kernel (sigma = {sigma} pixels)")

    # radius= fixes the kernel size to exactly (2*radius + 1) pixels across
    blurred = ndimage.gaussian_filter(original, sigma=sigma, radius=radius,
                                      mode="reflect")
    unsharp = original - blurred

    # The blur stays in the original range, so it can go back to 16-bit.
    save(np.round(blurred).astype(image.dtype), f"gauss{size}x{size}")
    # The difference has negative values, so keep it as 32-bit float.
    save(unsharp, f"unsharp{size}x{size}")

    # Add the mask back to sharpen.  Strong sharpening can push values below
    # 0 or above the 16-bit maximum, so clip them into range before saving.
    sharpened = original + AMOUNT * unsharp
    sharpened = np.clip(np.round(sharpened), 0, np.iinfo(image.dtype).max)
    save(sharpened.astype(image.dtype), f"sharpened{size}x{size}_amount{AMOUNT}")
