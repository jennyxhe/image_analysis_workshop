"""Rotate the actin image by 45 degrees and save it as a new image.

Run it from the top folder of the workshop repository with:

    uv run src/rotate_actin.py

It writes testImages/Actin_cropped_rot45.tif, with the same pixel size as the
original.  Look at it in Napari, or at its Fourier image by changing the file
name in src/fourier_actin_napari.py.
"""

from pathlib import Path

import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"
ANGLE = 45  # degrees, counter-clockwise
OUTPUT = INPUT.with_name(f"{INPUT.stem}_rot{ANGLE}.tif")

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
print(f"Input : {INPUT.name}  shape {image.shape}  type {image.dtype}")
print(f"Pixel size: {pixel_size_x:.4f} x {pixel_size_y:.4f} {unit}")

# ---------------------------------------------------------------------------
# 2. Rotate
# ---------------------------------------------------------------------------
# After turning, the new pixel positions fall between the old pixels, so each
# new value is worked out from its neighbours ("interpolation").
#   order=1    : straight-line (bilinear) interpolation.  It never creates
#                values brighter or darker than the original pixels.
#   reshape=True: make the image larger so the turned corners still fit.
#                 The empty corners that appear are filled with 0 (black).
rotated = ndimage.rotate(image, ANGLE, reshape=True, order=1, cval=0)
print(f"Output: {OUTPUT.name}  shape {rotated.shape}  type {rotated.dtype}")

# ---------------------------------------------------------------------------
# 3. Save, copying the pixel size
# ---------------------------------------------------------------------------
# Turning the image does not change how large one pixel is, so the original
# pixel size is copied to the new file.
tifffile.imwrite(
    OUTPUT,
    rotated,
    imagej=True,
    resolution=(1 / pixel_size_x, 1 / pixel_size_y),
    metadata={"unit": unit, "axes": "YX"},
)
print(f"Saved {OUTPUT.name}")
