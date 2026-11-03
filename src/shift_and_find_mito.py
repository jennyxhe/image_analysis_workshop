"""Shift an image by a known amount, then use cross-correlation to measure it.

Run it from the top folder of the workshop repository with:

    uv run src/shift_and_find_mito.py

It writes two new images into testImages/:

    Mito_Cropped_shifted.tif   the image moved SHIFT_DOWN down, SHIFT_RIGHT right
    Mito_Cropped_shiftxcorr.tif  cross-correlation of shifted vs. original

and prints the shift that the cross-correlation found.

The idea: this time we don't slide a small template, we slide the whole
original image over the shifted image.  For every possible offset we
multiply the overlapping pixels and add them up.  The offset where the two
images line up best gives the highest value - and that offset is how far
the image moved.  This is how drift between time points or channels is
measured and corrected in practice.

Before correlating, we subtract the average brightness from each image.
Otherwise the sum simply grows with the amount of overlap, and the peak
would always sit near "no shift".
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage, signal

REPO = Path(__file__).parent.parent
INPUT = REPO / "testImages/Mito_Cropped.tif"

# How far to move the image, in pixels - try other values!
SHIFT_DOWN = 7    # positive = down,  negative = up
SHIFT_RIGHT = 4   # positive = right, negative = left

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
print(f"Input: {INPUT.name}  shape {image.shape}")
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
    print(f"Saved {output.name}")


# ---------------------------------------------------------------------------
# 2. Make the shifted copy
# ---------------------------------------------------------------------------
# ndimage.shift moves the image by (rows, columns).  The strip that becomes
# empty on the other side is filled with the median (typical background)
# value, so it doesn't look like a black edge.
shifted = ndimage.shift(image, (SHIFT_DOWN, SHIFT_RIGHT), order=0,
                        mode="constant", cval=np.median(image))
print(f"\nShifted the image {SHIFT_DOWN} px down and {SHIFT_RIGHT} px right")
save(shifted.astype(image.dtype), "shifted")

# ---------------------------------------------------------------------------
# 3. Cross-correlate shifted vs. original
# ---------------------------------------------------------------------------
a = shifted.astype(np.float64)
b = image.astype(np.float64)
a -= a.mean()   # remove the average brightness (see explanation above)
b -= b.mean()

# signal.correlate tries every offset of b against a.  method="fft" uses
# the Fourier transform to do this quickly.  mode="same" returns a result
# the same size as the image, where the centre pixel means "no shift".
xcorr = signal.correlate(a, b, mode="same", method="fft")
save(xcorr.astype(np.float32), "shiftxcorr")

# ---------------------------------------------------------------------------
# 4. Find the peak and turn it into a shift
# ---------------------------------------------------------------------------
peak_row, peak_col = np.unravel_index(np.argmax(xcorr), xcorr.shape)
center_row, center_col = xcorr.shape[0] // 2, xcorr.shape[1] // 2
found_down = peak_row - center_row
found_right = peak_col - center_col

print(f"\nPeak of the cross-correlation at row {peak_row}, column {peak_col}")
print(f"Centre of the result (= no shift) at row {center_row}, column {center_col}")
print(f"\nMeasured shift: {found_down} px down, {found_right} px right")
print(f"             = {found_down * pixel_size_y:.2f} {unit} down, "
      f"{found_right * pixel_size_x:.2f} {unit} right")
print(f"Real shift:     {SHIFT_DOWN} px down, {SHIFT_RIGHT} px right")
