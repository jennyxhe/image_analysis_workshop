"""Keep only the fast waves: a soft-edged "high-pass filter" in Fourier space.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_highpass_actin.py

The script calculates the Fourier image of Actin_cropped.tif, multiplies it by
a soft-edged high-pass mask (0 in the center, rising smoothly to 1 further
out), and transforms back.  This removes slow changes in brightness (uneven
background, large blurry areas) and keeps fine details (the actin filaments).
Two images are saved next to the original in testImages/:

    Actin_cropped_fourier_highpass.tif  the Fourier image (amplitudes, log)
                                        after multiplying by the mask
    Actin_cropped_highpass.tif          the image transformed back
"""

from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Actin_cropped.tif"

# The width (sigma) of the Gaussian "hole" in the center of the mask, as a
# fraction of the image size (0.03 = 3%).  Try other values: a larger hole
# removes more of the slow waves, so only the finest details are left.
SIGMA_FRACTION = 0.03

with tifffile.TiffFile(INPUT) as tif:
    image = tif.asarray().astype(float)
    # Keep the pixel size, so the filtered image has the same size in microns.
    page = tif.pages[0]
    resolution = (page.tags["XResolution"].value, page.tags["YResolution"].value)
    unit = tif.imagej_metadata.get("unit", "micron")

rows, cols = image.shape
print(f"Read {INPUT.name}: {rows} x {cols} pixels")

# The Fourier image, with the zero frequency (the average brightness) moved to
# the center, so that the slow waves are in the middle and the fast waves
# (fine details) further out.
fourier = np.fft.fftshift(np.fft.fft2(image))

# Make the mask.  First a Gaussian bell: 1 in the center, dropping smoothly to
# 0 further out (this is the low-pass mask).  Then "1 minus" it turns it
# around: 0 in the center (slow waves removed), rising smoothly to 1 further
# out (fast waves kept).  No sharp edge, so no ripples ("ringing").
# The image is not exactly square, so use the smaller side for the size.
sigma = SIGMA_FRACTION * min(rows, cols)
row_distance, col_distance = np.ogrid[:rows, :cols]
distance = np.hypot(row_distance - rows // 2, col_distance - cols // 2)
mask = 1 - np.exp(-distance**2 / (2 * sigma**2))
print(f"High-pass mask with sigma = {sigma:.1f} pixels: waves at {sigma:.0f} pixels "
      f"from the center keep 39% of their strength, at {2 * sigma:.0f} pixels 86%, "
      f"at {3 * sigma:.0f} pixels 99%")
print(f"(A wave {sigma:.0f} pixels from the center has about {sigma:.0f} stripes "
      f"across the image, i.e. one every {min(rows, cols) / sigma:.0f} pixels)")

# Multiply each wave by the mask value at its position.
filtered_fourier = fourier * mask

# Transform back.  ifftshift undoes the fftshift from above.  The result has
# tiny imaginary parts left over from rounding; "real" drops them.
filtered = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_fourier)))
print(f"Pixel values: original {image.min():.0f} - {image.max():.0f}, "
      f"filtered {filtered.min():.0f} - {filtered.max():.0f}")

# The Fourier image to look at: the amplitudes, on a log scale (otherwise only
# the very bright center is visible).  Pixels in Fourier space are frequencies,
# not distances, so the pixel size of the image does not apply to this one.
fourier_view = np.log1p(np.abs(filtered_fourier))
output = INPUT.with_name(f"{INPUT.stem}_fourier_highpass.tif")
tifffile.imwrite(output, fourier_view.astype(np.float32), imagej=True)
print(f"Saved {output.name}")

# Saved as a 32-bit float image: the center of the mask is 0, so the average
# brightness is removed too.  The filtered values are spread around 0, with
# many negative values, which a normal 16-bit image cannot hold.
output = INPUT.with_name(f"{INPUT.stem}_highpass.tif")
tifffile.imwrite(output, filtered.astype(np.float32), imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {output.name}")
