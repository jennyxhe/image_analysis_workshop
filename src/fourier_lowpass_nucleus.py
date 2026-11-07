"""Remove noise by keeping only the slow waves: a "low-pass filter" in Fourier space.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_lowpass_nucleus.py

The script calculates the Fourier image of Nucleus_cropped_noise.tif,
multiplies it by a Gaussian "mask" (1 in the center, fading smoothly to 0
further out), and transforms back.  Two images are saved next to the original
in testImages/:

    Nucleus_cropped_noise_fourier_gaussian.tif  the Fourier image (amplitudes, log)
                                                after multiplying by the mask
    Nucleus_cropped_noise_lowpass_gaussian.tif  the image transformed back

When it is done, Napari opens and shows the two images side by side.
Close the Napari window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"

# The width (sigma) of the Gaussian mask, as a fraction of the image size
# (0.1 = 10%).  Try other values: a narrower mask removes more noise, but
# blurs more.
SIGMA_FRACTION = 0.2

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
# (fine details and noise) further out.
fourier = np.fft.fftshift(np.fft.fft2(image))

# Make the Gaussian mask: for every pixel of the Fourier image, calculate its
# distance to the center.  The mask is 1 in the center (waves kept completely)
# and drops smoothly towards 0 further out (waves weakened more and more),
# following a bell curve.  Unlike the circle, there is no sharp edge.
# The image is not exactly square, so use the smaller side for the size.
sigma = SIGMA_FRACTION * min(rows, cols)
row_distance, col_distance = np.ogrid[:rows, :cols]
distance = np.hypot(row_distance - rows // 2, col_distance - cols // 2)
mask = np.exp(-distance**2 / (2 * sigma**2))
print(f"Gaussian mask with sigma = {sigma:.1f} pixels: at {sigma:.0f} pixels from "
      f"the center waves keep 61% of their strength, at {2 * sigma:.0f} pixels 14%, "
      f"at {3 * sigma:.0f} pixels 1%")

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
output = INPUT.with_name(f"{INPUT.stem}_fourier_gaussian.tif")
tifffile.imwrite(output, fourier_view.astype(np.float32), imagej=True)
print(f"Saved {output.name}")

# Saved as a 32-bit float image: the filtered values have fractions, and can be
# slightly negative near sharp edges.
output = INPUT.with_name(f"{INPUT.stem}_lowpass_gaussian.tif")
tifffile.imwrite(output, filtered.astype(np.float32), imagej=True,
                 resolution=resolution, metadata={"unit": unit})
print(f"Saved {output.name}")

# Show both saved images in Napari.  Both layers are shown in pixel units, so
# they appear the same size on screen (the Fourier image has no micrometer
# scale: its "pixels" are frequencies, not distances).
viewer = napari.Viewer()
viewer.add_image(filtered, name="Gaussian lowpass (transformed back)", colormap="gray")
viewer.add_image(fourier_view, name="Fourier image after the Gaussian mask (log)",
                 colormap="magma")

# Grid mode puts the layers next to each other instead of on top of each other.
viewer.grid.enabled = True
viewer.grid.shape = (1, 2)  # 1 row, 2 columns

napari.run()  # keeps the window open until you close it
