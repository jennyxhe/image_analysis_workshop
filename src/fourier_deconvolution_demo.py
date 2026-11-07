"""Undo smoothing by dividing in Fourier space ("deconvolution").

Run it from the top folder of the workshop repository with:

    uv run src/fourier_deconvolution_demo.py

The noisy nucleus image is smoothed with the 3 x 3 "Smooth" kernel (all ones,
divided by 9).  Smoothing is a multiplication in Fourier space, so dividing
the Fourier image of the smoothed image by the Fourier image of the kernel
should give back the original.  The script prints how close it gets, and saves
a picture in Pictures/fourier2d/deconvolution_smooth_3x3_noise1.png.

Set NOISE below to 0 for a "perfect" smoothed image (division works almost
perfectly), or above 0 to add a little camera-like noise after smoothing,
as in a real microscope image (division falls apart).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw into files only, no window on screen
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "testImages" / "Nucleus_cropped_noise.tif"
# Standard deviation of the random noise added after smoothing, in the same
# units as the pixel values (which go up to about 3600 in this image).
NOISE = 10
OUTPUT = (REPO / "Pictures" / "fourier2d"
          / f"deconvolution_smooth_3x3_noise{NOISE:g}.png")

KERNEL = np.ones((3, 3)) / 9  # the Smooth kernel from Session 3

# Dividing by (almost) zero gives enormous numbers.  Where the kernel's
# Fourier amplitude is smaller than this, we do not divide but put 0 instead.
TOO_SMALL = 1e-6


def fourier(image):
    """2D Fourier transform, with the zero frequency moved to the center."""
    return np.fft.fftshift(np.fft.fft2(image))


def kernel_as_image(kernel, shape):
    """Put the small kernel in an image of the given size (zeros elsewhere),
    with the kernel's center pixel at position (0, 0).  The Fourier transform
    needs the kernel to be the same size as the image to divide by it."""
    padded = np.zeros(shape)
    rows, cols = kernel.shape
    padded[:rows, :cols] = kernel
    # Move the center of the kernel to the top left corner.  Pixels that move
    # "off" the top or left edge come back in at the bottom or right edge.
    return np.roll(padded, (-(rows // 2), -(cols // 2)), axis=(0, 1))


image = tifffile.imread(INPUT).astype(float)
kernel_image = kernel_as_image(KERNEL, image.shape)

# Step 1: smooth the image (mode="wrap" matches what the Fourier transform does).
smoothed = ndimage.convolve(image, KERNEL, mode="wrap")

# Step 1b: add a little random noise, like the camera does after the microscope
# has blurred the image.  The seed makes the "random" noise the same every run.
rng = np.random.default_rng(seed=0)
smoothed = smoothed + rng.normal(0, NOISE, size=image.shape)

# Step 2: divide the Fourier image of the smoothed image by the Fourier image
# of the kernel, pixel by pixel, but only where the kernel is not (almost) 0.
smoothed_f = np.fft.fft2(smoothed)
kernel_f = np.fft.fft2(kernel_image)
usable = np.abs(kernel_f) > TOO_SMALL
recovered_f = np.zeros_like(smoothed_f)
recovered_f[usable] = smoothed_f[usable] / kernel_f[usable]

# Step 3: transform back to a normal image.
recovered = np.real(np.fft.ifft2(recovered_f))

lost = (~usable).sum()
print(f"Image size: {image.shape[0]} x {image.shape[1]} pixels")
print(f"Fourier pixels where the kernel is (almost) 0: {lost} "
      f"of {image.size} ({100 * lost / image.size:.1f}%) - this information is lost")
print(f"Smallest kernel amplitude we divided by: {np.abs(kernel_f[usable]).min():.2g}")
print(f"Largest difference smoothed  vs original: {np.abs(smoothed - image).max():.0f}")
print(f"Noise added after smoothing (standard deviation): {NOISE:g}")
print(f"Largest difference recovered vs original: {np.abs(recovered - image).max():.0f}")

# --- The picture --------------------------------------------------------------

smoothed_amp = np.log1p(np.abs(fourier(smoothed)))
recovered_amp = np.log1p(np.abs(fourier(recovered)))
kernel_amp = np.abs(fourier(kernel_image))  # between 0 and 1: no log needed
img_range = dict(vmin=image.min(), vmax=image.max())
# The brightness scale for the Fourier images is set by the ORIGINAL image, so
# that boosted noise in the recovered image shows up as too bright.
image_amp = np.log1p(np.abs(fourier(image)))
amp_range = dict(vmin=smoothed_amp.min(), vmax=image_amp.max())

# Show the kernel enlarged: a small area around the center of the kernel image.
zoom = 5
kernel_zoom = np.fft.fftshift(kernel_image)
center_row, center_col = kernel_zoom.shape[0] // 2, kernel_zoom.shape[1] // 2
kernel_zoom = kernel_zoom[center_row - zoom:center_row + zoom + 1,
                          center_col - zoom:center_col + zoom + 1]

fig, axes = plt.subplots(2, 3, figsize=(13, 9))
panels = [
    (axes[0, 0], smoothed, f"smoothed image + noise ({NOISE:g})", img_range),
    (axes[0, 1], kernel_zoom, "3 x 3 Smooth kernel (enlarged)", {}),
    (axes[0, 2], recovered, "recovered image", img_range),
    (axes[1, 0], smoothed_amp, "Fourier image (amplitudes, log)", amp_range),
    (axes[1, 1], kernel_amp, "Fourier image of the kernel\n(amplitudes, not log)",
     dict(vmin=0, vmax=1)),
    (axes[1, 2], recovered_amp, "Fourier image (amplitudes, log)", amp_range),
]
for ax, data, title, brightness in panels:
    ax.imshow(data, cmap="gray", interpolation="nearest", **brightness)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

# The "equation" signs between the panels, placed in figure coordinates.
for x, y, sign in [(0.672, 0.73, "→"), (0.355, 0.27, "÷"), (0.672, 0.27, "=")]:
    fig.text(x, y, sign, fontsize=30, ha="center", va="center")
fig.text(0.015, 0.73, "real space", rotation=90, fontsize=13, va="center")
fig.text(0.015, 0.27, "Fourier space", rotation=90, fontsize=13, va="center")
fig.suptitle("Undoing the smoothing: division in Fourier space", fontsize=14)
fig.subplots_adjust(left=0.04, right=0.99, bottom=0.03, top=0.9,
                    wspace=0.25, hspace=0.2)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, dpi=100)
print(f"Saved {OUTPUT}")
