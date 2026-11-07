"""Show the mitochondria image next to its Fourier image (log of the amplitudes)
in Napari.

Run it from the top folder of the workshop repository with:

    uv run src/fourier_mito_napari.py

Napari opens in its own window, with the two images side by side.  Close that
window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
from bioio import BioImage

REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Mito_Cropped.tif"

# Read the pixels as a 2D array (rows "Y" x columns "X") and the pixel size.
img = BioImage(IMAGE)
pixels = img.get_image_data("YX").astype(float)
pixel_size = img.physical_pixel_sizes.X  # micrometers per pixel
height, width = pixels.shape

# The 2D Fourier transform: one complex number per pixel.  Each one describes a
# 2D wave (stripes) with a certain direction and spacing; its length is the
# amplitude (how strong those stripes are in the image).
# fftshift moves the "zero frequency" (the overall brightness) from the top
# left corner to the center, which is how Fourier images are normally shown.
spectrum = np.fft.fftshift(np.fft.fft2(pixels))
amplitude = np.abs(spectrum)

# The amplitudes span a huge range: the center is millions of times brighter
# than the edges.  Taking the logarithm squeezes that range so you can see
# everything.  log1p(x) = log(1 + x), which avoids log(0).
log_amplitude = np.log1p(amplitude)

# Each Fourier pixel away from the center is one more stripe per image width.
# In real units, that is a step of 1 / (image width in micrometers).
print(f"Loaded {IMAGE.name}: {width} x {height} pixels, "
      f"{pixel_size:.4f} micrometer per pixel")
print(f"In the Fourier image, one pixel step from the center = "
      f"{1 / (width * pixel_size):.4f} cycles per micrometer (left-right), "
      f"{1 / (height * pixel_size):.4f} (up-down)")
print(f"The edge of the Fourier image = {1 / (2 * pixel_size):.2f} cycles per "
      f"micrometer, i.e. stripes {2 * pixel_size:.2f} micrometer apart "
      f"(the finest detail the pixels can hold)")

# Show both in Napari.  Both layers are shown in pixel units, so they appear
# the same size on screen (the Fourier image has no micrometer scale; its
# "pixels" are frequencies, see the numbers printed above).
viewer = napari.Viewer()
viewer.add_image(pixels, name="Mito image", colormap="gray")
viewer.add_image(log_amplitude, name="log(amplitude) of Fourier image",
                 colormap="magma")

# Grid mode puts the layers next to each other instead of on top of each other.
viewer.grid.enabled = True
viewer.grid.shape = (1, 2)  # 1 row, 2 columns

napari.run()
