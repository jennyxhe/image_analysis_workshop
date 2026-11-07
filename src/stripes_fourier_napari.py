"""Two 64 x 64 stripe images (30 and 40 vertical stripes) and their Fourier
images, shown in Napari.

Run it from the top folder of the workshop repository with:

    uv run src/stripes_fourier_napari.py

Napari opens with four images in a 2 x 2 grid:

    top row    : 30 stripes            | its Fourier image
    bottom row : 40 stripes            | its Fourier image

Hover the mouse over the bright dots in a Fourier image: the bottom left of
the Napari window shows the position of the pixel under the mouse.
"""

import napari
import numpy as np

SIZE = 64  # pixels, both width and height
STRIPE_COUNTS = [30, 40]


def stripes(n_stripes, size=SIZE):
    """An image with n_stripes vertical stripes across its width.

    The brightness follows a smooth wave (a cosine) from left to right, going
    from 0 (dark) to 1 (bright) and back n_stripes times.  Every row is the
    same, so the stripes are vertical.
    """
    x = np.arange(size)                                   # 0, 1, ..., 63
    row = 0.5 + 0.5 * np.cos(2 * np.pi * n_stripes * x / size)
    return np.tile(row, (size, 1))                        # repeat for all rows


def log_fourier_amplitude(image):
    """Fourier image with the zero frequency in the center, shown as the
    logarithm of the amplitudes (log1p(x) = log(1 + x) avoids log(0))."""
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    return np.log1p(np.abs(spectrum))


viewer = napari.Viewer()
center = SIZE // 2
for n in STRIPE_COUNTS:
    image = stripes(n)
    fourier = log_fourier_amplitude(image)
    viewer.add_image(image, name=f"{n} stripes", colormap="gray",
                     interpolation2d="nearest")
    viewer.add_image(fourier, name=f"Fourier of {n} stripes", colormap="magma",
                     interpolation2d="nearest")

    # Where are the bright dots?  Skip the center (the average brightness).
    fourier_row = fourier[center].copy()
    fourier_row[center] = 0
    peaks = np.flatnonzero(fourier_row > 0.5 * fourier_row.max()) - center
    print(f"{n} stripes: bright dots at {peaks.tolist()} pixels left/right "
          f"of the center")

print(f"\nA {SIZE}-pixel image can hold at most {SIZE // 2} stripes "
      f"(one bright + one dark pixel per stripe).")

# Show the four images in a 2 x 2 grid instead of on top of each other.
# Layers fill the grid from left to right, top to bottom, in the order added.
viewer.grid.enabled = True
viewer.grid.shape = (2, 2)

napari.run()
