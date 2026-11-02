"""Shrink a color image to about 3% of its pixels and print their R, G, B values.

Run it from the top folder of the workshop repository with:

    uv run src/rgb_sample_table.py

The table is printed in the terminal (scroll up to see all of it).  Napari
then opens in its own window.  Close that window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.  Building the path this
# way means the script works no matter which folder you started it from.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "Cell_cropped_2.tif"

# Which fraction of the pixels to show in the table.
FRACTION = 0.03

# tifffile reads the pixels of a TIFF file into a NumPy array.  A color (RGB)
# image has 3 numbers per pixel, so the array has 3 dimensions:
# (rows, columns, 3).  The last 3 are the red, green and blue values.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()

    # The pixel size is stored as a "resolution": how many pixels fit in one
    # unit (here: one micrometer).  One divided by that gives the size of a
    # single pixel in micrometers.
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator

rows, columns, colors = pixels.shape
total = rows * columns

# To shrink the image we need the picked pixels to form a neat grid: keep
# every "step"-th row and every "step"-th column.  That keeps 1 / step of the
# rows and 1 / step of the columns, so 1 / (step x step) of all pixels.  For 3%
# we need step x step = 1 / 0.03 = 33.3, so step = the square root of 33.3 =
# 5.8, rounded to 6.  A whole number is needed, so we end up close to 3%, not
# exactly on it: 1 / 36 = 2.8%.
step = round(np.sqrt(1 / FRACTION))

# "::step" means "start at 0 and move step at a time to the end".  Doing that
# for rows and columns gives the small image directly: 48 rows x 48 columns,
# still with 3 colors per pixel.
small = pixels[::step, ::step]

# The row and column numbers in the original image that we kept.
kept_rows = np.arange(0, rows, step)
kept_columns = np.arange(0, columns, step)

# Print the table.  ":>6" means "right-align in a space 6 characters wide",
# so the numbers line up in neat columns.  The two loops go through every
# (row, column) pair of the small image.
print(f"Loaded {IMAGE.name}: {rows} rows x {columns} columns, {colors} colors per pixel")
print(f"Number type: {pixels.dtype}  (whole numbers from 0 to 255)\n")
print(f"{'Row':>6} {'Column':>6} {'R':>6} {'G':>6} {'B':>6}")
print(f"{'---':>6} {'------':>6} {'-':>6} {'-':>6} {'-':>6}")
for i, row in enumerate(kept_rows):
    for j, column in enumerate(kept_columns):
        red, green, blue = small[i, j]
        print(f"{row:>6} {column:>6} {red:>6} {green:>6} {blue:>6}")

kept = small.shape[0] * small.shape[1]
print(
    f"\nShowed {kept} of {total} pixels "
    f"(1 of every {step} rows and columns = {100 * kept / total:.2f}%)"
)
print(f"Small image: {small.shape[0]} rows x {small.shape[1]} columns")

# Each pixel of the small image stands for a step x step block of the
# original, so it covers step times as much distance in the sample.
# Multiplying the pixel size by step keeps the scale bar correct: the cell is
# shown at its real size, only with coarser pixels.
small_pixel_size = pixel_size * step
print(f"Pixel size : {pixel_size:.4f} um in the original, {small_pixel_size:.4f} um in the small image")

# Show the small image in Napari, not the original.  "rgb=True" tells Napari
# that the last dimension holds red, green and blue, so it draws one color
# picture instead of 3 separate gray images.  "scale" tells Napari how big each
# pixel is, so the scale bar shows real distances in the sample.
viewer = napari.Viewer()
viewer.add_image(
    small,
    name=f"{IMAGE.stem} (step {step})",
    rgb=True,
    scale=(small_pixel_size, small_pixel_size),
    units=("um", "um"),
)
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
