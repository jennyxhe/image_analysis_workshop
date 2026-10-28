"""Look inside an image: the numbers, its size, its number type, then the picture.

Run it from the top folder of the workshop repository with:

    uv run src/inspect_nucleus.py

The numbers are printed in the terminal.  Napari then opens in its own window.
Close that window to end the script.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile

# Where the image lives.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.  Building the path this
# way means the script works no matter which folder you started it from.
REPO = Path(__file__).resolve().parent.parent
IMAGE = REPO / "testImages" / "nucleus_cropped_1.tif"

# tifffile reads the pixels of a TIFF file into a NumPy array: a grid of
# numbers, one number per pixel.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()

    # The pixel size is stored as a "resolution": how many pixels fit in one
    # unit (here: one micrometer).  One divided by that gives the size of a
    # single pixel in micrometers.
    numerator, denominator = tif.pages[0].tags["XResolution"].value
    pixel_size = denominator / numerator
    unit = (tif.imagej_metadata or {}).get("unit", "micron")

# An image is a matrix.  "pixels[:10, :10]" takes rows 0-9 and columns 0-9,
# the top-left corner of the image.
print(f"Loaded {IMAGE.name}\n")
print("First 10 rows and 10 columns (top-left corner):")
with np.printoptions(linewidth=200):
    print(pixels[:10, :10])

# "shape" holds the size of each dimension: (rows, columns) for a 2D image.
rows, columns = pixels.shape
print(f"\nNumber of rows   : {rows}")
print(f"Number of columns: {columns}")

# "dtype" is the number type every pixel is stored as.  uint16 means
# "unsigned integer, 16 bits": whole numbers from 0 to 65535.
info = np.iinfo(pixels.dtype)
print(f"Number type      : {pixels.dtype}  (whole numbers from {info.min} to {info.max})")
print(f"Pixel size       : {pixel_size:.4f} {unit}")

# Show the image in Napari.  "scale" tells Napari how big each pixel is, so the
# scale bar shows real distances in the sample.
viewer = napari.Viewer()
viewer.add_image(pixels, name=IMAGE.stem, scale=(pixel_size, pixel_size), units=("um", "um"))
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
