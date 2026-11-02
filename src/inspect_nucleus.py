"""Look inside an image: the numbers, its size, its number type, then the picture.

Run it from the top folder of the workshop repository with:

    uv run src/inspect_nucleus.py

The script lists the TIFF images in testImages and asks which one to open.
Type its name (with or without ".tif") and press Enter.  The numbers are
printed in the terminal.  Napari then opens in its own window.  Close that
window to end the script.
"""

import sys
from pathlib import Path

import napari
import numpy as np
import tifffile

# Where the images live.  Path(__file__) is this script, ".parent.parent" walks
# up two folders to the top of the workshop repository.  Building the path this
# way means the script works no matter which folder you started it from.
REPO = Path(__file__).resolve().parent.parent
IMAGE_DIR = REPO / "testImages"

# Show which TIFF files there are to choose from.
print(f"TIFF images in {IMAGE_DIR.name}:")
for path in sorted(IMAGE_DIR.glob("*.tif")):
    print(f"  {path.name}")

# input() waits for you to type something and press Enter.  Keep asking until
# the name matches a file that exists.  If you leave out ".tif", it is added.
while True:
    name = input("\nWhich image do you want to open? ").strip()
    if not name.endswith(".tif"):
        name += ".tif"
    IMAGE = IMAGE_DIR / name
    if IMAGE.is_file():
        break
    print(f"There is no file called {name} in {IMAGE_DIR.name}.  Please try again.")

# tifffile reads the pixels of a TIFF file into a NumPy array: a grid of
# numbers, one number per pixel.
with tifffile.TiffFile(IMAGE) as tif:
    pixels = tif.asarray()

    # The pixel size is stored as a "resolution": how many pixels fit in one
    # unit (here: one micrometer).  One divided by that gives the size of a
    # single pixel in micrometers.  Some files do not store it; then we fall
    # back to 1, so the scale bar counts pixels instead of micrometers.
    resolution = tif.pages[0].tags.get("XResolution")
    if resolution is not None:
        numerator, denominator = resolution.value
        pixel_size = denominator / numerator
        unit = (tif.imagej_metadata or {}).get("unit", "micron")
    else:
        pixel_size = 1.0
        unit = "pixel"

# This script works on a single 2D image: one grid of rows and columns.  Some
# files hold more, for example 3 color channels, which adds a third dimension.
if pixels.ndim != 2:
    print(f"\n{IMAGE.name} has {pixels.ndim} dimensions, shape {pixels.shape}.")
    print("This script only handles 2D images (rows x columns).  Please pick another file.")
    sys.exit(1)

# An image is a matrix.  "pixels[:10, :10]" takes rows 0-9 and columns 0-9,
# the top-left corner of the image.
corner = pixels[:10, :10]
print(f"\nLoaded {IMAGE.name}\n")
print("First 10 rows and 10 columns (top-left corner):")
with np.printoptions(linewidth=200, precision=2, suppress=True):
    print(corner)

# "shape" holds the size of each dimension: (rows, columns) for a 2D image.
rows, columns = pixels.shape
print(f"\nNumber of rows   : {rows}")
print(f"Number of columns: {columns}")

# "dtype" is the number type every pixel is stored as.  uint8 and uint16 are
# "unsigned integers": whole numbers from 0 to 255 or 0 to 65535.  float32 is a
# "floating point" number, which can have decimals and be negative.
if np.issubdtype(pixels.dtype, np.integer):
    info = np.iinfo(pixels.dtype)
    print(f"Number type      : {pixels.dtype}  (whole numbers from {info.min} to {info.max})")
else:
    print(f"Number type      : {pixels.dtype}  (numbers with decimals)")
print(f"Pixel size       : {pixel_size:.4f} {unit}")

# Show the 10 x 10 corner in Napari, not the original image.  We cut out
# neighbouring pixels without skipping any, so each pixel still covers the
# same distance in the sample and the pixel size stays the same.  "scale"
# tells Napari how big each pixel is, so the scale bar shows real distances.
viewer = napari.Viewer()
viewer.add_image(
    corner,
    name=f"{IMAGE.stem} top-left 10x10",
    scale=(pixel_size, pixel_size),
    units=("um", "um") if unit != "pixel" else None,
)
viewer.scale_bar.visible = True

# Start the Napari window and wait until you close it.
napari.run()
