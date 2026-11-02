"""Crop a small region out of the DAPI channel and save it as its own TIFF.

Run it from the top folder of the workshop repository with:

    uv run src/crop_dapi.py

The cropped image is saved next to the original image, with "_DAPI_crop"
added to the name.  The pixel size (µm per pixel) is copied into the new file,
so Fiji / Napari will still show correct scale bars and measurements.
"""

from pathlib import Path

import tifffile
from bioio import BioImage

# ---------------------------------------------------------------------------
# Settings - change these to crop a different region or channel
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT = (
    REPO
    / "testImages/3Channel_FF_Corrected_1"
    / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
)
CHANNEL_NAME = "DAPI"

# Corners of the crop as (x, y) in pixels, i.e. (column, row), like in Fiji.
# x runs left -> right, y runs top -> bottom.
CORNER_1 = (100, 60)
CORNER_2 = (135, 90)

OUTPUT = INPUT.with_name(INPUT.name.replace(".ome.tif", "_DAPI_crop.ome.tif"))

# ---------------------------------------------------------------------------
# 1. Open the image and find the DAPI channel by its name
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
print("Channels in file:", img.channel_names)
channel_index = img.channel_names.index(CHANNEL_NAME)

# get_image_data("YX", C=...) gives us a plain 2D array (rows, columns)
# for just that one channel.
dapi = img.get_image_data("YX", C=channel_index, T=0, Z=0)
print("Full DAPI image shape (rows, columns):", dapi.shape)

# ---------------------------------------------------------------------------
# 2. Crop.  NumPy arrays are indexed [row, column] = [y, x], so y comes first.
# ---------------------------------------------------------------------------
x_start, x_end = sorted([CORNER_1[0], CORNER_2[0]])
y_start, y_end = sorted([CORNER_1[1], CORNER_2[1]])
crop = dapi[y_start:y_end, x_start:x_end]
print(f"Cropped x {x_start}-{x_end}, y {y_start}-{y_end} -> shape {crop.shape}")

# ---------------------------------------------------------------------------
# 3. Save, copying the pixel size so the physical scale is preserved
# ---------------------------------------------------------------------------
pixel_size = img.physical_pixel_sizes  # in µm, has .X and .Y
print(f"Pixel size: {pixel_size.X} x {pixel_size.Y} µm")

tifffile.imwrite(
    OUTPUT,
    crop,
    ome=True,
    metadata={
        "axes": "YX",
        "PhysicalSizeX": pixel_size.X,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size.Y,
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": [CHANNEL_NAME]},
    },
)
print("Saved:", OUTPUT)
