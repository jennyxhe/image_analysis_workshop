"""Estimate and remove the background of the DAPI channel with a White Top Hat
filter, and show the original, the background and the corrected image in
napari.

How a White Top Hat works:
  1. "Opening" = erosion followed by dilation with a disk-shaped "footprint".
     Erosion replaces every pixel by the MINIMUM of the disk around it, so
     every bright object smaller than the disk disappears.  Dilation (the
     MAXIMUM) then restores the shape of what is left.  Result: an image
     with everything that is smaller than the disk (the nuclei) removed -
     this is our BACKGROUND estimate.
  2. White Top Hat = original - opening = only the small bright objects,
     sitting on a flat, zero background: the BACKGROUND-CORRECTED image.

The disk must be LARGER than the nuclei (otherwise parts of nuclei are seen
as background), but not much larger (otherwise local background variations
are missed).  Nuclei here are up to ~20 µm (~60 pixels) across.

Run it from the top folder of the workshop repository with:

    uv run src/tophat_background_dapi.py

The background and corrected images are saved next to the original, with
"_DAPI_tophat_background" and "_DAPI_tophat_corrected" added to the name.
The pixel size is copied from the original image.
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from bioio import BioImage
from skimage.morphology import disk, opening

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT = (
    REPO
    / "testImages/3Channel_FF_Corrected_1"
    / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
)
CHANNEL_NAME = "DAPI"
RADIUS_PIXELS = 40  # radius of the disk; 40 pixels = ~13 µm (diameter ~26 µm)

# ---------------------------------------------------------------------------
# 1. Load the DAPI channel
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y
print(f"DAPI: {dapi.shape} pixels, pixel size {pixel_size.X} µm, "
      f"disk radius {RADIUS_PIXELS} px = {RADIUS_PIXELS * pixel_size.X:.1f} µm")

# ---------------------------------------------------------------------------
# 2. White Top Hat, done in two visible steps.
#    decomposition="sequence" builds the big disk from a few small steps:
#    almost the same result, but much faster.  Because it is an
#    approximation, a handful of pixels in the opening can end up slightly
#    brighter than the original.  For 16-bit images, original - opening would
#    then not become negative but "wrap around" to ~65535!  np.minimum()
#    prevents this: the background is never allowed to be brighter than the
#    original.
# ---------------------------------------------------------------------------
footprint = disk(RADIUS_PIXELS, decomposition="sequence")
background = np.minimum(opening(dapi, footprint), dapi)  # the opening
corrected = dapi - background                            # the white top hat

print(f"Background: {background.min()} - {background.max()} "
      f"(mean {background.mean():.0f})")
print(f"Corrected:  {corrected.min()} - {corrected.max()} "
      f"(mean {corrected.mean():.0f})")

# ---------------------------------------------------------------------------
# 3. Save both images, with the pixel size of the original
# ---------------------------------------------------------------------------
for name, data in [("background", background), ("corrected", corrected)]:
    out = INPUT.with_name(INPUT.name.replace(".ome.tif", f"_DAPI_tophat_{name}.ome.tif"))
    tifffile.imwrite(
        out,
        data.astype("uint16"),
        ome=True,
        metadata={
            "axes": "YX",
            "PhysicalSizeX": pixel_size.X, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size.Y, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": [f"{CHANNEL_NAME} top hat {name}"]},
        },
    )
    print("Saved:", out)

# ---------------------------------------------------------------------------
# 4. Show in napari.  Three layers on top of each other: tick the eye icons
#    to switch between them, or press Ctrl+G (Cmd+G on a Mac) to show them
#    side by side ("grid mode").  The background uses its own contrast
#    limits, otherwise it would look black next to the bright nuclei.
# ---------------------------------------------------------------------------
scale = (pixel_size.Y, pixel_size.X)
units = ("µm", "µm")
viewer = napari.Viewer(title="White Top Hat background - DAPI")
viewer.add_image(dapi, name="DAPI original", colormap="gray",
                 scale=scale, units=units)
viewer.add_image(background, name="Background (opening)", colormap="magma",
                 scale=scale, units=units)
viewer.add_image(corrected, name="Background corrected (top hat)", colormap="gray",
                 scale=scale, units=units)
viewer.grid.enabled = True
viewer.scale_bar.visible = True
napari.run()
