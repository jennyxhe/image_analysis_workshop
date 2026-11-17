"""Segment nuclei (DAPI) and whole cells (FITC) with Cellpose, and show the
image plus both masks in napari.

Cellpose (version 4, "Cellpose-SAM") is a trained neural network: you give
it an image, it returns a "mask" (label image) in which every cell has its
own number.  The same model is used twice:
  - nuclei: we give it only the DAPI channel, after removing the faint
            DAPI glow of the cytoplasm with a White Top Hat
  - cells:  we give it the FITC channel (the cytoplasm) together with DAPI.
            The nuclei help Cellpose decide where one cell ends and the next
            begins, and that each cell should have one nucleus.

Run it from the top folder of the workshop repository with:

    uv run src/cellpose_nuclei_cells.py

The first time, Cellpose downloads its model (~1.2 GB).  On a Mac with an
M-chip the GPU is used (see src/check_cellpose_gpu.py).

The masks are saved next to the original image, with "_cellpose_nuclei" and
"_cellpose_cells" added to the name.  The pixel size is copied.
"""

import time
from pathlib import Path

import napari
import numpy as np
import tifffile
from bioio import BioImage
from cellpose import core, models
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
NUCLEUS_CHANNEL = "DAPI"
CYTOPLASM_CHANNEL = "FITC"
# Typical diameter in pixels.  None = let Cellpose work at the original size,
# which is fine for objects of roughly 10-100 pixels.  If objects are cut into
# pieces (too big for Cellpose) or merged (too small), set a number here.
NUCLEUS_DIAMETER = None
CELL_DIAMETER = None
# The cytoplasm of these cells glows faintly in DAPI, and Cellpose (trained
# to find cells!) then sometimes outlines the whole cell instead of the
# nucleus.  We remove that glow first with a White Top Hat (see
# src/tophat_background_dapi.py) with a disk larger than a nucleus.
# Set to None to switch this off and see the difference.
TOPHAT_RADIUS = 40

# ---------------------------------------------------------------------------
# 1. Load the image
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
data = img.get_image_data("CYX", T=0, Z=0)  # all channels, shape (3, Y, X)
names = list(img.channel_names)
dapi = data[names.index(NUCLEUS_CHANNEL)]
fitc = data[names.index(CYTOPLASM_CHANNEL)]
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y
print(f"Image: {dapi.shape} pixels, channels {names}, pixel size {pixel_size.X} µm")

# ---------------------------------------------------------------------------
# 2. Create the Cellpose model, on the GPU if there is one
# ---------------------------------------------------------------------------
use_gpu = core.use_gpu()
model = models.CellposeModel(gpu=use_gpu)
print(f"Cellpose running on: {model.device}")

# ---------------------------------------------------------------------------
# 3. Segment.  model.eval() returns (masks, flows, styles); we only need the
#    masks.  For the cells we stack FITC and DAPI into one 2-channel image;
#    channel_axis=0 tells Cellpose that the first axis holds the channels.
# ---------------------------------------------------------------------------
if TOPHAT_RADIUS:
    # np.minimum: the background may never be brighter than the image,
    # otherwise the 16-bit subtraction would "wrap around" to ~65535.
    background = np.minimum(opening(dapi, disk(TOPHAT_RADIUS, decomposition="sequence")),
                            dapi)
    dapi_for_nuclei = dapi - background
else:
    dapi_for_nuclei = dapi

start = time.perf_counter()
nuclei, _, _ = model.eval(dapi_for_nuclei, diameter=NUCLEUS_DIAMETER)
print(f"Nuclei: {nuclei.max()} found ({time.perf_counter() - start:.0f} s)")

start = time.perf_counter()
cells, _, _ = model.eval(np.stack([fitc, dapi]), channel_axis=0,
                         diameter=CELL_DIAMETER)
print(f"Cells:  {cells.max()} found ({time.perf_counter() - start:.0f} s)")

# ---------------------------------------------------------------------------
# 4. Save both masks, with the pixel size of the original
# ---------------------------------------------------------------------------
for name, mask in [("nuclei", nuclei), ("cells", cells)]:
    out = INPUT.with_name(INPUT.name.replace(".ome.tif", f"_cellpose_{name}.ome.tif"))
    tifffile.imwrite(
        out,
        mask.astype(np.uint16),
        ome=True,
        compression="zlib",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": pixel_size.X, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size.Y, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": [f"Cellpose {name}"]},
        },
    )
    print("Saved:", out)

# ---------------------------------------------------------------------------
# 5. Show in napari: every channel in its own colour, and the two masks.
#    The cell mask is shown as outlines only (contour), so you can still see
#    the nuclei mask and the image underneath.
# ---------------------------------------------------------------------------
scale = (pixel_size.Y, pixel_size.X)
units = ("µm", "µm")
colors = {"DAPI": "blue", "FITC": "green", "Cy3": "magenta"}
viewer = napari.Viewer(title="Cellpose: nuclei and cells")
for name, channel in zip(names, data):
    viewer.add_image(channel, name=name, colormap=colors.get(name, "gray"),
                     blending="additive", scale=scale, units=units,
                     visible=name in (NUCLEUS_CHANNEL, CYTOPLASM_CHANNEL))
viewer.add_labels(nuclei, name="Nuclei (Cellpose)", opacity=0.5,
                  scale=scale, units=units)
cell_layer = viewer.add_labels(cells, name="Cells (Cellpose)", scale=scale, units=units)
cell_layer.contour = 3  # outline thickness in pixels
viewer.scale_bar.visible = True
napari.run()
