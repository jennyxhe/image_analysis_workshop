"""Segment the nuclei in the DAPI channel with a simple intensity threshold.

Every pixel brighter than THRESHOLD is called "nucleus", everything else is
"background".  Touching nucleus pixels are then grouped into separate objects
(each nucleus gets its own number, a "label"), and the result is shown in
napari on top of the original DAPI image.

Run it from the top folder of the workshop repository with:

    uv run src/segment_dapi_threshold.py

The label image is also saved next to the original image, with
"_DAPI_labels" added to the name.  The pixel size (µm per pixel) is copied
into the new file so scale bars and measurements stay correct.
"""

from pathlib import Path

import napari
import tifffile
from bioio import BioImage
from skimage.measure import label

# ---------------------------------------------------------------------------
# Settings - change these to try a different threshold or channel
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT = (
    REPO
    / "testImages/3Channel_FF_Corrected_1"
    / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
)
CHANNEL_NAME = "DAPI"
THRESHOLD = 2000  # pixels with intensity > THRESHOLD count as nucleus

OUTPUT = INPUT.with_name(INPUT.name.replace(".ome.tif", "_DAPI_labels.ome.tif"))

# ---------------------------------------------------------------------------
# 1. Open the image and pull out the DAPI channel as a 2D array
# ---------------------------------------------------------------------------
img = BioImage(INPUT)
channel_index = img.channel_names.index(CHANNEL_NAME)
dapi = img.get_image_data("YX", C=channel_index, T=0, Z=0)

pixel_size = img.physical_pixel_sizes  # in µm, has .X and .Y
print(f"DAPI image: {dapi.shape} pixels, intensities {dapi.min()}-{dapi.max()}")
print(f"Pixel size: {pixel_size.X} x {pixel_size.Y} µm")

# ---------------------------------------------------------------------------
# 2. Threshold: a True/False ("binary") image, True where DAPI > THRESHOLD
# ---------------------------------------------------------------------------
mask = dapi > THRESHOLD
print(f"{mask.mean() * 100:.1f}% of pixels are above the threshold {THRESHOLD}")

# ---------------------------------------------------------------------------
# 3. Label: give every separate blob of True pixels its own number (1, 2, 3...)
#    Background stays 0.
# ---------------------------------------------------------------------------
labels = label(mask)
print(f"Found {labels.max()} separate objects")

# ---------------------------------------------------------------------------
# 4. Save the labels, copying the pixel size from the original image
# ---------------------------------------------------------------------------
tifffile.imwrite(
    OUTPUT,
    labels.astype("uint16"),
    ome=True,
    metadata={
        "axes": "YX",
        "PhysicalSizeX": pixel_size.X,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": pixel_size.Y,
        "PhysicalSizeYUnit": "µm",
        "Channel": {"Name": [f"{CHANNEL_NAME} labels"]},
    },
)
print("Saved:", OUTPUT)

# ---------------------------------------------------------------------------
# 5. Show in napari.  "scale" tells napari the pixel size, so the layers are
#    drawn in µm and the scale bar is correct.
# ---------------------------------------------------------------------------
scale = (pixel_size.Y, pixel_size.X)
viewer = napari.Viewer()
viewer.add_image(dapi, name=CHANNEL_NAME, colormap="gray", scale=scale)
viewer.add_labels(labels, name=f"{CHANNEL_NAME} > {THRESHOLD}", scale=scale, opacity=0.5)
viewer.scale_bar.visible = True
viewer.scale_bar.unit = "µm"
napari.run()
