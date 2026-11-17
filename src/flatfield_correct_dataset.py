"""Flatfield-correct all 25 positions of 3Channel_Raw_25Pos_1 (20x objective),
all three channels (Cy3, FITC, DAPI), and save them as new OME-TIFF files.

For every channel (see Session6_Segmentation2/README.md):

    corrected = (measured - dark) / flatfield_normalized

    dark                 = Flatfield/Dark.tif (camera image without light)
    flatfield_normalized = (Flatfield/20x/<channel>-20x.tif - dark), divided
                           by its own mean, so it is ~1 on average.  Pixels
                           that got less light (corners) are divided by a
                           number < 1 and become brighter; pixels that got
                           more light (centre) become a bit dimmer.  The
                           average intensity of the image stays the same.

The right flatfield is chosen by the channel NAME stored in the image (Cy3,
FITC, DAPI), not by position, so a different channel order cannot mix them up.

Output: folder 3Channel_FF_Corrected_25Pos_1, one file per position, e.g.
    3Channel_FF_Corrected_25Pos_1_Pos-1-002_003.ome.tif
16-bit (like the original), with channel names, pixel size, and the stage
position and exposure time of every channel copied from the original.

Run it from the top folder of the workshop repository with:

    uv run src/flatfield_correct_dataset.py
"""

from pathlib import Path

import numpy as np
import tifffile
from bioio import BioImage

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT_FOLDER = REPO / "3Channel_Raw_25Pos_1"
FLATFIELD_FOLDER = REPO / "Flatfield/20x"   # 20x objective
DARK_FILE = REPO / "Flatfield/Dark.tif"
OUTPUT_FOLDER = REPO / "3Channel_FF_Corrected_25Pos_1"

# ---------------------------------------------------------------------------
# 1. Open the dataset.  Micro-Manager saved 25 files, but the FIRST file
#    describes all 25 positions; bioio calls each position a "scene".
#    (Opening the other 24 files one by one would give position 1 each time!)
# ---------------------------------------------------------------------------
first_file = sorted(INPUT_FOLDER.glob("*.ome.tif"))[0]
img = BioImage(first_file)
channel_names = [str(c) for c in img.channel_names]
pixel_size = img.physical_pixel_sizes  # µm per pixel
print(f"{len(img.scenes)} positions, channels {channel_names}, "
      f"pixel size {pixel_size.X} µm")

# ---------------------------------------------------------------------------
# 2. Load the dark image and one normalised flatfield per channel
# ---------------------------------------------------------------------------
dark = tifffile.imread(DARK_FILE).astype(np.float32)
flatfields = {}
for name in channel_names:
    flat = tifffile.imread(FLATFIELD_FOLDER / f"{name}-20x.tif").astype(np.float32)
    flat = flat - dark
    flatfields[name] = flat / flat.mean()
    print(f"  flatfield {name}: corners get {flatfields[name].min():.0%} "
          f"of the average light, brightest spot {flatfields[name].max():.0%}")

# ---------------------------------------------------------------------------
# 3. Correct every position and save it
# ---------------------------------------------------------------------------
OUTPUT_FOLDER.mkdir(exist_ok=True)
for scene_index, scene in enumerate(img.scenes):
    img.set_scene(scene)
    ome_image = img.ome_metadata.images[scene_index]
    position = ome_image.name.split("MMStack_")[1]          # e.g. Pos-1-002_003
    data = img.get_image_data("CYX", T=0, Z=0)              # shape (3, Y, X)

    corrected = np.empty_like(data)                         # uint16, like the original
    for c, name in enumerate(channel_names):
        fixed = (data[c].astype(np.float32) - dark) / flatfields[name]
        # A 16-bit image can only hold 0-65535.  Values outside that range
        # (e.g. a pixel slightly darker than the dark image) are clipped,
        # and we report how many there were.
        clipped = int(((fixed < 0) | (fixed > 65535)).sum())
        if clipped > 100:
            print(f"    warning: {position} {name}: {clipped} pixels clipped")
        corrected[c] = np.clip(np.round(fixed), 0, 65535).astype(np.uint16)

    # Copy stage position and exposure time of every channel ("plane"), so
    # stitching software still knows where each image was taken.
    planes = ome_image.pixels.planes
    plane_info = {
        "PositionX": [p.position_x for p in planes],
        "PositionXUnit": ["µm"] * len(planes),
        "PositionY": [p.position_y for p in planes],
        "PositionYUnit": ["µm"] * len(planes),
        "PositionZ": [p.position_z for p in planes],
        "PositionZUnit": ["µm"] * len(planes),
        "ExposureTime": [p.exposure_time for p in planes],
        "ExposureTimeUnit": ["ms"] * len(planes),
    }

    out = OUTPUT_FOLDER / f"3Channel_FF_Corrected_25Pos_1_{position}.ome.tif"
    tifffile.imwrite(
        out,
        corrected,
        ome=True,
        compression="zlib",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": pixel_size.X, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size.Y, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": channel_names},
            "Plane": plane_info,
        },
    )
    print(f"[{scene_index + 1:2d}/{len(img.scenes)}] {position}: saved {out.name}")

print("Done. Corrected images are in:", OUTPUT_FOLDER)
