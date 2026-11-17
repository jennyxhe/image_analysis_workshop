"""Find all nuclei with Cellpose in the 25 positions of the dataset, once in the
flatfield-corrected images and once in the raw images, measure their size and
background-corrected summed DAPI intensity, and compare the two.

Input:
  - flatfield corrected: 3Channel_FF_Corrected_25Pos_1/  (made by
                         src/flatfield_correct_dataset.py - run that first)
  - raw:                 3Channel_Raw_25Pos_1/

For every position and both versions:
  1. Cellpose finds the nuclei in the DAPI channel.  Before that, the faint
     DAPI glow of the cytoplasm is removed with a White Top Hat (otherwise
     Cellpose sometimes outlines a whole cell; see src/cellpose_nuclei_cells.py).
     The top hat is ONLY used to find the nuclei; all measurements below use
     the unchanged DAPI image.
  2. Nuclei touching the image border are removed (they are cut off), as
     are objects that are too small (debris) or too big / irregular (whole
     cells instead of nuclei) - see MIN_AREA_UM2, MAX_AREA_UM2, MIN_SOLIDITY.
  3. Local background per nucleus: the median of a ring 10-20 pixels outside
     the nucleus, leaving out pixels within 10 pixels of any other nucleus.
  4. Summed intensity = sum of the nucleus pixels - (area x ring background).
     This is proportional to the amount of DNA: G1 nuclei form one peak,
     G2 nuclei a second peak at about twice that value.

Cellpose takes ~10 s per image on a Mac GPU (~9 minutes for all 50).  The
masks are saved, and reused when you run the script again (delete the
"masks" folder to make Cellpose run again).

Results in 3Channel_FF_Corrected_25Pos_1_results/:
  - nuclei_flatfield.csv, nuclei_raw.csv : one row per nucleus
  - histograms.png                       : comparison plots
  - masks/                               : Cellpose nucleus masks (with pixel size)

Run it from the top folder of the workshop repository with:

    uv run src/cellpose_nuclei_dataset.py
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from bioio import BioImage
from cellpose import core, models
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops
from skimage.morphology import disk, opening
from skimage.segmentation import clear_border

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
RAW_FOLDER = REPO / "3Channel_Raw_25Pos_1"
CORRECTED_FOLDER = REPO / "3Channel_FF_Corrected_25Pos_1"
OUTPUT_FOLDER = REPO / "3Channel_FF_Corrected_25Pos_1_results"
CHANNEL_NAME = "DAPI"
NUCLEUS_DIAMETER = 47   # typical nucleus diameter in pixels (~15 µm)
TOPHAT_RADIUS = 40      # pixels; must be larger than a nucleus
RING_INNER = 10         # ring background: from 10 ...
RING_OUTER = 20         # ... to 20 pixels outside the nucleus
MIN_RING_PIXELS = 50    # fewer usable ring pixels -> nucleus is skipped
# Quality filter: Cellpose also returns some debris (tiny) and, where the
# cytoplasm glows strongly in DAPI, some whole cells (big and irregular).
# Real nuclei here are ~100-500 µm² and smooth ovals.
MIN_AREA_UM2 = 60
MAX_AREA_UM2 = 550
# Solidity = area / area of the "rubber band" stretched around the object
# (convex hull).  Ovals: ~0.98.  Cells with protrusions: much lower.
MIN_SOLIDITY = 0.90

# ---------------------------------------------------------------------------
# 1. Collect the DAPI images of both versions, matched by position name
# ---------------------------------------------------------------------------
# Raw: the first file describes all 25 positions ("scenes").
raw_img = BioImage(sorted(RAW_FOLDER.glob("*.ome.tif"))[0])
pixel_size = raw_img.physical_pixel_sizes  # µm per pixel
raw_dapi = {}
for scene_index, scene in enumerate(raw_img.scenes):
    raw_img.set_scene(scene)
    position = raw_img.ome_metadata.images[scene_index].name.split("MMStack_")[1]
    raw_dapi[position] = raw_img.get_image_data(
        "YX", C=raw_img.channel_names.index(CHANNEL_NAME), T=0, Z=0)

# Flatfield corrected: one file per position, made by flatfield_correct_dataset.py
corrected_dapi = {}
for f in sorted(CORRECTED_FOLDER.glob("*.ome.tif")):
    position = f.name.split("_25Pos_1_")[1].replace(".ome.tif", "")
    img = BioImage(f)
    corrected_dapi[position] = img.get_image_data(
        "YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
if len(corrected_dapi) != len(raw_dapi):
    raise SystemExit(f"Found {len(corrected_dapi)} corrected images, expected "
                     f"{len(raw_dapi)}. Run  uv run src/flatfield_correct_dataset.py  first.")
positions = sorted(raw_dapi)
versions = {"flatfield": corrected_dapi, "raw": raw_dapi}
print(f"{len(positions)} positions, pixel size {pixel_size.X} µm")

OUTPUT_FOLDER.mkdir(exist_ok=True)
(OUTPUT_FOLDER / "masks").mkdir(exist_ok=True)
model = None  # Cellpose is only loaded if a mask has to be made


def find_nuclei(dapi, mask_file):
    """Return the Cellpose nucleus mask, from disk if it was made before."""
    global model
    if mask_file.exists():
        return tifffile.imread(mask_file)
    if model is None:
        model = models.CellposeModel(gpu=core.use_gpu())
        print(f"Cellpose running on: {model.device}")
    # White Top Hat: remove the cytoplasmic DAPI glow (only for finding nuclei).
    # np.minimum prevents the 16-bit subtraction from "wrapping around".
    background = np.minimum(opening(dapi, disk(TOPHAT_RADIUS, decomposition="sequence")),
                            dapi)
    start = time.perf_counter()
    mask, _, _ = model.eval(dapi - background, diameter=NUCLEUS_DIAMETER)
    print(f"    Cellpose: {mask.max()} nuclei in {time.perf_counter() - start:.0f} s")
    tifffile.imwrite(
        mask_file, mask.astype(np.uint16), ome=True, compression="zlib",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": pixel_size.X, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size.Y, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["Cellpose nuclei"]},
        },
    )
    return mask


# ---------------------------------------------------------------------------
# 2. Segment and measure every position, for both versions
# ---------------------------------------------------------------------------
results = {}
for version, images in versions.items():
    print(f"\n=== {version} ===")
    rows = []
    skipped = 0
    filtered = 0
    for position in positions:
        dapi = images[position].astype(np.float64)
        all_nuclei = find_nuclei(images[position],
                                 OUTPUT_FOLDER / "masks" / f"{position}_{version}_nuclei.ome.tif")
        # Only nuclei that are completely inside the image are measured, but
        # ALL nuclei (also those at the border) are avoided by the rings.
        nuclei = clear_border(all_nuclei)
        near_nucleus = distance_transform_edt(all_nuclei == 0) <= RING_INNER
        h, w = dapi.shape

        for nucleus in regionprops(nuclei, intensity_image=dapi):
            area_um2 = nucleus.area * pixel_size.X * pixel_size.Y
            if (not MIN_AREA_UM2 <= area_um2 <= MAX_AREA_UM2
                    or nucleus.solidity < MIN_SOLIDITY):
                filtered += 1   # debris or whole cell, not a nucleus
                continue

            # Ring background, calculated in a small box around the nucleus
            # (bounding box + RING_OUTER) to save time.
            r0, c0, r1, c1 = nucleus.bbox
            r0, c0 = max(r0 - RING_OUTER - 1, 0), max(c0 - RING_OUTER - 1, 0)
            r1, c1 = min(r1 + RING_OUTER + 1, h), min(c1 + RING_OUTER + 1, w)
            distance_to_this = distance_transform_edt(nuclei[r0:r1, c0:c1] != nucleus.label)
            ring = (distance_to_this <= RING_OUTER) & ~near_nucleus[r0:r1, c0:c1]
            if ring.sum() < MIN_RING_PIXELS:
                skipped += 1   # too crowded for a reliable background
                continue
            background = float(np.median(dapi[r0:r1, c0:c1][ring]))
            summed = nucleus.intensity_mean * nucleus.area - nucleus.area * background
            y, x = nucleus.centroid
            rows.append({
                "image": position,
                "nucleus": nucleus.label,
                "area_pixels": int(nucleus.area),
                "area_um2": round(float(area_um2), 1),
                "solidity": round(float(nucleus.solidity), 3),
                "mean_intensity": round(float(nucleus.intensity_mean), 1),
                "ring_background": round(background, 1),
                "summed_intensity_bg_corrected": round(float(summed)),
                "distance_from_center_pixels": round(float(np.hypot(y - h / 2, x - w / 2)), 1),
                "centroid_x_pixels": round(float(x), 1),
                "centroid_y_pixels": round(float(y), 1),
            })
    print(f"{len(rows)} nuclei measured ({filtered} removed by size/solidity "
          f"filter, {skipped} skipped: too few ring pixels)")
    results[version] = rows
    with open(OUTPUT_FOLDER / f"nuclei_{version}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# 3. Plots.  Columns: raw (left) and flatfield corrected (right).
#    Row 1: histogram of the summed intensity (G1 and G2 peaks?)
#    Row 2: summed intensity vs distance from the image centre.  Without
#           flatfield correction, nuclei near the edges look dimmer.
#    Row 3: summed intensity vs nucleus area (G2 nuclei are bigger).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(11, 12))
all_summed = np.concatenate([[r["summed_intensity_bg_corrected"] for r in rows]
                             for rows in results.values()])
bins = np.linspace(0, np.percentile(all_summed, 99.5), 80)

print("\n=== Summary ===")
for col, version in enumerate(["raw", "flatfield"]):
    rows = results[version]
    summed = np.array([r["summed_intensity_bg_corrected"] for r in rows])
    dist = np.array([r["distance_from_center_pixels"] for r in rows])
    area = np.array([r["area_um2"] for r in rows])
    r_dist = np.corrcoef(dist, summed)[0, 1]  # ~0: brightness independent of position

    ax = axes[0, col]
    ax.hist(summed, bins=bins, color="tab:blue")
    ax.set_title(f"{version}: {len(summed)} nuclei")
    ax.set_xlabel("Background-corrected summed DAPI intensity")
    ax.set_ylabel("Number of nuclei")

    ax = axes[1, col]
    ax.scatter(dist, summed, s=4, alpha=0.4)
    ax.set_ylim(bins[0], bins[-1])
    ax.set_title(f"{version}: correlation with distance r = {r_dist:.2f}")
    ax.set_xlabel("Distance of nucleus from image centre (pixels)")
    ax.set_ylabel("Summed DAPI intensity")

    ax = axes[2, col]
    ax.scatter(area, summed, s=4, alpha=0.4)
    ax.set_ylim(bins[0], bins[-1])
    ax.set_title(f"{version}: summed intensity vs size")
    ax.set_xlabel("Nucleus area (µm²)")
    ax.set_ylabel("Summed DAPI intensity")

    print(f"{version:>10}: {len(summed)} nuclei, median area {np.median(area):.0f} µm², "
          f"median summed intensity {np.median(summed):.0f}, "
          f"correlation with distance r = {r_dist:.2f}")

fig.suptitle("Cellpose nuclei, 3Channel_Raw_25Pos_1: raw vs flatfield corrected")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "histograms.png", dpi=150)
print("Results saved in:", OUTPUT_FOLDER)
plt.show()
