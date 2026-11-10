"""Measure the area and average DAPI brightness of every nucleus, and save
the results as a .csv file (opens in Excel, Google Sheets, Fiji, R, ...).

This script uses the label image made by src/otsu_fill_label_lut.py
(Otsu + fill holes + watershed + border objects removed), so run that first:

    uv run src/otsu_fill_label_lut.py
    uv run src/measure_nuclei.py

The nucleus numbers in the .csv are the same as the label numbers in that
image, so you can find each nucleus back in napari or Fiji.
"""

import csv
from pathlib import Path

import tifffile
from bioio import BioImage
from skimage.measure import regionprops

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT = (
    REPO
    / "testImages/3Channel_FF_Corrected_1"
    / "3Channel_FF_Corrected_1_MMStack_Pos0.ome.tif"
)
LABELS = INPUT.with_name(INPUT.name.replace(".ome.tif", "_DAPI_watershed_LUT.tif"))
CHANNEL_NAME = "DAPI"
OUTPUT = INPUT.with_name(INPUT.name.replace(".ome.tif", "_DAPI_nuclei.csv"))

# ---------------------------------------------------------------------------
# 1. Load the original DAPI image (for brightness) and the label image
#    (which pixels belong to which nucleus)
# ---------------------------------------------------------------------------
if not LABELS.exists():
    raise SystemExit(f"Label image not found: {LABELS}\n"
                     "Run  uv run src/otsu_fill_label_lut.py  first.")

img = BioImage(INPUT)
dapi = img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
pixel_size = img.physical_pixel_sizes  # µm per pixel, has .X and .Y
pixel_area_um2 = pixel_size.X * pixel_size.Y

labels = tifffile.imread(LABELS)
if labels.shape != dapi.shape:
    raise SystemExit("Label image and DAPI image have different sizes.")

# ---------------------------------------------------------------------------
# 2. Measure each nucleus.  regionprops() goes through the label numbers one
#    by one; intensity_image tells it which image to use for brightness.
#      area           = number of pixels in the nucleus
#      intensity_mean = average DAPI value of those pixels (raw camera units)
#      centroid       = centre of the nucleus as (row, column) = (y, x)
# ---------------------------------------------------------------------------
rows = []
for nucleus in regionprops(labels, intensity_image=dapi):
    y, x = nucleus.centroid
    rows.append({
        "nucleus": nucleus.label,
        "area_pixels": int(nucleus.area),
        "area_um2": round(float(nucleus.area * pixel_area_um2), 2),
        "mean_DAPI_intensity": round(float(nucleus.intensity_mean), 1),
        "centroid_x_pixels": round(float(x), 1),
        "centroid_y_pixels": round(float(y), 1),
    })

# ---------------------------------------------------------------------------
# 3. Save as .csv: one row per nucleus, one column per measurement
# ---------------------------------------------------------------------------
with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Measured {len(rows)} nuclei (pixel size {pixel_size.X} µm)")
print("First 5 rows:")
for r in rows[:5]:
    print("  ", r)
print("Saved:", OUTPUT)
