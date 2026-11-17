"""Analyse the nuclei (DAPI channel) in all 25 positions of 3Channel_Raw_25Pos_1,
once WITHOUT and once WITH flatfield correction, and compare the results.

Flatfield correction (see Session6_Segmentation2/README.md):

    corrected = (measured - dark) / flatfield_normalized

    dark                 = Flatfield/Dark.tif (camera image with no light)
    flatfield_normalized = (Flatfield/20x/DAPI-20x.tif - dark), divided by
                           its own mean, so it is ~1 on average:
                           > 1 where the illumination is bright (centre),
                           < 1 where it is dim (edges and corners).

For both versions ("raw" and "flatfield") the script, per image:
  1. Smooths the DAPI channel a little (Gaussian) to reduce noise.
  2. Thresholds with Otsu (per image, or one threshold for the whole
     dataset - see THRESHOLD_MODE below), fills holes, and splits touching
     nuclei with a watershed.  Nuclei touching the border and tiny objects
     are removed.
  3. Measures the background in two ways:
     a) "image background": one value per image, the median of all pixels
        that are well away from any nucleus.
     b) "ring background": one value per nucleus, the median of a ring of
        pixels 10-20 pixels outside that nucleus.  Pixels close to other
        nuclei (within 10 pixels) are left out of the ring.
  4. Measures every nucleus: area, and the "background corrected summed
     intensity" = sum of all pixel values in the nucleus minus
     (area x background), for both kinds of background.  This number is
     proportional to the amount of DNA, so G1 nuclei should form one peak
     and G2 nuclei a second peak at about twice that value.

Results (saved in the folder 3Channel_Raw_25Pos_1_results):
  - nuclei_raw.csv, nuclei_flatfield.csv : one row per nucleus
  - histograms.png : histograms of the summed intensity, raw vs flatfield,
                     image vs ring background, plus summed intensity vs
                     distance from the image centre
  - images/ : the flatfield-corrected DAPI images and the label images of
              both versions (pixel size copied from the original images)

Run it from the top folder of the workshop repository with:

    uv run src/flatfield_nuclei_analysis.py

It takes a minute or two.  At the end, one position is opened in napari so
you can compare raw vs corrected images and their segmentations.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import tifffile
from bioio import BioImage
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border, relabel_sequential, watershed

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT_FOLDER = REPO / "3Channel_Raw_25Pos_1"
FLATFIELD_FILE = REPO / "Flatfield/20x/DAPI-20x.tif"
DARK_FILE = REPO / "Flatfield/Dark.tif"
OUTPUT_FOLDER = REPO / "3Channel_Raw_25Pos_1_results"
CHANNEL_NAME = "DAPI"

# "per_image": calculate an Otsu threshold for every image separately.
# "global":    calculate one Otsu threshold from all 25 images together.
THRESHOLD_MODE = "per_image"
SMOOTH_SIGMA = 2        # Gaussian smoothing (pixels) before thresholding
MIN_DISTANCE = 15       # watershed: nucleus centres at least this far apart (pixels)
MIN_AREA_PIXELS = 300   # smaller objects are debris, not nuclei
# Background pixels must be more than RING_INNER pixels away from ANY nucleus.
# The local "ring" background of a nucleus uses the pixels between RING_INNER
# and RING_OUTER pixels from its edge (minus those too close to other nuclei).
RING_INNER = 10
RING_OUTER = 20
MIN_RING_PIXELS = 50    # fewer usable ring pixels -> use the image background
SAVE_IMAGES = True      # save corrected DAPI images and label images
SHOW_POSITION = 12      # which position (0-24) to open in napari; None = don't

# ---------------------------------------------------------------------------
# 1. Load the dark image and the flatfield, and normalise the flatfield
# ---------------------------------------------------------------------------
dark = tifffile.imread(DARK_FILE).astype(np.float32)
flat = tifffile.imread(FLATFIELD_FILE).astype(np.float32) - dark
flat_norm = flat / flat.mean()
print(f"Normalised flatfield: min {flat_norm.min():.2f}, max {flat_norm.max():.2f} "
      "(1.0 = average illumination)")

# Micro-Manager saved 25 files, but the FIRST file contains the description
# of all 25 positions.  bioio calls each position a "scene"; we open the first
# file and step through its scenes (opening the other files one by one would
# give you position 1 25 times!).
first_file = sorted(INPUT_FOLDER.glob("*.ome.tif"))[0]
img = BioImage(first_file)
positions = []   # names of the positions, e.g. "Pos-1-002_003"
raw_images = []  # the DAPI channel of every position, kept in memory
for scene_index, scene in enumerate(img.scenes):
    img.set_scene(scene)
    positions.append(img.ome_metadata.images[scene_index].name.split("MMStack_")[1])
    raw_images.append(
        img.get_image_data("YX", C=img.channel_names.index(CHANNEL_NAME), T=0, Z=0)
    )
pixel_size = img.physical_pixel_sizes  # µm per pixel, the same for all images
print(f"Read {len(raw_images)} positions from {INPUT_FOLDER.name}")
print(f"Pixel size: {pixel_size.X} µm")


def make_version(raw, version):
    """Return the DAPI image as floating point numbers, either as recorded
    ("raw") or flatfield corrected ("flatfield")."""
    raw = raw.astype(np.float32)
    if version == "raw":
        return raw
    return (raw - dark) / flat_norm


def segment(smoothed, threshold):
    """Otsu mask -> fill holes -> watershed -> remove border/small objects.
    Returns a label image: 0 = background, 1, 2, 3 ... = nuclei."""
    mask = binary_fill_holes(smoothed > threshold)
    distance = distance_transform_edt(mask)
    peaks = peak_local_max(distance, min_distance=MIN_DISTANCE,
                           labels=label(mask), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-distance, markers, mask=mask, connectivity=2)
    labels = clear_border(labels)
    sizes = np.bincount(labels.ravel())
    too_small = sizes < MIN_AREA_PIXELS
    too_small[0] = False
    labels[too_small[labels]] = 0
    labels, _, _ = relabel_sequential(labels)
    return labels


def save_tif(path, data, channel_name):
    """Save a 2D image as OME-TIFF with the pixel size of the original."""
    tifffile.imwrite(
        path, data, ome=True, compression="zlib",
        metadata={
            "axes": "YX",
            "PhysicalSizeX": pixel_size.X, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": pixel_size.Y, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": [channel_name]},
        },
    )


OUTPUT_FOLDER.mkdir(exist_ok=True)
(OUTPUT_FOLDER / "images").mkdir(exist_ok=True)
center_y, center_x = np.array(raw_images[0].shape) / 2
results = {}   # version -> list of rows (one row per nucleus)
napari_layers = {}

for version in ["raw", "flatfield"]:
    print(f"\n=== {version} ===")
    # -----------------------------------------------------------------------
    # 2. Smooth every image, and choose the threshold(s)
    # -----------------------------------------------------------------------
    images = [make_version(r, version) for r in raw_images]
    smoothed = [gaussian(im, sigma=SMOOTH_SIGMA, preserve_range=True) for im in images]
    if THRESHOLD_MODE == "global":
        # Every 4th pixel of every image is plenty to compute the threshold.
        all_pixels = np.concatenate([s[::4, ::4].ravel() for s in smoothed])
        thresholds = [threshold_otsu(all_pixels)] * len(images)
        print(f"Global Otsu threshold: {thresholds[0]:.0f}")
    else:
        thresholds = [threshold_otsu(s) for s in smoothed]
        print(f"Per-image Otsu thresholds: {min(thresholds):.0f} - {max(thresholds):.0f}")

    rows = []
    for i, (position, im, sm, thr) in enumerate(
            zip(positions, images, smoothed, thresholds)):
        labels = segment(sm, thr)

        # -------------------------------------------------------------------
        # 3a. Image background: median of all pixels more than RING_INNER
        #     pixels away from anything above the threshold.
        #     distance_transform_edt gives every pixel its distance to the
        #     nearest above-threshold pixel.  We use "anything above the
        #     threshold" rather than the label image, so nuclei removed at the
        #     border and small debris are also avoided.
        # -------------------------------------------------------------------
        distance_to_any_nucleus = distance_transform_edt(~(sm > thr))
        near_nucleus = distance_to_any_nucleus <= RING_INNER
        background = float(np.median(im[~near_nucleus]))

        # -------------------------------------------------------------------
        # 4. Measure every nucleus
        # -------------------------------------------------------------------
        ring_image = np.zeros_like(labels)  # rings, for display in napari
        h, w = labels.shape
        for nucleus in regionprops(labels, intensity_image=im):
            y, x = nucleus.centroid
            summed = nucleus.intensity_mean * nucleus.area

            # ---------------------------------------------------------------
            # 3b. Local ring background.  To save time we only look at a small
            #     box around the nucleus (its bounding box + RING_OUTER).
            #     In that box: distance of every pixel to THIS nucleus;
            #     the ring = pixels at most RING_OUTER away, but not "near"
            #     any nucleus (which also removes the inner 0-RING_INNER zone
            #     of this nucleus, and other nuclei plus their inner zones).
            # ---------------------------------------------------------------
            r0, c0, r1, c1 = nucleus.bbox
            r0, c0 = max(r0 - RING_OUTER - 1, 0), max(c0 - RING_OUTER - 1, 0)
            r1, c1 = min(r1 + RING_OUTER + 1, h), min(c1 + RING_OUTER + 1, w)
            this_nucleus = labels[r0:r1, c0:c1] == nucleus.label
            distance_to_this = distance_transform_edt(~this_nucleus)
            ring = (distance_to_this <= RING_OUTER) & ~near_nucleus[r0:r1, c0:c1]
            if ring.sum() >= MIN_RING_PIXELS:
                ring_background = float(np.median(im[r0:r1, c0:c1][ring]))
            else:
                ring_background = background  # too crowded: fall back
            ring_image[r0:r1, c0:c1][ring] = nucleus.label

            rows.append({
                "image": position,
                "nucleus": nucleus.label,
                "area_um2": round(float(nucleus.area * pixel_size.X * pixel_size.Y), 1),
                "background": round(background, 1),
                "ring_background": round(ring_background, 1),
                "ring_pixels": int(ring.sum()),
                "mean_intensity": round(float(nucleus.intensity_mean), 1),
                "summed_intensity_bg_corrected":
                    round(float(summed - nucleus.area * background)),
                "summed_intensity_ring_corrected":
                    round(float(summed - nucleus.area * ring_background)),
                "distance_from_center_pixels":
                    round(float(np.hypot(y - center_y, x - center_x)), 1),
                "centroid_x_pixels": round(float(x), 1),
                "centroid_y_pixels": round(float(y), 1),
            })
        print(f"  {position}: threshold {thr:.0f}, background {background:.0f}, "
              f"{labels.max()} nuclei")

        if SAVE_IMAGES:
            stem = OUTPUT_FOLDER / "images" / position
            save_tif(f"{stem}_{version}_labels.ome.tif", labels.astype(np.uint16),
                     f"{CHANNEL_NAME} labels ({version})")
            if version == "flatfield":
                save_tif(f"{stem}_DAPI_flatfield.ome.tif", im.astype(np.float32),
                         f"{CHANNEL_NAME} flatfield corrected")
        if i == SHOW_POSITION:
            napari_layers[version] = (im, labels, ring_image)

    results[version] = rows
    with open(OUTPUT_FOLDER / f"nuclei_{version}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

# ---------------------------------------------------------------------------
# 5. Plot.  Columns: raw / flatfield.  Rows:
#      1. histogram, one background value per image
#      2. histogram, local ring background per nucleus
#      3. ring-corrected summed intensity vs distance from the image centre
#         (without flatfield correction, nuclei near the edges look dimmer,
#         which smears out the G1 and G2 peaks)
# ---------------------------------------------------------------------------
METHODS = {
    "image background": "summed_intensity_bg_corrected",
    "ring background": "summed_intensity_ring_corrected",
}
fig, axes = plt.subplots(3, 2, figsize=(11, 12))
all_values = np.concatenate([[r[column] for r in rows]
                             for rows in results.values()
                             for column in METHODS.values()])
bins = np.linspace(0, np.percentile(all_values, 99.5), 80)

print("\n=== Summary ===")
for col, (version, rows) in enumerate(results.items()):
    dist = np.array([r["distance_from_center_pixels"] for r in rows])
    for row, (method, column) in enumerate(METHODS.items()):
        summed = np.array([r[column] for r in rows])
        ax = axes[row, col]
        ax.hist(summed, bins=bins, color="tab:blue")
        ax.set_title(f"{version}, {method}: {len(summed)} nuclei")
        ax.set_xlabel("Background-corrected summed DAPI intensity")
        ax.set_ylabel("Number of nuclei")
        # Correlation: close to 0 means brightness does not depend on position.
        r = np.corrcoef(dist, summed)[0, 1]
        print(f"{version:>10}, {method:>16}: median summed intensity "
              f"{np.median(summed):.0f}, correlation with distance r = {r:.2f}")
    ax = axes[2, col]
    ax.scatter(dist, summed, s=4, alpha=0.4)  # summed = ring-corrected values
    ax.set_ylim(bins[0], bins[-1])
    ax.set_xlabel("Distance of nucleus from image centre (pixels)")
    ax.set_ylabel("Summed DAPI intensity (ring bg)")
    ax.set_title(f"{version}, ring background: correlation with distance r = {r:.2f}")
    fallback = sum(r["ring_pixels"] < MIN_RING_PIXELS for r in rows)
    print(f"{'':>10}  ({fallback} nuclei had too few ring pixels -> image background)")

fig.suptitle(f"3Channel_Raw_25Pos_1, DAPI (threshold: {THRESHOLD_MODE})")
fig.tight_layout()
fig.savefig(OUTPUT_FOLDER / "histograms.png", dpi=150)
print("Results saved in:", OUTPUT_FOLDER)

# ---------------------------------------------------------------------------
# 6. Show one position in napari: raw vs flatfield corrected, with labels
# ---------------------------------------------------------------------------
plt.show(block=False)
if SHOW_POSITION is not None:
    scale = (pixel_size.Y, pixel_size.X)
    viewer = napari.Viewer(title=positions[SHOW_POSITION])
    viewer.add_image(flat_norm, name="Normalised flatfield", colormap="magma",
                     scale=scale, units=("µm", "µm"), visible=False)
    for version, (im, labels, rings) in napari_layers.items():
        viewer.add_image(im, name=f"DAPI {version}", colormap="gray",
                         scale=scale, units=("µm", "µm"), visible=(version == "raw"))
        viewer.add_labels(labels, name=f"Labels {version}", scale=scale,
                          units=("µm", "µm"), opacity=0.5, visible=(version == "raw"))
        viewer.add_labels(rings, name=f"Background rings {version}", scale=scale,
                          units=("µm", "µm"), opacity=0.5, visible=(version == "raw"))
    viewer.scale_bar.visible = True
    napari.run()
