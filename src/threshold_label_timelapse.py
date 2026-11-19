"""Find and label the nuclei in every frame of a time-lapse movie with
conventional segmentation (no machine learning).

Input:   Tracking/P31-crop-crop.tif           (86 frames, 1 minute apart)
Output:  Tracking/P31-crop-crop_nuclei_labels.tif

Steps, done separately for every frame:
  1. Gaussian blur     - smooths away pixel noise, so the outlines are clean
                         instead of ragged.
  2. White Top Hat     - removes the uneven background (this movie is darker
                         at the bottom than at the top).  It subtracts
                         everything that is wider than a nucleus.
  3. Li threshold      - automatically picks the grey value that separates
                         nuclei from background.  Otsu's method was tried
                         first, but the few very bright nuclei pull its
                         threshold up so far that dim nuclei are missed.
  4. Clean up          - fill holes inside nuclei, cut thin bridges between
                         nuclei ("opening"), remove tiny specks.
  5. Watershed         - nuclei that touch become one blob after
                         thresholding.  The "distance transform" gives each
                         pixel its distance to the background, so every
                         nucleus gets a peak in its middle.  Each peak is a
                         seed, and the watershed grows the seeds outward
                         until they meet: touching nuclei are split.

Nuclei at the image border are kept: for tracking, we want to follow nuclei
that move in or out of the field of view.

The output is a "label image" with the same number of frames as the movie.
In each frame, background pixels are 0 and all pixels of one nucleus share
the same number (1, 2, 3, ...).  Numbers are given per frame: number 5 in
frame 1 is NOT necessarily the same nucleus as number 5 in frame 2.
Connecting nuclei between frames is what tracking (btrack) does next.

The output keeps the metadata of the input movie (pixel size if there is
one, and the time between frames), so napari and Fiji overlay it correctly.

Run it from the top folder of the workshop repository with:

    uv run src/threshold_label_timelapse.py

Then compare the result to the movie in napari (drag both files in, and
right-click the labels layer > "Convert to labels" to see colored nuclei).
"""

from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_li
from skimage.measure import label
from skimage.morphology import disk, remove_small_objects, white_tophat
from skimage.segmentation import watershed

# ---------------------------------------------------------------------------
# Settings (all in pixels; nuclei in this movie are ~24 pixels across)
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
INPUT_FILE = REPO / "Tracking" / "P31-crop-crop.tif"
OUTPUT_FILE = INPUT_FILE.with_name(INPUT_FILE.stem + "_nuclei_labels.tif")
BLUR_SIGMA = 1.5        # strength of the Gaussian blur
TOPHAT_RADIUS = 20      # must be larger than the radius of a nucleus
OPENING_RADIUS = 2      # cuts bridges thinner than ~2x this between nuclei
MIN_NUCLEUS_AREA = 50   # smaller objects are removed as noise/debris
SEED_DISTANCE = 7       # minimum distance between two nucleus centres.
                        # Lower it if touching nuclei stay merged; raise it
                        # if single nuclei are cut into pieces.


def segment_frame(image):
    """Return a label image of the nuclei in one 2D frame."""
    smooth = gaussian(image.astype(np.float32), sigma=BLUR_SIGMA, preserve_range=True)
    smooth = white_tophat(smooth, disk(TOPHAT_RADIUS))
    nuclei = smooth > threshold_li(smooth)
    nuclei = ndi.binary_fill_holes(nuclei)
    nuclei = ndi.binary_opening(nuclei, disk(OPENING_RADIUS))
    nuclei = remove_small_objects(nuclei, max_size=MIN_NUCLEUS_AREA - 1)

    # Watershed: one seed in the centre of each nucleus, then grow.
    distance = ndi.distance_transform_edt(nuclei)
    peaks = peak_local_max(distance, min_distance=SEED_DISTANCE,
                           labels=label(nuclei), exclude_border=False)
    seeds = np.zeros(image.shape, dtype=np.int32)
    seeds[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-distance, seeds, mask=nuclei).astype(np.uint16)


# ---------------------------------------------------------------------------
# 1. Read the movie and its metadata
# ---------------------------------------------------------------------------
with tifffile.TiffFile(INPUT_FILE) as tif:
    movie = tif.asarray()                      # shape: (frames, rows, columns)
    imagej_info = tif.imagej_metadata or {}
    resolution_tag = tif.pages[0].tags.get("XResolution")
    resolution = resolution_tag.value if resolution_tag else None

# Pixel size: ImageJ stores it as "pixels per unit" in the resolution tag.
pixel_size = None
if resolution and resolution != (1, 1):
    pixel_size = resolution[1] / resolution[0]
    print(f"Pixel size: {pixel_size:.4f} {imagej_info.get('unit', 'micron')}")
else:
    print("Note: this movie has no pixel size in its metadata; sizes are in pixels.")
print(f"Movie: {movie.shape[0]} frames of {movie.shape[1]} x {movie.shape[2]} pixels")

# ---------------------------------------------------------------------------
# 2. Segment every frame
# ---------------------------------------------------------------------------
labels = np.zeros(movie.shape, dtype=np.uint16)
for frame in range(movie.shape[0]):
    labels[frame] = segment_frame(movie[frame])
    print(f"  frame {frame + 1:3d}/{movie.shape[0]}: {labels[frame].max()} nuclei")

# ---------------------------------------------------------------------------
# 3. Save the labels as an ImageJ-style TIFF with the same metadata
# ---------------------------------------------------------------------------
metadata = {
    "axes": "TYX",
    "finterval": imagej_info.get("finterval", 1),
    "tunit": imagej_info.get("tunit", "min"),
}
save_options = {}
if pixel_size is not None:
    metadata["unit"] = imagej_info.get("unit", "micron")
    save_options["resolution"] = (1 / pixel_size, 1 / pixel_size)

tifffile.imwrite(OUTPUT_FILE, labels, imagej=True, metadata=metadata,
                 compression="zlib", **save_options)
print(f"Saved: {OUTPUT_FILE.relative_to(REPO)}")
