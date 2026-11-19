"""Track nuclei through a time-lapse movie with btrack, and show the tracks
in napari on top of the movie and the labels.

Input (in the Tracking folder):
  - P31-crop-crop.tif          the time-lapse movie (86 frames, 1 min apart)
  - P31-crop-crop_labels.tif   nuclei segmented and labeled in every frame

Output (in the Tracking folder):
  - P31-crop-crop_tracks.csv         one row per nucleus per frame
  - P31-crop-crop_track_scores.csv   the quality score of every setting tried

HOW BTRACK WORKS (in short)
  1. Frame to frame: btrack predicts where each nucleus will be in the next
     frame from how it moved so far (a "Kalman filter"), and links it to the
     nucleus it finds there.  This gives short pieces of track ("tracklets").
  2. tracker.optimize(): the tracklets are joined into full tracks.  A
     tracklet end can be linked to a tracklet start that is close by in
     space AND in time, and one track can split into two (cell division).
     Two settings decide what "close by" means:
       dist_thresh - maximum distance (pixels) between a track end and the
                     start it can be linked to
       time_thresh - maximum number of frames between them
     This matters here, because nuclei in mitosis are missing from the labels
     for a few frames: btrack has to bridge that gap to connect the mother
     to her daughters.

HOW THE SETTINGS ARE OPTIMIZED
  We have no "true" tracks to compare with, so we count errors that we can
  recognize without them.  In a perfect result, a track only starts or ends:
    - at the first or last frame of the movie,
    - at the image border (the nucleus moves in or out of view), or
    - at a division.
  Score = number of errors (lower is better):
    + tracks that start or end anywhere else ("unexplained" start/end)
    + links where a nucleus would move faster than MAX_SPEED
    + daughters that start further than MAX_DAUGHTER_DISTANCE from their
      mother (btrack "explaining" a track start with a division that cannot
      be real)
  All combinations of DIST_THRESHOLDS and TIME_THRESHOLDS are tried, and the
  one with the lowest score is used (if several tie: the smallest values,
  as they are the most cautious).

The first run downloads btrack's example cell_config file (needs internet).

Run it from the top folder of the workshop repository with:

    uv run src/btrack_nuclei.py
"""

import csv
import logging
from itertools import product
from pathlib import Path

import btrack
import napari
import numpy as np
import tifffile
from btrack import datasets

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent
MOVIE_FILE = REPO / "Tracking" / "P31-crop-crop.tif"
LABELS_FILE = REPO / "Tracking" / "P31-crop-crop_labels.tif"
TRACKS_CSV = REPO / "Tracking" / "P31-crop-crop_tracks.csv"
SCORES_CSV = REPO / "Tracking" / "P31-crop-crop_track_scores.csv"

# Values tried for the two thresholds (cell_config() uses 40 and 2)
DIST_THRESHOLDS = [10, 15, 20, 25, 30, 40, 60]   # pixels
TIME_THRESHOLDS = [2, 4, 6, 8, 10, 12, 15]       # frames

# Used by the quality score (nuclei here are ~24 pixels across)
BORDER_MARGIN = 15           # pixels; starts/ends this close to the edge are OK
MAX_SPEED = 8                # pixels per frame; faster links are errors
MAX_DAUGHTER_DISTANCE = 25   # pixels; ~1 nucleus diameter

logging.getLogger("btrack").setLevel(logging.WARNING)   # less text output

# ---------------------------------------------------------------------------
# 1. Read the data and turn the labels into objects btrack understands
# ---------------------------------------------------------------------------
with tifffile.TiffFile(MOVIE_FILE) as tif:
    movie = tif.asarray()
    resolution_tag = tif.pages[0].tags.get("XResolution")
    resolution = resolution_tag.value if resolution_tag else None
labels = tifffile.imread(LABELS_FILE)
n_frames, height, width = labels.shape

# Pixel size (if the movie has one), so napari shows all layers in µm
pixel_size = None
if resolution and resolution != (1, 1):
    pixel_size = resolution[1] / resolution[0]
scale = (1, pixel_size, pixel_size) if pixel_size else (1, 1, 1)

# One "object" per labeled nucleus per frame: its centre, frame and area
objects = btrack.utils.segmentation_to_objects(labels, properties=("area",))
print(f"{len(objects)} nuclei in {n_frames} frames")

config_file = datasets.cell_config()   # btrack's default settings for cells


def run_tracker(dist_thresh, time_thresh):
    """Track all objects with the given thresholds; return the tracker results."""
    with btrack.BayesianTracker(verbose=False) as tracker:
        tracker.configure(config_file)
        tracker.configuration.hypothesis_model.dist_thresh = dist_thresh
        tracker.configuration.hypothesis_model.time_thresh = time_thresh
        tracker.max_search_radius = 50
        tracker.append(objects)
        tracker.volume = ((0, width), (0, height))
        tracker.track(step_size=100)
        # join tracklets, find divisions (msg_lev: don't print solver details)
        tracker.optimize(tm_lim=60000, msg_lev="GLP_MSG_OFF")
        napari_data = tracker.to_napari()
        return tracker.tracks, napari_data


def near_border(x, y):
    return (x < BORDER_MARGIN or y < BORDER_MARGIN
            or x > width - BORDER_MARGIN or y > height - BORDER_MARGIN)


def score_tracks(tracks):
    """Count recognizable tracking errors (see explanation at the top)."""
    by_id = {track.ID: track for track in tracks}
    mothers = {track.parent for track in tracks if track.parent != track.ID}

    unexplained_starts = unexplained_ends = fast_links = far_daughters = 0
    for track in tracks:
        is_daughter = track.parent != track.ID
        if (not is_daughter and track.t[0] > 0
                and not near_border(track.x[0], track.y[0])):
            unexplained_starts += 1
        if (track.ID not in mothers and track.t[-1] < n_frames - 1
                and not near_border(track.x[-1], track.y[-1])):
            unexplained_ends += 1
        if is_daughter:
            mother = by_id[track.parent]
            if np.hypot(track.x[0] - mother.x[-1],
                        track.y[0] - mother.y[-1]) > MAX_DAUGHTER_DISTANCE:
                far_daughters += 1
        # speed between consecutive real (not filled-in) positions
        real = ~np.asarray(track.dummy, dtype=bool)
        t, x, y = (np.asarray(v)[real] for v in (track.t, track.x, track.y))
        if len(t) > 1:
            speed = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
            fast_links += int(np.sum(speed > MAX_SPEED))

    return {
        "tracks": len(tracks),
        "divisions": len(mothers),
        "unexplained_starts": unexplained_starts,
        "unexplained_ends": unexplained_ends,
        "fast_links": fast_links,
        "far_daughters": far_daughters,
        "score": unexplained_starts + unexplained_ends + fast_links + far_daughters,
    }


# ---------------------------------------------------------------------------
# 2. Try all combinations of dist_thresh and time_thresh
# ---------------------------------------------------------------------------
print("\ndist  time | tracks divisions  unexpl.start unexpl.end  fast far.daughters | score")
scores = []
for dist_thresh, time_thresh in product(DIST_THRESHOLDS, TIME_THRESHOLDS):
    tracks, _ = run_tracker(dist_thresh, time_thresh)
    s = {"dist_thresh": dist_thresh, "time_thresh": time_thresh, **score_tracks(tracks)}
    scores.append(s)
    print(f"{dist_thresh:4d}  {time_thresh:4d} | {s['tracks']:6d} {s['divisions']:9d} "
          f"{s['unexplained_starts']:12d} {s['unexplained_ends']:10d} {s['fast_links']:5d} "
          f"{s['far_daughters']:13d} | {s['score']:5d}")

with open(SCORES_CSV, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=scores[0].keys())
    writer.writeheader()
    writer.writerows(scores)

best = min(scores, key=lambda s: (s["score"], s["dist_thresh"], s["time_thresh"]))
default = next((s for s in scores if s["dist_thresh"] == 40 and s["time_thresh"] == 2), None)
if default:
    print(f"\ncell_config() default (dist 40, time 2): score {default['score']}")
print(f"Best: dist_thresh = {best['dist_thresh']}, time_thresh = {best['time_thresh']} "
      f"-> score {best['score']}, {best['tracks']} tracks, {best['divisions']} divisions")

# ---------------------------------------------------------------------------
# 3. Final tracking with the best settings, save the tracks as csv
# ---------------------------------------------------------------------------
tracks, (track_data, track_properties, track_graph) = run_tracker(
    best["dist_thresh"], best["time_thresh"])

with open(TRACKS_CSV, "w", newline="") as fh:
    writer = csv.writer(fh)
    header = ["track_id", "frame", "x_pixels", "y_pixels", "area_pixels",
              "parent_id", "root_id", "generation", "filled_in"]
    if pixel_size:
        header += ["x_um", "y_um"]
    writer.writerow(header)
    for track in tracks:
        areas = track.properties.get("area", [np.nan] * len(track))
        for i in range(len(track)):
            row = [track.ID, track.t[i], round(track.x[i], 2), round(track.y[i], 2),
                   areas[i], track.parent, track.root, track.generation,
                   bool(track.dummy[i])]
            if pixel_size:
                row += [round(track.x[i] * pixel_size, 3), round(track.y[i] * pixel_size, 3)]
            writer.writerow(row)
print(f"Saved: {TRACKS_CSV.relative_to(REPO)} and {SCORES_CSV.relative_to(REPO)}")

# ---------------------------------------------------------------------------
# 4. Show movie, labels and tracks in napari
# ---------------------------------------------------------------------------
viewer = napari.Viewer()
viewer.add_image(movie, name="movie", scale=scale)
viewer.add_labels(labels, name="labels", scale=scale, opacity=0.4)
viewer.add_tracks(track_data, properties=track_properties, graph=track_graph,
                  name=f"tracks (dist {best['dist_thresh']}, time {best['time_thresh']})",
                  scale=scale, tail_length=20, color_by="track_id", blending="translucent")
napari.run()
