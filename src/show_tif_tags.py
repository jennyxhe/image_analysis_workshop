"""Print all TIFF tags of an image in a human-readable way.

Run it from the top folder of the workshop repository with:

    uv run src/show_tif_tags.py

or point it at another file:

    uv run src/show_tif_tags.py testImages/some_other_image.tif

Background: a TIFF file is a list of "pages" (one 2D image plane each).
Every page carries a set of "tags": numbered labels (e.g. 256 = ImageWidth)
that describe the image - its size, bit depth, resolution, and any extra
metadata the acquisition software (Micro-Manager, ImageJ, ...) wrote in.
"""

import sys
from enum import Enum
from fractions import Fraction
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).parent.parent
DEFAULT_FILE = REPO / "testImages/Cell_cropped.tif"

MAX_ITEMS = 8  # long lists / arrays are shortened to this many items


def format_value(name, value):
    """Turn a raw tag value into a short, readable string."""
    # Named constants, e.g. PHOTOMETRIC.MINISBLACK -> "MINISBLACK (1)"
    if isinstance(value, Enum):
        return f"{value.name} ({value.value})"

    # Resolution is stored as a fraction (numerator, denominator)
    if name in ("XResolution", "YResolution") and isinstance(value, tuple):
        return f"{value[0] / value[1]:.6g} pixels per unit  (stored as {value[0]}/{value[1]})"

    # Multi-line text (e.g. the ImageJ description) -> one indented line each
    if isinstance(value, str):
        lines = value.strip().splitlines()
        if len(lines) > 1:
            return "\n" + "\n".join(f"        {line}" for line in lines)
        return value

    # Dictionaries (e.g. ImageJ's own metadata) -> one key per line
    if isinstance(value, dict):
        return "\n" + "\n".join(
            f"        {k}: {format_value(k, v)}" for k, v in value.items()
        )

    # Big blocks of numbers (e.g. lookup tables): describe instead of dumping
    if isinstance(value, np.ndarray):
        return f"array of shape {value.shape}, type {value.dtype}"

    if isinstance(value, (list, tuple)):
        if value and all(isinstance(v, np.ndarray) for v in value):
            return f"{len(value)} arrays of shape {value[0].shape}"
        shown = ", ".join(str(v) for v in value[:MAX_ITEMS])
        more = f", ... ({len(value)} values in total)" if len(value) > MAX_ITEMS else ""
        return f"({shown}{more})"

    if isinstance(value, bytes):
        return f"{len(value)} bytes of binary data"

    return str(value)


def print_tags(tags, only=None):
    for tag in tags:
        if only is not None and tag.name not in only:
            continue
        print(f"  [{tag.code:>5}] {tag.name:<28} {format_value(tag.name, tag.value)}")


path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FILE

with tifffile.TiffFile(path) as tif:
    print(f"File: {path}")
    print(f"Number of pages (image planes): {len(tif.pages)}")
    series = tif.series[0]
    print(f"Image shape: {series.shape}  axes: {series.axes}  data type: {series.dtype}")
    print(f"Written by ImageJ: {tif.is_imagej}   OME-TIFF: {tif.is_ome}")

    # --- All tags of the first page --------------------------------------
    first = tif.pages[0]
    print("\n=== Tags of page 0 ===")
    print_tags(first.tags)

    # --- Pixel size, worked out from the resolution tags -----------------
    # ImageJ stores "pixels per unit" in XResolution and the unit in the
    # description (e.g. unit=micron).  Pixel size = 1 / resolution.
    if "XResolution" in first.tags:
        num, den = first.tags["XResolution"].value
        unit = (tif.imagej_metadata or {}).get("unit", "unit")
        if num:
            print(f"\n=> Pixel size: {float(Fraction(den, num)):.4f} {unit} per pixel")

    # --- Other pages: only show tags whose value differs from page 0 -----
    for i, page in enumerate(tif.pages[1:], start=1):
        different = [
            t.name
            for t in page.tags
            if t.name not in first.tags
            or not np.array_equal(np.asarray(t.value, dtype=object),
                                  np.asarray(first.tags[t.name].value, dtype=object))
        ]
        print(f"\n=== Page {i}: tags that differ from page 0 ===")
        if different:
            print_tags(page.tags, only=different)
        else:
            print("  (none - same as page 0)")
