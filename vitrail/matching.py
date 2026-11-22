"""Replacement-glass sample matching under two lighting conditions.

The workbench receives two photographs of the same laid-out pieces: one under
the daytime white lamp and one under the evening warm lamp. Every photograph
shows the surviving original fragment, three replacement samples and one clear
blank piece. The blank is measured on each photograph and used as an in-frame
illumination reference, so the lighting of the two captures can be balanced.

Hue agreement and lightness (brightness) agreement are reported separately.
The ranking belongs to the lighting condition: when the order changes between
the white-lamp and warm-lamp photographs, no single first choice is invented.
"""
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

from .bench import Result, ratio
from .camera import brightness, linear_to_srgb
from .contracts import Box, Problem, integer, keys, label, rounded, scalar
from .photometry import rgb_to_lab
from .picture import csv_table, mark_image, mask_overlay, png


ROLES = ("original", "sample_1", "sample_2", "sample_3", "blank")
ROLE_TITLES = {
    "original": "Original fragment",
    "sample_1": "Sample 1",
    "sample_2": "Sample 2",
    "sample_3": "Sample 3",
    "blank": "Clear blank reference",
}
SAMPLE_ROLES = ("sample_1", "sample_2", "sample_3")
# Slightly inset, non-overlapping defaults for a five-piece horizontal layout.
DEFAULT_BOXES = {
    "original": [0.06, 0.32, 0.20, 0.68],
    "sample_1": [0.24, 0.32, 0.38, 0.68],
    "sample_2": [0.42, 0.32, 0.56, 0.68],
    "sample_3": [0.60, 0.32, 0.74, 0.68],
    "blank": [0.78, 0.32, 0.92, 0.68],
}
OPTION_KEYS = set(ROLES) | {"brightness_limit_dl", "hue_tolerance_deg", "minimum_pixels", "sample_labels"}
MIN_SIGNAL = 0.002


@dataclass(frozen=True)
class PatchReading:
    role: str
    box: Box
    observed: int
    coverage: float
    clipped: int
    rgb_linear: np.ndarray
    spread: float

    def chip(self):
        return tuple(int(v) for v in linear_to_srgb(np.clip(self.rgb_linear, 0, 1)))


def parse_layout(opts):
    keys(opts, *sorted(OPTION_KEYS))
    boxes = {}
    for role in ROLES:
        boxes[role] = Box.parse(opts.get(role, DEFAULT_BOXES[role]), ROLE_TITLES[role])
    brightness_limit = scalar(opts.get("brightness_limit_dl", 6.0), "Visible brightness gap (delta L)", 0.5, 40)
    hue_tolerance = scalar(opts.get("hue_tolerance_deg", 4.0), "Hue tie tolerance (degrees)", 0.1, 45)
    minimum = integer(opts.get("minimum_pixels", 64), "Minimum patch pixels", 12, 200000)
    labels = {}
    raw_labels = opts.get("sample_labels", {})
    if raw_labels is not None:
        if not isinstance(raw_labels, dict):
            raise Problem("Sample labels must be an object keyed by sample_1, sample_2 and sample_3.")
        for key, value in raw_labels.items():
            if key not in SAMPLE_ROLES:
                raise Problem("Sample labels may only use sample_1, sample_2 and sample_3 keys.")
            labels[key] = label(value, f"Label for {key}", 40)
    for role in SAMPLE_ROLES:
        labels.setdefault(role, ROLE_TITLES[role])
    return boxes, brightness_limit, hue_tolerance, minimum, labels


def _read_patch(frame, role, box, minimum):
    width, height = frame.rgb.shape[1], frame.rgb.shape[0]
    mask = box.mask(width, height)
    usable = mask & frame.measurable
    observed = int(usable.sum())
    if observed < minimum:
        raise Problem(
            f"{ROLE_TITLES[role]} only has {observed} clear pixels; mark a larger interior region (need {minimum}).",
            422,
        )
    data = frame.rgb[usable]
    median = np.median(data, axis=0).astype(np.float64)
    if role != "blank" and np.any(median <= 0):
        raise Problem(f"{ROLE_TITLES[role]} is too dark to read; choose a better-lit interior.")
    luma = brightness(data)
    spread = float(np.percentile(luma, 90) - np.percentile(luma, 10))
    clipped = int((mask & frame.clipped).sum())
    coverage = ratio(usable, mask)
    return PatchReading(role, box, observed, coverage, clipped, median, spread)


def _hue_angle(lab):
    if _chroma(lab) < 0.05:
        return 0.0
    return float(np.degrees(np.arctan2(lab[2], lab[1])) % 360.0)


def _hue_gap(angle_a, angle_b):
    return float(abs((angle_a - angle_b + 180.0) % 360.0 - 180.0))


def _chroma(lab):
    return float(np.hypot(lab[1], lab[2]))


def _normalize(reading, blank_rgb):
    """Blank-balanced glass transmittance: glass signal divided by blank signal."""
    factor = reading.rgb_linear / np.maximum(blank_rgb, MIN_SIGNAL)
    return np.clip(factor, 0.0, 1.0)


def _measure_frame(frame, boxes, minimum, labels):
    readings = {role: _read_patch(frame, role, box, minimum) for role, box in boxes.items()}
    blank = readings["blank"]
    if np.any(blank.rgb_linear <= MIN_SIGNAL):
        raise Problem("The clear blank reference is too dark to balance the photograph.")
    if blank.clipped:
        raise Problem("The clear blank reference contains saturated pixels; re-shoot or mark a smaller blank interior.")
    blank_lab = rgb_to_lab(np.clip(blank.rgb_linear, 0, 1))
    rows = {}
    for role in ROLES:
        reading = readings[role]
        normalized = _normalize(reading, blank.rgb_linear)
        # Lab of the blank-balanced glass color is computed on the normalized
        # transmittance, so the warm/white lamp color is divided out per channel.
        lab = rgb_to_lab(normalized)
        rows[role] = {
            "role": role,
            "name": labels.get(role, ROLE_TITLES[role]),
            "region": reading.box.public(),
            "observed_pixels": reading.observed,
            "coverage": reading.coverage,
            "clipped_pixels": reading.clipped,
            "median_linear_rgb": [rounded(v) for v in reading.rgb_linear],
            "preview_srgb": list(reading.chip()),
            "blank_balanced_rgb": [rounded(v) for v in normalized],
            "luminance_spread": rounded(reading.spread),
            "lab": [rounded(float(v)) for v in lab],
            "hue_deg": rounded(_hue_angle(lab)),
            "chroma": rounded(_chroma(lab)),
            "blank_lab": [rounded(float(v)) for v in blank_lab],
        }
    return rows, readings


def _compare_sample(original, sample, brightness_limit, hue_tolerance):
    o_lab = np.asarray(original["lab"], dtype=float)
    s_lab = np.asarray(sample["lab"], dtype=float)
    hue_distance = _hue_gap(original["hue_deg"], sample["hue_deg"])
    lightness_gap = float(s_lab[0] - o_lab[0])
    chroma_gap = float(_chroma(s_lab) - _chroma(o_lab))
    total_delta_e = float(np.linalg.norm(s_lab - o_lab))
    flags = []
    if abs(lightness_gap) >= brightness_limit:
        direction = "brighter" if lightness_gap > 0 else "darker"
        flags.append(f"brightness gap {abs(lightness_gap):.1f} delta L ({direction})")
    if hue_distance >= max(2.0 * hue_tolerance, 8.0):
        flags.append(f"hue gap {hue_distance:.1f} degrees")
    return {
        "sample": sample["role"],
        "name": sample["name"],
        "hue_distance_deg": rounded(hue_distance),
        "lightness_delta_l": rounded(lightness_gap),
        "chroma_delta": rounded(chroma_gap),
        "total_delta_e": rounded(total_delta_e),
        "brightness_flag": abs(lightness_gap) >= brightness_limit,
        "hue_flag": hue_distance >= max(2.0 * hue_tolerance, 8.0),
        "flags": flags,
        "sample_srgb": sample["preview_srgb"],
        "sample_lab": sample["lab"],
    }


def _rank(comparisons, hue_tolerance):
    ordered = sorted(comparisons, key=lambda item: (item["hue_distance_deg"], item["total_delta_e"]))
    best = ordered[0]["hue_distance_deg"]
    tied = [item["sample"] for item in ordered if abs(item["hue_distance_deg"] - best) <= hue_tolerance]
    for rank, item in enumerate(ordered, 1):
        item["hue_rank"] = rank
    return ordered, tied


def _frame_markup(rows):
    shapes = []
    palette = {
        "original": (247, 218, 132),
        "sample_1": (159, 224, 202),
        "sample_2": (150, 196, 235),
        "sample_3": (216, 167, 224),
        "blank": (238, 236, 224),
    }
    for role in ROLES:
        x0, y0, x1, y1 = rows[role]["region"]
        shapes.append(([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], palette[role], rows[role]["name"]))
    return shapes


def _chip_strip(condition_rows):
    width, row_h = 760, 96
    picture = Image.new("RGB", (width, 34 + row_h * len(condition_rows)), (246, 241, 228))
    draw = ImageDraw.Draw(picture)
    for index, row in enumerate(condition_rows):
        top = 12 + index * row_h
        draw.text((14, top + 6), row["condition"], fill=(41, 53, 54))
        chips = [("Original", row["patches"]["original"]["preview_srgb"])]
        for item in row["ranking"]:
            chips.append((f"#{item['hue_rank']} {item['name']}", item["sample_srgb"]))
        x = 16
        for name, color in chips:
            draw.rectangle((x, top + 28, x + 96, top + 76), fill=tuple(color), outline=(72, 72, 72))
            draw.text((x, top + 80), name[:22], fill=(58, 58, 73))
            x += 118
        draw.line((14, top + row_h - 4, width - 14, top + row_h - 4), fill=(214, 207, 190))
    output = BytesIO()
    picture.save(output, format="PNG", compress_level=5)
    return png("sample-chip-strip", picture)


def matching_report(ctx):
    if len(ctx.frames) != 2:
        raise Problem("Replacement matching needs exactly two captures: white lamp and warm lamp.")
    opts = ctx.order.toolbox
    boxes, brightness_limit, hue_tolerance, minimum, labels = parse_layout(opts)
    conditions = []
    csv_rows = []
    for frame in ctx.frames:
        patches, readings = _measure_frame(frame, boxes, minimum, labels)
        comparisons = [_compare_sample(patches["original"], patches[role], brightness_limit, hue_tolerance)
                       for role in SAMPLE_ROLES]
        ranking, tied = _rank(comparisons, hue_tolerance)
        condition = {
            "capture": frame.record["id"],
            "condition": frame.record["data"]["title"],
            "illuminant": frame.record["data"]["illuminant"],
            "patches": patches,
            "ranking": ranking,
            "hue_tied_samples": tied,
            "brightness_limit_dl": brightness_limit,
            "hue_tolerance_deg": hue_tolerance,
        }
        conditions.append(condition)
        for role in ROLES:
            patch = patches[role]
            csv_rows.append([frame.record["data"]["title"], patch["name"], role, *patch["region"],
                             patch["observed_pixels"], patch["coverage"], patch["clipped_pixels"],
                             *patch["median_linear_rgb"], *patch["lab"], patch["hue_deg"], patch["chroma"]])
        marked = mark_image(mask_overlay(frame.rgb, [], frame.present), _frame_markup(patches))
        condition["map_asset_key"] = f"patch-map-{frame.record['id'][:8]}"
        condition["_map"] = png(condition["map_asset_key"], marked, ctx, crop=False)

    first, second = conditions
    first_order = [item["sample"] for item in first["ranking"]]
    second_order = [item["sample"] for item in second["ranking"]]
    order_swapped = first_order != second_order
    first_leader = first["ranking"][0]
    second_leader = second["ranking"][0]
    if order_swapped:
        verdict = "ranking_changes"
        recommendation = (
            f"No single first choice: {first_leader['name']} leads in '{first['condition']}' while "
            f"{second_leader['name']} leads in '{second['condition']}'. Keep both condition rankings and decide on site."
        )
    elif len(first["hue_tied_samples"]) > 1 or len(second["hue_tied_samples"]) > 1:
        verdict = "hue_tie"
        recommendation = (
            f"{first_leader['name']} is the hue leader in both lights, but the lead is within the "
            f"{hue_tolerance:g} degree tie tolerance; treat the top samples as a hue match rather than a winner."
        )
    else:
        verdict = "consistent_leader"
        recommendation = f"{first_leader['name']} ranks first for hue under both lamps."
    brightness_warnings = sorted({
        flag
        for condition in conditions
        for item in condition["ranking"]
        for flag in item["flags"]
    })

    summary = {
        "conditions": len(conditions),
        "first_capture_leader": first_leader["name"],
        "second_capture_leader": second_leader["name"],
        "ranking_consistent": not order_swapped,
        "verdict": verdict,
        "brightness_warnings": len(brightness_warnings),
    }
    details = {
        **summary,
        "recommendation": recommendation,
        "brightness_limit_dl": brightness_limit,
        "hue_tolerance_deg": hue_tolerance,
        "conditions": [{k: v for k, v in condition.items() if not k.startswith("_")} for condition in conditions],
        "first_capture_order": [labels[key] for key in first_order],
        "second_capture_order": [labels[key] for key in second_order],
        "brightness_warnings": brightness_warnings,
        "note": (
            "Hue is ranked on blank-balanced CIELAB hue angle per lighting condition. Lightness delta L is "
            "reported separately; absolute delta L at or above the visibility limit is flagged. Rankings are "
            "photograph-specific and are not merged when the two lamps disagree."
        ),
    }
    maps = tuple(condition["_map"] for condition in conditions)
    strip = _chip_strip(conditions)
    table = csv_table(
        "sample-readings",
        ("photo", "patch", "role", "box_left", "box_top", "box_right", "box_bottom", "observed_pixels",
         "coverage", "clipped_pixels", "linear_r", "linear_g", "linear_b", "L", "a", "b", "hue_deg", "chroma"),
        csv_rows,
    )
    ranking_table = csv_table(
        "sample-ranking",
        ("photo", "hue_rank", "sample", "hue_distance_deg", "lightness_delta_l", "chroma_delta",
         "total_delta_e", "brightness_flag", "hue_flag"),
        [
            [condition["condition"], item["hue_rank"], item["name"], item["hue_distance_deg"],
             item["lightness_delta_l"], item["chroma_delta"], item["total_delta_e"],
             item["brightness_flag"], item["hue_flag"]]
            for condition in conditions for item in condition["ranking"]
        ],
    )
    return Result(
        "replacement-match",
        "Replacement glass sample matching",
        summary,
        details,
        (*maps, strip, table, ranking_table),
    )
