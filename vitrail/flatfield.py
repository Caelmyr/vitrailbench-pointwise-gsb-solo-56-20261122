from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, find_objects, label as connected

from .bench import Bench, Result, masked_blur, percentiles, ratio
from .camera import brightness
from .contracts import Problem, WorkOrder, keys, rounded, scalar
from .leads import CONNECTION
from .picture import colorize, coded, csv_table, mark_image, png


PHOTOMETRIC_FLOOR = 0.001
FIELD_FLOOR = 0.0005
FIELD_WEIGHT_FLOOR = 0.03
REFERENCE_FLOOR = 36
LOW_CONFIDENCE = 0.6
REPORTED_COMPONENTS = 160

# Each fill class carries its own confidence; a borrowed reading is never
# presented with the same standing as a measured flat-field pixel.
CATEGORIES = {
    "direct": dict(code=1, color=(86, 132, 116), base=1.0, label="Direct flat reading",
                   description="Measured flat-field pixel."),
    "isolated-hole": dict(code=2, color=(238, 207, 120), base=0.7, label="Isolated hole",
                          description="Small compact gap filled from surrounding flat readings."),
    "extended-gap": dict(code=3, color=(206, 118, 84), base=0.5, label="Extended gap",
                         description="Wide gap filled across a longer span."),
    "long-strip": dict(code=4, color=(226, 146, 79), base=0.35, label="Long strip",
                       description="Elongated unphotographed strip filled along its length; the along-strip gradient stays unobserved."),
    "unfilled": dict(code=5, color=(150, 58, 60), base=0.0, label="Not corrected",
                     description="No flat reading within the borrow limit."),
}
CODES = {name: item["code"] for name, item in CATEGORIES.items()}


@dataclass(frozen=True)
class GapComponent:
    index: int
    label_id: int
    category: str
    pixels: int
    area_mm2: float
    width_mm: float
    height_mm: float
    elongation: float
    nearest_mm: float
    farthest_mm: float
    confidence: float
    bounds: tuple

    def public(self):
        return dict(component=self.index, category=self.category, category_label=CATEGORIES[self.category]["label"],
                    pixels=self.pixels, area_mm2=self.area_mm2, width_mm=self.width_mm, height_mm=self.height_mm,
                    elongation=self.elongation, nearest_reading_mm=self.nearest_mm,
                    farthest_reading_mm=self.farthest_mm, confidence=self.confidence)


@dataclass(frozen=True)
class Correction:
    target: object
    flat: object
    flat_source: str
    reference: float
    field: np.ndarray
    valid: np.ndarray
    distance: np.ndarray
    direct: np.ndarray
    categories: np.ndarray
    confidence: np.ndarray
    gain: np.ndarray
    corrected_valid: np.ndarray
    corrected_rgb: np.ndarray
    components: tuple
    counts: dict
    parameters: dict

    def summary(self, ctx):
        observed = self.target.present & ctx.selection
        selection = ctx.selection
        field_ev = np.log2(np.maximum(self.field, 1e-6) / max(self.reference, 1e-6))
        range_ev = percentiles(field_ev, self.valid & selection, (5, 50, 95))
        span = None if range_ev[0] is None else rounded(range_ev[2] - range_ev[0])
        borrowed = self.corrected_valid & ~self.direct
        return dict(
            flat_usable_fraction=ratio(self.direct & selection, selection),
            observed_glass_fraction=ratio(observed, selection),
            corrected_share_of_observed=ratio(self.corrected_valid, observed),
            uncorrected_share_of_observed=ratio(observed & ~self.corrected_valid, observed),
            direct_share_of_corrected=ratio(self.corrected_valid & self.direct, self.corrected_valid),
            isolated_hole_share=ratio(borrowed & (self.categories == CODES["isolated-hole"]), self.corrected_valid),
            extended_gap_share=ratio(borrowed & (self.categories == CODES["extended-gap"]), self.corrected_valid),
            long_strip_share=ratio(borrowed & (self.categories == CODES["long-strip"]), self.corrected_valid),
            low_confidence_share=ratio(self.corrected_valid & (self.confidence < LOW_CONFIDENCE), self.corrected_valid),
            flat_range_ev=span, flat_range_quantiles_ev=range_ev, reference_linear=rounded(self.reference),
            gap_components=sum(self.counts.values()), component_counts=self.counts,
            target_capture=self.target.record["id"], flat_capture=self.flat.record["id"],
            flat_source=self.flat_source)


def _pair(ctx):
    frames = ctx.exactly(2)
    roles = [frame.record["data"]["role"] for frame in frames]
    if roles.count("flat") > 1:
        raise Problem("Select one open-backlight capture and one glass capture, not two flat captures.")
    if roles.count("flat") == 1:
        flat_index = roles.index("flat")
        source = "capture tagged as a flat backlight"
    else:
        flat_index = 1
        source = "second selected capture (no capture is tagged flat)"
    return frames[1 - flat_index], frames[flat_index], source


def _direct_readings(flat, obscured):
    measured = flat.present & ~flat.clipped & (flat.signal > PHOTOMETRIC_FLOOR)
    values = flat.signal[measured]
    if values.size < REFERENCE_FLOOR:
        raise Problem("The open-backlight capture has too few usable pixels in this panel plane.", 422)
    middle = float(np.median(values))
    if middle <= PHOTOMETRIC_FLOOR:
        raise Problem("The open-backlight capture is too dark to define a correction field.", 422)
    # A clamp shadow is still "present" but reads far below the flat level; treating
    # it as a direct reading would divide by nearly zero.
    direct = measured & (flat.signal >= middle * obscured)
    if direct.sum() < REFERENCE_FLOOR:
        raise Problem("The open-backlight capture has too few unblocked pixels for a stable field.", 422)
    return direct, middle


def _field(signal, direct, scale, smooth_mm, borrow_mm):
    sigma = (max(1e-3, smooth_mm / scale.step_y), max(1e-3, smooth_mm / scale.step_x))
    smooth, enough = masked_blur(signal, direct, sigma, floor=FIELD_WEIGHT_FLOOR)
    distance, nearest = distance_transform_edt(~direct, sampling=(scale.step_y, scale.step_x),
                                               return_indices=True)
    nearest_value = signal[nearest[0], nearest[1]]
    field = np.where(direct, smooth, np.where(enough, smooth, nearest_value))
    valid = direct | ((~direct) & (distance <= borrow_mm))
    valid &= field > FIELD_FLOOR
    return field.astype(np.float32), valid, distance.astype(np.float32)


def _classify(distance, direct, scale, borrow_mm, isolated_max, strip_min):
    gap = ~direct
    ids, total = connected(gap, structure=CONNECTION)
    if total == 0:
        empty = np.zeros(distance.shape, dtype=np.uint8)
        empty[direct] = CODES["direct"]
        return empty, np.where(direct, 1.0, 0.0).astype(np.float32), tuple(), dict.fromkeys(
            (name for name in CATEGORIES if name != "unfilled"), 0)
    counts = np.bincount(ids.ravel(), minlength=total + 1)
    farthest = np.zeros(total + 1)
    nearest = np.full(total + 1, np.inf)
    np.maximum.at(farthest, ids.ravel(), distance.ravel())
    np.minimum.at(nearest, ids.ravel(), distance.ravel())
    nearest[0] = 0
    farthest[0] = 0
    slices = find_objects(ids)
    rows = []
    for label_id in range(1, total + 1):
        area = float(counts[label_id]) * scale.pixel_area
        region = slices[label_id - 1]
        if region is None or counts[label_id] == 0:
            continue
        height_mm = (region[0].stop - region[0].start) * scale.step_y
        width_mm = (region[1].stop - region[1].start) * scale.step_x
        long_side = max(width_mm, height_mm)
        short_side = max(1e-6, min(width_mm, height_mm))
        elongation = long_side / short_side
        if long_side >= strip_min and elongation >= 3:
            category = "long-strip"
        elif area <= isolated_max:
            category = "isolated-hole"
        else:
            category = "extended-gap"
        falloff = float(np.exp(-nearest[label_id] / max(borrow_mm, 1e-6)))
        rows.append((label_id, category, counts[label_id], area, width_mm, height_mm, elongation,
                     nearest[label_id], farthest[label_id], falloff,
                     ((region[1].start / scale.width_px, region[0].start / scale.height_px),
                      (region[1].stop / scale.width_px, region[0].stop / scale.height_px))))
    rows.sort(key=lambda row: (-row[3], row[0]))
    code_lut = np.zeros(total + 1, np.uint8)
    conf_lut = np.zeros(total + 1, np.float32)
    components = []
    category_counts = {name: 0 for name in CATEGORIES if name != "unfilled"}
    for index, (label_id, category, pixels, area, width_mm, height_mm, elongation,
                near, far, falloff, bounds) in enumerate(rows, 1):
        reachable = near <= borrow_mm
        confidence = CATEGORIES[category]["base"] * falloff if reachable else 0.0
        code_lut[label_id] = CODES[category]
        # Per-pixel confidence decays with the distance to the borrowed reading; the
        # component figure above reports the reading at its nearest edge.
        conf_lut[label_id] = CATEGORIES[category]["base"] if reachable else 0.0
        if reachable:
            category_counts[category] += 1
        components.append(GapComponent(index, label_id, category, int(pixels), rounded(area),
                                       rounded(width_mm), rounded(height_mm), rounded(elongation),
                                       rounded(near), rounded(far), rounded(confidence), bounds))
    categories = np.zeros(ids.shape, dtype=np.uint8)
    categories[direct] = CODES["direct"]
    categories[gap] = code_lut[ids[gap]]
    confidence = np.zeros(ids.shape, dtype=np.float32)
    confidence[direct] = 1.0
    confidence[gap] = conf_lut[ids[gap]] * np.exp(-distance[gap] / max(borrow_mm, 1e-6))
    beyond = gap & (distance > borrow_mm)
    categories[beyond] = CODES["unfilled"]
    confidence[beyond] = 0.0
    return categories, confidence, tuple(components), category_counts


def solve(ctx, target, flat, source, options):
    smooth_mm = scalar(options.get("smooth_mm", 2.5), "Flat-field smoothing (mm)", 0.2, 20)
    borrow_mm = scalar(options.get("borrow_mm", 12), "Nearest-reading borrow distance (mm)", 0, 60)
    obscured = scalar(options.get("obscured_flat", 0.06), "Blocked flat fraction", 0.005, 0.5)
    isolated_max = scalar(options.get("isolated_max_mm2", 60), "Largest isolated hole (mm²)", 1, 5000)
    strip_min = scalar(options.get("strip_min_mm", 10), "Longest strip side for a strip (mm)", 2, 200)
    direct, middle = _direct_readings(flat, obscured)
    field, valid, distance = _field(flat.signal, direct, ctx.scale, smooth_mm, borrow_mm)
    categories, confidence, components, counts = _classify(distance, direct, ctx.scale, borrow_mm,
                                                           isolated_max, strip_min)
    basis = valid & ctx.selection
    if not basis.any():
        basis = valid
    if not basis.any():
        raise Problem("No flat-field support overlaps the selected panel window.", 422)
    reference = float(np.median(field[basis]))
    if reference <= FIELD_FLOOR:
        raise Problem("The flat field is too dark to normalize this correction.", 422)
    gain = np.where(valid, reference / np.maximum(field, 1e-6), 0).astype(np.float32)
    corrected_valid = target.measurable & valid
    corrected_rgb = (target.rgb * gain[..., None]).astype(np.float32)
    corrected_rgb[~corrected_valid] = 0
    if corrected_valid.sum() < REFERENCE_FLOOR:
        raise Problem("The selected glass capture has too few pixels that can be corrected.", 422)
    parameters = dict(smooth_mm=smooth_mm, borrow_mm=borrow_mm, obscured_flat=obscured,
                      isolated_max_mm2=isolated_max, strip_min_mm=strip_min)
    return Correction(target, flat, source, reference, field, valid, distance, direct, categories,
                      confidence, gain, corrected_valid, corrected_rgb, components, counts, parameters)


def _display_scale(correction, ctx):
    sample = correction.corrected_valid & ctx.selection
    if not sample.any():
        sample = correction.target.measurable & correction.valid
    if not sample.any():
        return 1.0
    value = float(np.percentile(brightness(correction.corrected_rgb)[sample], 98))
    return value if value > 1e-4 else 1.0


def _side_by_side(before, after, left_caption, right_caption):
    height, width = before.shape[:2]
    gap = 8
    header = 24
    canvas = Image.new("RGB", (2 * width + gap, height + header), (28, 32, 39))
    canvas.paste(Image.fromarray(before), (0, header))
    canvas.paste(Image.fromarray(after), (width + gap, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), left_caption[:64], fill=(241, 234, 220))
    draw.text((width + gap + 6, 6), right_caption[:64], fill=(241, 234, 220))
    draw.line((width + gap // 2, header, width + gap // 2, header + height), fill=(124, 130, 138))
    return np.asarray(canvas)


def _provenance(correction):
    flat = correction.flat
    basis = flat.measurable
    ceiling = float(np.percentile(flat.signal[basis], 99)) if basis.any() else 1.0
    # A dimmed base keeps the borrowed classes legible against a bright open backlight.
    scene = np.clip(flat.rgb / max(ceiling, 1e-4), 0, 1) * 0.5
    picture = coded(scene, flat.present).astype(np.float32)
    for name, item in CATEGORIES.items():
        if name == "direct":
            continue
        mask = correction.categories == item["code"]
        picture[mask] = picture[mask] * 0.42 + np.asarray(item["color"]) * 0.58
    picture = np.rint(picture).astype(np.uint8)
    marks = []
    for component in correction.components[:36]:
        left, right = component.bounds
        x0, y0 = left
        x1, y1 = right
        marks.append(([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)],
                      CATEGORIES[component.category]["color"], str(component.index)))
    return mark_image(picture, marks)


def flatfield(ctx):
    target, flat, source = _pair(ctx)
    opts = ctx.order.toolbox
    keys(opts, "smooth_mm", "borrow_mm", "obscured_flat", "isolated_max_mm2", "strip_min_mm")
    correction = solve(ctx, target, flat, source, opts)
    ctx.demand(correction.corrected_valid & ctx.selection)
    summary = correction.summary(ctx)
    display = _display_scale(correction, ctx)
    before = coded(target.rgb / display, target.present)
    after = coded(correction.corrected_rgb / display, correction.corrected_valid)
    side = _side_by_side(before, after, f"Original transmission · {target.record['data']['title']}",
                         f"Corrected by flat field · {flat.record['data']['title']}")
    corrected = after
    field_ev = np.log2(np.maximum(correction.field, 1e-6) / max(correction.reference, 1e-6))
    spread = max(0.6, abs(float(np.percentile(field_ev[correction.valid], 5))) if correction.valid.any() else 0.6,
                 abs(float(np.percentile(field_ev[correction.valid], 95))) if correction.valid.any() else 0.6)
    provenance = _provenance(correction)
    reported = [component for component in correction.components
                if component.area_mm2 >= 0.25][:REPORTED_COMPONENTS]
    table = csv_table("flat-components",
                      ("component", "category", "pixels", "area_mm2", "width_mm", "height_mm", "elongation",
                       "nearest_reading_mm", "farthest_reading_mm", "confidence"),
                      [(c.index, c.category, c.pixels, c.area_mm2, c.width_mm, c.height_mm, c.elongation,
                        c.nearest_mm, c.farthest_mm, c.confidence) for c in reported])
    details = {**summary, "parameters": correction.parameters,
               "reference_linear": rounded(correction.reference),
               "legend": [dict(name=name, code=item["code"], label=item["label"], base_confidence=item["base"],
                               description=item["description"]) for name, item in CATEGORIES.items()],
               "gaps": [component.public() for component in reported],
               "gaps_below_report_threshold": max(0, len(correction.components) - len(reported)),
               "note": ("The open backlight photograph is divided into the glass capture as a smooth illumination "
                        "field. Pixels without a direct flat reading borrow the nearest reading within the borrow "
                        "limit and keep a lower confidence; elongated strips and unfilled areas are listed "
                        "separately. Source photographs and their bytes stay unchanged; the corrected image is a "
                        "derived view.")}
    assets = (png("before-after", side),
              png("corrected", corrected, ctx, crop=False),
              png("borrow-provenance", provenance),
              png("flat-field", colorize(field_ev, -spread, spread, "diverging",
                                         correction.valid & ctx.selection), ctx),
              table)
    return Result("flatfield", "Flat-field transmission correction", summary, details, assets)


def _source_changes(library, facts):
    rows = []
    for source in facts["sources"]:
        try:
            current = library.photo(source["id"])
        except Problem as error:
            if error.status != 404:
                raise
            rows.append(dict(capture=source["id"], title=source["title"], missing=True))
            continue
        rows.append(dict(capture=source["id"], title=source["title"], missing=False,
                         revised=current["revision"] != source["revision"],
                         original_matches=current["original"] == source["original"],
                         alignment_matches=current["active_alignment"] == source["alignment_id"]))
    return rows


def reopen_correction(library, record):
    """Rebuild the retained correction as a current preview without touching any source."""
    if record["kind"] != "flatfield":
        raise Problem("Only a retained flat-field correction can reopen a corrected preview.", 400)
    facts = record["facts"]
    panel_id = facts["panel"]["id"]
    panel = library.panel_summary(panel_id)["panel"]
    if [panel["width_mm"], panel["height_mm"]] != [facts["panel"]["width_mm"], facts["panel"]["height_mm"]]:
        raise Problem("The panel dimensions changed since this correction was retained.", 409)
    order = WorkOrder.parse(dict(facts["parameters"]))
    bench = Bench.load(library, panel_id, order)
    target, flat, source = _pair(bench)
    correction = solve(bench, target, flat, source, order.toolbox)
    display = _display_scale(correction, bench)
    corrected = coded(correction.corrected_rgb / display, correction.corrected_valid)
    asset = png("corrected-preview", corrected)
    summary = correction.summary(bench)
    return dict(image=asset.presentation(), summary=summary, parameters=order.public(),
                source_changes=_source_changes(library, facts),
                fingerprint=record["fingerprint"], record=record["id"],
                note=("Derived preview rebuilt from the retained correction settings and registrations. "
                      "No source photograph was rewritten; copied readings still carry their lower confidence."))
