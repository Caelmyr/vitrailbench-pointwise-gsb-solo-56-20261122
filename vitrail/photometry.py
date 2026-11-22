from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, option, rounded, scalar
from .picture import BRIGHT, chart, csv_table, map_asset, mask_overlay, mark_image, png


PHOTOMETRIC_FLOOR = 0.001
LOG_EPS = 0.00005


def _histogram(signal, valid, bins, limit):
    counts, edges = np.histogram(signal[valid], bins=bins, range=(0, limit))
    outlying = int(np.sum(valid & (signal > limit)))
    intervals = [{"from": rounded(edges[i], 6), "to": rounded(edges[i+1], 6), "pixels": int(count)}
                 for i, count in enumerate(counts)]
    return intervals, outlying


def _tile(index, x0, y0, x1, y1, signal, good, region):
    allowed = good[y0:y1, x0:x1]
    chosen = region[y0:y1, x0:x1]
    total = int(chosen.sum())
    n = int(allowed.sum())
    return dict(tile=index, raster_bounds=[x0, y0, x1, y1], observed=n,
                selected=total, coverage=rounded(n/max(total, 1)),
                p10_median_p90=percentiles(signal[y0:y1, x0:x1], allowed, (10, 50, 90), 8))


def transmission(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "bins", "display_ceiling", "tiles")
    bins = integer(opts.get("bins", 20), "Bins", 8, 64)
    limit = scalar(opts.get("display_ceiling", 1), "Display ceiling", 0.05, 6)
    cells = integer(opts.get("tiles", 4), "Tiles per dimension", 2, 10)
    good = ctx.demand(ctx.measured(frame))
    signal = frame.signal
    histogram, over = _histogram(signal, good, bins, limit)
    height, width = signal.shape
    x0, y0, x1, y1 = ctx.order.region.indices(width, height)
    xs = np.linspace(x0, x1, cells + 1).round().astype(int)
    ys = np.linspace(y0, y1, cells + 1).round().astype(int)
    tiles = []
    lines = []
    for row in range(cells):
        for col in range(cells):
            item = _tile(f"{row+1}-{col+1}", int(xs[col]), int(ys[row]), int(xs[col+1]), int(ys[row+1]),
                         signal, good, ctx.selection)
            tiles.append(item)
            lines.append(([(xs[col]/width, ys[row]/height), (xs[col+1]/width, ys[row]/height),
                           (xs[col+1]/width, ys[row+1]/height), (xs[col]/width, ys[row+1]/height),
                           (xs[col]/width, ys[row]/height)], BRIGHT, item["tile"]))
    stats = dict(observed_fraction=ratio(good, ctx.selection), clipped_pixels=int((ctx.selection & frame.clipped).sum()),
                 quantiles=percentiles(signal, good), above_display_ceiling=over)
    rendered = map_asset("signal", signal, good, ctx, low=0, high=limit)
    preview = png("tile-grid", mark_image(mask_overlay(frame.rgb, [], frame.present), lines), ctx)
    table = csv_table("tiles", ["tile", "raster_left", "raster_top", "raster_right", "raster_bottom",
                                "observed_pixels", "selected_pixels", "coverage", "p10", "median", "p90"],
                      [[row["tile"], *row["raster_bounds"], row["observed"], row["selected"], row["coverage"],
                        *row["p10_median_p90"]] for row in tiles])
    return Result("transmission", "Transmitted signal", stats, {**stats, "histogram": histogram, "tiles": tiles,
                  "note": "Camera-derived linear signal after declared exposure and black subtraction."}, (rendered, preview, table))


@dataclass(frozen=True)
class ReferenceReading:
    value: float
    spread: float
    samples: int
    box: Box

    def public(self):
        return dict(median=rounded(self.value), central_spread=rounded(self.spread), samples=self.samples,
                    region=self.box.public())


def _read_reference(frame, ctx, box, minimum=36):
    signal = frame.signal
    mask = box.mask(ctx.scale.width_px, ctx.scale.height_px) & frame.measurable
    samples = signal[mask & (signal > PHOTOMETRIC_FLOOR)]
    if samples.size < minimum:
        raise Problem("Use a larger, adequately lit reference area without clipping.", 422)
    q10, target, q90 = np.percentile(samples, (10, 50, 90))
    return ReferenceReading(float(target), float(q90-q10), int(samples.size), box)


def density(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "reference", "max_density", "allow_brighter")
    box = Box.parse(opts.get("reference", [0.44, 0.38, 0.59, 0.49]), "Reference")
    ceiling = scalar(opts.get("max_density", 2.5), "Density display limit", 0.3, 5)
    brighter = option(opts.get("allow_brighter", "signed"), "Brighter result", ("signed", "clamp"))
    ref = _read_reference(frame, ctx, box)
    signal = frame.signal
    good = ctx.demand(ctx.measured(frame) & (signal > PHOTOMETRIC_FLOOR))
    measured = np.log10(ref.value / np.maximum(signal, LOG_EPS))
    values = np.maximum(measured, 0) if brighter == "clamp" else measured
    left, right = (-ceiling/3, ceiling) if brighter == "signed" else (0, ceiling)
    contour = map_asset("density", values, good, ctx, low=left, high=right, palette="ink")
    x0, y0, x1, y1 = box.public()
    picture = mark_image(mask_overlay(frame.rgb, [], frame.present),
                         [([(x0,y0), (x1,y0), (x1,y1), (x0,y1), (x0,y0)], BRIGHT, "Reference")])
    density_hist, excess = _histogram(np.maximum(values, 0), good, 16, ceiling)
    stats = dict(density_quantiles=percentiles(values, good), observed_fraction=ratio(good, ctx.selection),
                 reference=ref.public(), brighter_pixels=int(np.sum(good & (measured < 0))),
                 above_display_ceiling=excess)
    return Result("density", "Reference optical density", stats,
                  {**stats, "histogram": density_hist, "control": brighter,
                   "note": "Relative optical density from the chosen bright patch, not a spectrophotometer result."},
                  (contour, png("reference", picture, ctx, crop=False)))


def _smooth_plane(values, support, radius, scale):
    return masked_blur(values, support, (radius/scale.step_y, radius/scale.step_x), floor=0.55)


def flatness(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "radius_mm", "allowed_delta_ev", "reference")
    radius = scalar(opts.get("radius_mm", 8), "Neighborhood radius in mm", 1, 55)
    tolerance = scalar(opts.get("allowed_delta_ev", 0.2), "Allowed EV deviation", 0.03, 1.5)
    box = Box.parse(opts.get("reference", [0.4, 0.35, 0.6, 0.65]), "Reference")
    source = frame.measurable & (frame.signal > PHOTOMETRIC_FLOOR)
    good = ctx.demand(source & ctx.selection)
    ref = _read_reference(frame, ctx, box)
    field, neighborhood = _smooth_plane(frame.signal, source, radius, ctx.scale)
    valid = good & neighborhood & (field > PHOTOMETRIC_FLOOR)
    deviation = np.log2(np.maximum(field, LOG_EPS) / ref.value)
    bright = (deviation > tolerance) & valid
    dim = (deviation < -tolerance) & valid
    result = dict(reference=ref.public(), supported_fraction=ratio(valid, ctx.selection),
                  dim_fraction=ratio(dim, valid), bright_fraction=ratio(bright, valid),
                  deviation_quantiles_ev=percentiles(deviation, valid), reference_radius_mm=radius)
    diagnostics = mask_overlay(frame.rgb, [(bright, (242, 113, 71), 0.55), (dim, (87, 141, 226), 0.58)], frame.present)
    return Result("flatness", "Backlight level survey", result,
                  {**result, "threshold_ev": tolerance,
                   "note": "A field observation, without changing the originals or applying a correction."},
                  (map_asset("backlight-ev", deviation, valid, ctx, low=-tolerance*2, high=tolerance*2, palette="diverging"),
                   png("tolerance", diagnostics, ctx)))


def _safe_divide(numerator, denominator):
    return numerator / np.maximum(denominator, PHOTOMETRIC_FLOOR)


def registered_compare(ctx):
    original, comparison = ctx.two()
    opts = ctx.order.toolbox
    keys(opts, "tolerance_ev", "reference", "mix")
    mix = scalar(opts.get("mix", 0.5), "Wipe position", 0.05, 0.95)
    band = scalar(opts.get("tolerance_ev", 0.25), "Signal comparison tolerance (EV)", 0.02, 3)
    box = Box.parse(opts.get("reference"), "Common reference")
    reference = box.mask(ctx.scale.width_px, ctx.scale.height_px)
    light_a, light_b = original.signal, comparison.signal
    overlap = original.measurable & comparison.measurable
    overlap &= (light_a >= PHOTOMETRIC_FLOOR) & (light_b >= PHOTOMETRIC_FLOOR)
    common = ctx.demand(ctx.selection & overlap)
    radiometric_offset = np.log2(_safe_divide(light_b, light_a))
    if int((reference & overlap).sum()) < 36:
        raise Problem("The common reference needs at least 36 measurable pixels.", 422)
    center = float(np.median(radiometric_offset[reference & overlap]))
    balanced = radiometric_offset - center
    changed = common & (np.abs(balanced) >= band)
    x = ctx.scale.width_px
    wipe_mask = np.broadcast_to(np.arange(x)[None, :] < round(x * mix), common.shape)
    merged = np.where(wipe_mask[..., None], original.rgb, comparison.rgb)
    merged_valid = np.where(wipe_mask, original.present, comparison.present)
    only_a = ctx.selection & original.present & ~comparison.present
    only_b = ctx.selection & comparison.present & ~original.present
    summary = dict(shared_fraction=ratio(common, ctx.selection), offset_ev=rounded(center),
                   affected_fraction=ratio(changed, common), deviation_ev=percentiles(balanced, common))
    details = {**summary, "only_first_fraction": ratio(only_a, ctx.selection),
               "only_second_fraction": ratio(only_b, ctx.selection), "cut": mix,
               "reference": box.public(), "deviation_limit_ev": band,
               "note": "Same coordinate plane, normalized by an observed common patch. Color and shape can both change this reading."}
    return Result("compare", "Registered capture comparison", summary, details,
                  (png("wipe", mask_overlay(merged, [], merged_valid), ctx),
                   map_asset("ev-change", balanced, common, ctx, low=-band*2, high=band*2, palette="diverging"),
                   png("overlap", mask_overlay(original.rgb, [(only_a, (236, 120, 75), .72),
                                                              (only_b, (116, 160, 223), .72)], original.present), ctx)))


def rgb_to_lab(rgb):
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = (rgb @ matrix.T) / np.array([0.95047, 1.0, 1.08883])
    threshold = 6/29
    transformed = np.where(xyz > threshold**3, np.cbrt(xyz), xyz / (3*threshold**2) + 4/29)
    x, y, z = np.moveaxis(transformed, -1, 0)
    return np.stack((116*y-16, 500*(x-y), 200*(y-z)), axis=-1)


def color_field(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "reference", "threshold")
    ref = Box.parse(opts.get("reference", [0.44, 0.38, 0.59, 0.49]), "Color reference")
    threshold = scalar(opts.get("threshold", 18.0), "Color distance display limit", 2, 90)
    source = frame.measurable & (frame.signal > PHOTOMETRIC_FLOOR)
    good = ctx.demand(source & ctx.selection)
    ref_mask = ref.mask(ctx.scale.width_px, ctx.scale.height_px) & source
    if np.sum(ref_mask) < 25:
        raise Problem("Color reference contains fewer than 25 measurable pixels.", 422)
    lab = rgb_to_lab(np.minimum(frame.rgb, 1))
    midpoint = np.median(lab[ref_mask], axis=0)
    delta = np.linalg.norm(lab - midpoint, axis=2)
    chroma = np.linalg.norm(lab[..., 1:], axis=2)
    a, b = lab[..., 1], lab[..., 2]
    hue = np.arctan2(b, a)
    bins = np.linspace(-np.pi, np.pi, 19)
    histogram, _ = np.histogram(hue[good & (chroma > 7)], bins=bins)
    hue_bins = [(round(float(np.degrees(bins[i]))), int(count)) for i, count in enumerate(histogram)]
    summary = dict(reference_lab=[rounded(value) for value in midpoint],
                   distance_quantiles=percentiles(delta, good),
                   chroma_quantiles=percentiles(chroma, good), measured_fraction=ratio(good, ctx.selection))
    reports = [map_asset("lab-distance", delta, good, ctx, low=0, high=threshold, palette="thermal"),
               map_asset("chroma", chroma, good, ctx, low=0, high=70, palette="field"),
               csv_table("hue", ["lower_angle_degrees", "measured_pixels"], hue_bins)]
    return Result("color", "Transmission color map", summary, {**summary, "reference": ref.public(),
                  "distance": "CIELAB Euclidean", "hue_histogram": hue_bins}, tuple(reports))


def _polyline(points, sz, sample_spacing):
    xy = np.asarray(points, dtype=float)
    scale = np.asarray([sz.width_mm, sz.height_mm])
    physical = xy * scale
    lengths = np.linalg.norm(np.diff(physical, axis=0), axis=1)
    if np.sum(lengths) < 0.2 or np.any(lengths < 0.08):
        raise Problem("The line or one of its segments is too short.")
    total = float(lengths.sum())
    samples = min(1600, max(3, int(np.ceil(total/sample_spacing))+1))
    cumulative = np.concatenate(([0], np.cumsum(lengths)))
    along = np.linspace(0, total, samples)
    interpolated = np.stack([np.interp(along, cumulative, xy[:, i]) for i in range(2)], axis=-1)
    return along, interpolated


def optical_section(ctx):
    from .contracts import pair
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "path", "spacing_mm")
    user = opts.get("path", [[0.15, 0.5], [0.85, 0.5]])
    if not isinstance(user, list) or not 2 <= len(user) <= 30:
        raise Problem("A section expects between two and thirty points.")
    points = [pair(part, "Line point") for part in user]
    distance, read_at = _polyline(points, ctx.scale,
                                  scalar(opts.get("spacing_mm", .7), "Spacing (mm)", .1, 20))
    signal = frame.signal
    selections = ctx.measured(frame)
    bytes_code = []
    capture_points = frame.alignment.to_camera(read_at)
    for step, point, camera in zip(distance, read_at, capture_points, strict=True):
        col, row = ctx.scale.point_px(point)
        valid = bool(selections[row, col])
        bytes_code.append({
            "mm": rounded(step), "panel": [rounded(value, 7) for value in point],
            "camera": [rounded(value, 7) for value in camera] if valid else None,
            "luma": rounded(signal[row, col]) if valid else None,
        })
    summary = dict(length_mm=rounded(distance[-1]), samples=len(distance),
                   measured_samples=sum(item["luma"] is not None for item in bytes_code))
    visual = chart("profile", distance, [("Transmitted signal", (171, 75, 56), [row["luma"] for row in bytes_code])],
                   "Transmitted light along marked line", "Linear")
    lines = [(points, (253, 228, 119), "Section")]
    filename = csv_table("samples", ("distance_mm", "panel_x", "panel_y", "camera_x", "camera_y", "linear"),
                         [[row["mm"], *row["panel"], *(row["camera"] or [None, None]), row["luma"]] for row in bytes_code])
    return Result("section", "Optical section", summary, {**summary, "path": points, "readings": bytes_code},
                  (visual, png("line", mark_image(mask_overlay(frame.rgb, [], frame.present), lines), ctx), filename))
