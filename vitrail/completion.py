from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, map_coordinates

from .bench import Result, ratio
from .camera import apply_matrix, linear_to_srgb, normalized_grid
from .contracts import Problem, keys, label, object_data, option, pair, rounded, scalar
from .leads import CONNECTION
from .photometry import rgb_to_lab
from .picture import csv_table, mark_image, mask_overlay, png


BASE = (126, 196, 158)
SUPPLEMENT = (120, 166, 222)
UNSEEN = (58, 62, 72)
OUTSIDE = (92, 96, 104)
CHECKER = ((104, 108, 116), (78, 82, 90))
POLICIES = {"original": "Keep the whole-window capture in shared areas",
            "supplement": "Use the supplementary capture across its footprint"}
LABELS = {"original": "Original kept where shared", "supplement": "Supplementary used across its footprint"}


@dataclass(frozen=True)
class Match:
    name: str
    first: tuple
    second: tuple

    @classmethod
    def read(cls, value, index):
        row = object_data(value, "Matching marks")
        keys(row, "name", "first", "second")
        return cls(label(row.get("name", f"Mark {index}"), "Mark name", 64),
                   pair(row.get("first"), "First capture point"),
                   pair(row.get("second"), "Second capture point"))


def _matches(raw, model):
    if raw is None:
        raw = []
    if not isinstance(raw, list) or len(raw) > 40:
        raise Problem("Provide up to forty matching marks.")
    marks = [Match.read(row, index) for index, row in enumerate(raw, 1)]
    if len({mark.name for mark in marks}) != len(marks):
        raise Problem("Matching marks require different names.")
    least = {"similarity": 2, "affine": 3, "projective": 4}[model]
    if len(marks) < least:
        raise Problem(f"The {model} placement needs at least {least} matching marks.")
    return marks


def _similarity(source, target):
    design, outcome = [], []
    for (x, y), (u, v) in zip(source, target, strict=True):
        design.extend(((x, -y, 1, 0), (y, x, 0, 1)))
        outcome.extend((u, v))
    design = np.asarray(design, dtype=float)
    if np.linalg.matrix_rank(design) < 4:
        raise Problem("The marked points do not determine a single scale and rotation.")
    a, b, tx, ty = np.linalg.lstsq(design, np.asarray(outcome, dtype=float), rcond=None)[0]
    if abs(a) + abs(b) < 1e-6:
        raise Problem("The marked points collapse to a single location.")
    return np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]], dtype=float)


def _affine(source, target):
    design = np.column_stack((source, np.ones(len(source))))
    if np.linalg.matrix_rank(design) < 3:
        raise Problem("The marked points are too nearly collinear for an affine placement.")
    cx = np.linalg.lstsq(design, target[:, 0], rcond=None)[0]
    cy = np.linalg.lstsq(design, target[:, 1], rcond=None)[0]
    matrix = np.array([[cx[0], cx[1], cx[2]], [cy[0], cy[1], cy[2]], [0, 0, 1]], dtype=float)
    if np.linalg.det(matrix[:2, :2]) <= 0:
        raise Problem("The marked points reverse the supplementary capture.")
    return matrix


def _projective(source, target):
    design, outcome = [], []
    for (x, y), (u, v) in zip(source, target, strict=True):
        design.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y), (0, 0, 0, x, y, 1, -v * x, -v * y)))
        outcome.extend((u, v))
    design = np.asarray(design, dtype=float)
    if np.linalg.matrix_rank(design) < 8:
        raise Problem("The marked points are too nearly collinear for a perspective placement.")
    matrix = np.append(np.linalg.lstsq(design, np.asarray(outcome, dtype=float), rcond=None)[0], 1.0).reshape(3, 3)
    if np.linalg.cond(matrix) > 400:
        raise Problem("The proposed perspective placement is too unstable.")
    return matrix


def _fit(model, source, target):
    matrix = {"similarity": _similarity, "affine": _affine, "projective": _projective}[model](source, target)
    corners = apply_matrix(matrix, np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float))
    if not np.all(np.isfinite(corners)):
        raise Problem("The marked points do not place the supplementary capture.")
    area = .5 * abs(sum(corners[i, 0] * corners[(i + 1) % 4, 1] - corners[(i + 1) % 4, 0] * corners[i, 1]
                        for i in range(4)))
    if area < 1e-6:
        raise Problem("The marked points collapse the supplementary capture.")
    if np.any(np.stack(([0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1])) @ matrix[2] <= 0):
        raise Problem("The perspective placement falls behind the capture plane.")
    return matrix


def _residuals(marks, matrix, scale):
    placed = apply_matrix(matrix, np.array([mark.second for mark in marks], dtype=float))
    difference = (placed - np.array([mark.first for mark in marks], dtype=float)) * np.array([scale.width_mm,
                                                                                              scale.height_mm])
    distances = np.linalg.norm(difference, axis=1)
    rows = [dict(name=mark.name, first=[rounded(v) for v in mark.first], second=[rounded(v) for v in mark.second],
                 placed=[rounded(v) for v in point], offset_mm=[rounded(v) for v in mm], distance_mm=rounded(distance))
            for mark, point, mm, distance in zip(marks, placed, difference, distances, strict=True)]
    return rows, float(np.sqrt(np.mean(distances ** 2))), float(np.max(distances))


def _warp(frame, matrix, width, height):
    points = apply_matrix(matrix, normalized_grid(width, height).reshape(-1, 2)).reshape(height, width, 2)
    inside = np.all((points >= 0) & (points <= 1), axis=2)
    x = points[..., 0] * width - .5
    y = points[..., 1] * height - .5
    coords = np.stack((np.clip(y, -1, height), np.clip(x, -1, width)))
    weight = map_coordinates(frame.present.astype(np.float32), coords, order=1, mode="constant", cval=0.0)
    premultiplied = frame.rgb * frame.present[..., None]
    rgb = np.stack([map_coordinates(premultiplied[..., channel], coords, order=1, mode="constant", cval=0.0)
                    for channel in range(3)], axis=-1)
    rgb = rgb / np.maximum(weight[..., None], 0.001)
    present = inside & (weight >= .5)
    rgb[~present] = 0
    clipped = map_coordinates(frame.clipped.astype(np.float32), coords, order=0, mode="constant", cval=0.0) > .5
    return rgb.astype(np.float32), present, clipped & present


def _plan(base, supplement, selection, policy):
    if policy == "supplement":
        fill = selection & supplement["measurable"]
    else:
        fill = selection & supplement["measurable"] & ~base["measurable"]
    used_supplement = fill
    used_base = base["present"] & ~used_supplement
    unseen = selection & ~used_supplement & ~base["present"]
    return dict(used_base=used_base, used_supplement=used_supplement, unseen=unseen)


def _rendering(base_rgb, supplement_rgb, overlap):
    base = base_rgb[overlap]
    supplement = supplement_rgb[overlap]
    difference = np.linalg.norm(rgb_to_lab(np.minimum(base, 1)) - rgb_to_lab(np.minimum(supplement, 1)), axis=1)
    light_a = base @ np.array([.2126, .7152, .0722])
    light_b = supplement @ np.array([.2126, .7152, .0722])
    comparable = (light_a > 1e-5) & (light_b > 1e-5)
    shift = rounded(float(np.median(np.log2(light_b[comparable] / light_a[comparable]))), 3) if comparable.any() else None
    return dict(median_delta_e=rounded(float(np.median(difference))),
                high_delta_e=rounded(float(np.percentile(difference, 90))), median_ev_shift=shift,
                pixels=int(overlap.sum()))


def _rgba(rgb, used):
    layer = np.concatenate((linear_to_srgb(rgb), np.uint8(255 * used[..., None])), axis=2)
    layer[~used] = 0
    return layer


def _checker(rgb, used, unseen):
    rows, cols = np.mgrid[:used.shape[0], :used.shape[1]]
    parity = (cols // 13 + rows // 13) % 2
    pad = np.where(parity[..., None], np.array(CHECKER[0]), np.array(CHECKER[1])).astype(np.uint8)
    picture = np.where(used[..., None], linear_to_srgb(rgb), pad)
    picture[unseen] = UNSEEN
    return picture


def _source_map(selection, plan):
    image = np.zeros((*selection.shape, 3), dtype=np.uint8)
    image[:] = OUTSIDE
    image[selection] = UNSEEN
    image[plan["used_base"] & selection] = BASE
    image[plan["used_supplement"]] = SUPPLEMENT
    return image


def _sheet(columns):
    layers = [layer for column in columns for layer in column]
    height = max(layer.shape[0] for layer, _ in layers)
    width = max(layer.shape[1] for layer, _ in layers)
    rows = max(len(column) for column in columns)
    gap, caption, margin = 14, 24, 18
    canvas = Image.new("RGB", (2 * margin + len(columns) * width + (len(columns) - 1) * gap,
                               28 + rows * (caption + height + gap)), (238, 234, 222))
    draw = ImageDraw.Draw(canvas)
    for index, column in enumerate(columns):
        x = margin + index * (width + gap)
        for row, (layer, words) in enumerate(column):
            y = 28 + row * (caption + height + gap)
            draw.text((x, y), words[:64], fill=(58, 64, 70))
            canvas.paste(Image.fromarray(layer), (x, y + caption))
    return np.asarray(canvas)


def _region(plan, key, scale, selection):
    return dict(pixels=int(plan[key].sum()), area_mm2=rounded(plan[key].sum() * scale.pixel_area),
                fraction=ratio(plan[key], selection))


def _placement_table(base, supplement, plans, scale, selection):
    rows = []
    for name, plan in plans.items():
        for slot, label_text, frame, mask in (
                ("first", "whole-window original", base, plan["used_base"]),
                ("second", "supplementary capture", supplement, plan["used_supplement"]),
                ("unseen", "not captured by either", None, plan["unseen"])):
            rows.append((name, label_text, frame.record["id"] if frame else "", frame.record["data"]["title"] if frame else "",
                         int(mask.sum()), rounded(mask.sum() * scale.pixel_area), ratio(mask, selection)))
    return csv_table("placement", ("policy", "placement", "capture", "title", "pixels", "area_mm2",
                                   "fraction_of_window"), rows)


def supplementary_completion(ctx):
    base, supplement = ctx.two()
    opts = ctx.order.toolbox
    keys(opts, "controls", "model", "tolerance_mm", "color_tolerance", "overlap")
    model = option(opts.get("model", "affine"), "Placement model", ("similarity", "affine", "projective"))
    tolerance = scalar(opts.get("tolerance_mm", 1.5), "Largest accepted mark residual (mm)", .1, 40)
    color_tolerance = scalar(opts.get("color_tolerance", 7), "Rendering difference limit (delta E)", 1, 45)
    requested = option(opts.get("overlap", "auto"), "Shared-area placement", ("auto", "original", "supplement"))
    marks = _matches(opts.get("controls"), model)
    matrix = _fit(model, np.array([mark.second for mark in marks], dtype=float),
                  np.array([mark.first for mark in marks], dtype=float))
    rows, rms, largest = _residuals(marks, matrix, ctx.scale)
    if largest > tolerance:
        raise Problem(f"A marked pair disagrees by {largest:.2f} mm, beyond the {tolerance:g} mm limit.", 422)

    width, height = ctx.scale.width_px, ctx.scale.height_px
    warped_rgb, warped_present, warped_clipped = _warp(supplement, matrix, width, height)
    supplement_view = dict(rgb=warped_rgb, present=warped_present, measurable=warped_present & ~warped_clipped,
                           clipped=warped_clipped)
    base_view = dict(rgb=base.rgb, present=base.present, measurable=base.measurable, clipped=base.clipped)
    selection = ctx.selection
    overlap = selection & base_view["measurable"] & supplement_view["measurable"]
    ctx.demand(overlap, floor=.01, pixels=36)
    if not (selection & supplement_view["measurable"]).any():
        raise Problem("The supplementary capture does not land over this window.", 422)

    rendering = _rendering(base_view["rgb"], supplement_view["rgb"], overlap)
    differs = rendering["median_delta_e"] is not None and rendering["median_delta_e"] > color_tolerance
    needed = bool((selection & supplement_view["measurable"] & ~base_view["measurable"]).any())
    plans = {name: _plan(base_view, supplement_view, selection, name) for name in POLICIES}
    ambiguous = bool(requested == "auto" and differs and needed)
    chosen = None if ambiguous else (requested if requested != "auto" else "original")

    merged = {name: np.where(plan["used_supplement"][..., None], supplement_view["rgb"], base_view["rgb"])
              for name, plan in plans.items()}
    assets = []
    if ambiguous:
        completed = {name: _checker(merged[name], plan["used_base"] | plan["used_supplement"], plan["unseen"])
                     for name, plan in plans.items()}
        reached = {name: _source_map(selection, plan) for name, plan in plans.items()}
        assets.append(png("shared-area-choices", _sheet([
            [(completed["original"], "Whole-window kept in shared areas"),
             (reached["original"], "Where that placement reaches")],
            [(completed["supplement"], "Supplementary used across its footprint"),
             (reached["supplement"], "Where that placement reaches")]]), crop=False))
        for name in ("original", "supplement"):
            assets.append(png(f"option-{name}", completed[name], crop=False))
            assets.append(png(f"reach-{name}", reached[name], crop=False))
    else:
        plan = plans[chosen]
        used = plan["used_base"] | plan["used_supplement"]
        assets.append(png("completed-panel", _checker(merged[chosen], used, plan["unseen"]), crop=False))
        assets.append(png("completed-transparency", _rgba(merged[chosen], used), crop=False))
        assets.append(png("source-map", _source_map(selection, plan), crop=False))

    footprint = warped_present & ~binary_erosion(warped_present, structure=CONNECTION)
    outline = (plans[chosen]["unseen"] if chosen else
               plans["original"]["unseen"] & plans["supplement"]["unseen"])
    canvas = linear_to_srgb(base.rgb).copy()
    canvas[outline] = (176, 96, 96)
    canvas[~base.present] = UNSEEN
    canvas[footprint] = canvas[footprint] * .18 + np.asarray(SUPPLEMENT) * .82
    assets.append(png("control-points", mark_image(canvas, [
        ([row["first"]], (137, 222, 190) if row["distance_mm"] <= tolerance else (242, 133, 113), row["name"])
        for row in rows]), crop=False))
    assets.append(_placement_table(base, supplement, plans, ctx.scale, selection))
    assets.append(csv_table("matching-marks", ("name", "first_x", "first_y", "second_x", "second_y",
                                               "placed_x", "placed_y", "distance_mm"),
                            [(row["name"], *row["first"], *row["second"], *row["placed"], row["distance_mm"])
                             for row in rows]))

    chosen_plan = plans[chosen] if chosen else None
    summary = dict(marks=len(rows), registration_rms_mm=rounded(rms), largest_mark_mm=rounded(largest),
                   shared_fraction=ratio(overlap, selection), rendering_delta_e=rendering["median_delta_e"],
                   color_limit=color_tolerance, choice_required=ambiguous, chosen_policy=chosen,
                   second_capture_area_mm2=rounded(chosen_plan["used_supplement"].sum() * ctx.scale.pixel_area)
                   if chosen_plan else None,
                   unseen_area_mm2=rounded(chosen_plan["unseen"].sum() * ctx.scale.pixel_area) if chosen_plan else None,
                   unseen_fraction=ratio(chosen_plan["unseen"], selection) if chosen_plan else None)
    details = {**summary, "model": model, "tolerance_mm": tolerance, "color_tolerance": color_tolerance,
               "shared_area_request": requested, "shared_area_differs": differs, "supplement_needed": needed,
               "rendering": rendering, "matching_marks": rows,
               "policies": {name: dict(label=POLICIES[name],
                                       first_capture=_region(plan, "used_base", ctx.scale, selection),
                                       second_capture=_region(plan, "used_supplement", ctx.scale, selection),
                                       unseen=_region(plan, "unseen", ctx.scale, selection))
                           for name, plan in plans.items()},
               "sources": [
                   dict(role="whole-window original", id=base.record["id"], title=base.record["data"]["title"],
                        revision=base.record["revision"], original=base.record["original"],
                        alignment_id=base.alignment_ref["id"] if base.alignment_ref else None),
                   dict(role="supplementary capture", id=supplement.record["id"], title=supplement.record["data"]["title"],
                        revision=supplement.record["revision"], original=supplement.record["original"],
                        alignment_id=supplement.alignment_ref["id"] if supplement.alignment_ref else None)],
               "note": ("The whole-window capture is kept wherever it is measurable; the supplementary capture fills "
                        "only unsupported or saturated areas. Ground neither capture reaches stays empty."
                        if chosen != "supplement" else
                        "The supplementary capture is used across its whole footprint; anything outside that footprint "
                        "and unseen by both captures stays empty.") + " Both sources, revisions and original checksums "
                       "are recorded for review. This is a derived composite; the source files are unchanged."}
    blocked = ("The shared area renders differently in the two captures. Choose a placement before retaining."
               if ambiguous else None)
    return Result("completion", "Supplementary panel completion", summary, details, tuple(assets), blocked)
