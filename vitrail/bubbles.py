from math import hypot
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, find_objects, label
from scipy.optimize import linear_sum_assignment

from .bench import Result, masked_blur, ratio
from .camera import open_camera
from .contracts import Problem, keys, option, rounded, scalar
from .leads import CONNECTION
from .picture import csv_table, mark_image, mask_overlay, png
from .shape_checks import PatchShape


VIEWS = ("first", "second")
SIDE_CODES = ("A", "B")
STATUS_COLORS = {"confirmed": (137, 222, 190), "merged": (120, 206, 233),
                 "unresolved": (242, 133, 113), "first-only": (130, 180, 238),
                 "second-only": (236, 199, 120)}


def _frame_spots(frame, ctx, radius, cutoff, low, high, polarity):
    sigma = (radius/ctx.scale.step_y, radius/ctx.scale.step_x)
    baseline, support = masked_blur(frame.signal, frame.measurable, sigma, floor=.66)
    residue = frame.signal - baseline
    allowed = ctx.measured(frame) & support
    ctx.demand(allowed, floor=.01, pixels=24)
    candidate = allowed & ((-residue if polarity == "dark" else residue) > cutoff)
    codes, count = label(candidate, structure=CONNECTION)
    sizes = np.bincount(codes.ravel(), minlength=count+1)
    floor = max(4, int(low/ctx.scale.pixel_area))
    ceiling = max(floor, int(high/ctx.scale.pixel_area))
    keep = (sizes >= floor) & (sizes <= ceiling)
    keep[0] = False
    kept = keep[codes]
    interior = binary_erosion(frame.measurable, structure=CONNECTION)
    region_core = binary_erosion(ctx.selection, structure=CONNECTION)
    photo = frame.record["image"]
    spots = []
    for index, bounds in enumerate(find_objects(codes), 1):
        if bounds is None or not keep[index]:
            continue
        if len(spots) >= 300:
            raise Problem("One capture yields more than 300 candidate spots in this window. Select a smaller region.", 422)
        y0, y1 = bounds[0].start, bounds[0].stop
        x0, x1 = bounds[1].start, bounds[1].stop
        component = codes[bounds] == index
        shape = PatchShape.measure(component, (x0, y0, x1, y1), ctx.scale)
        yy, xx = np.where(component)
        yy, xx = yy+y0, xx+x0
        position = [rounded(shape.centroid[0]/ctx.scale.width_mm, 7),
                    rounded(shape.centroid[1]/ctx.scale.height_mm, 7)]
        camera = frame.alignment.to_camera([position])[0]
        spots.append({
            "area_mm2": rounded(sizes[index]*ctx.scale.pixel_area),
            "centroid_mm": [rounded(shape.centroid[0]), rounded(shape.centroid[1])],
            "panel_position": position,
            "panel_bounds": shape.normalized_bbox,
            "camera_position": [rounded(value, 7) for value in camera],
            "camera_pixel": [rounded(camera[0]*photo["width"]-.5), rounded(camera[1]*photo["height"]-.5)],
            "fitted_axes_mm": shape.public()["fitted_axes_mm"],
            "angle_degrees": shape.public()["angle_degrees"],
            "median_signal_difference": rounded(np.median(residue[bounds][component])),
            "near_unsupported_edge": bool(np.any(~interior[yy, xx])),
            "at_region_edge": bool(np.any(~region_core[yy, xx])),
        })
    return spots, kept


def _distance(a, b):
    return hypot(a["centroid_mm"][0]-b["centroid_mm"][0], a["centroid_mm"][1]-b["centroid_mm"][1])


def _overlap_mm(a, b, scale):
    ax0, ay0, ax1, ay1 = (a["panel_bounds"][0]*scale.width_mm, a["panel_bounds"][1]*scale.height_mm,
                          a["panel_bounds"][2]*scale.width_mm, a["panel_bounds"][3]*scale.height_mm)
    bx0, by0, bx1, by1 = (b["panel_bounds"][0]*scale.width_mm, b["panel_bounds"][1]*scale.height_mm,
                          b["panel_bounds"][2]*scale.width_mm, b["panel_bounds"][3]*scale.height_mm)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _clusters(first, second, tolerance, scale):
    near_first = [[j for j, b in enumerate(second)
                   if _distance(a, b) <= tolerance or _overlap_mm(a, b, scale)] for a in first]
    near_second = [[i for i, a in enumerate(first)
                    if _distance(a, b) <= tolerance or _overlap_mm(a, b, scale)] for b in second]
    seen_first, seen_second = set(), set()
    groups = []
    for start in range(len(first)):
        if start in seen_first:
            continue
        pending = [("first", start)]
        left, right = set(), set()
        while pending:
            side, index = pending.pop()
            if side == "first":
                if index in seen_first:
                    continue
                seen_first.add(index)
                left.add(index)
                pending.extend(("second", j) for j in near_first[index] if j not in seen_second)
            else:
                if index in seen_second:
                    continue
                seen_second.add(index)
                right.add(index)
                pending.extend(("first", i) for i in near_second[index] if i not in seen_first)
        groups.append((sorted(left), sorted(right)))
    groups.extend(([], [index]) for index in range(len(second)) if index not in seen_second)
    return groups

def _matched_pairs(left, right, first, second):
    cost = np.asarray([[_distance(first[i], second[j]) for j in right] for i in left])
    rows, cols = linear_sum_assignment(cost)
    return [(left[i], right[j], float(cost[i, j])) for i, j in zip(rows, cols, strict=True)]


def _pieces(spot, typical, piece_ratio=1.6, ceiling=6):
    if not typical:
        return 1
    share = spot["area_mm2"]/typical
    if share < piece_ratio:
        return 1
    return max(2, min(ceiling, int(round(share))))


def _basis(left, right, first, second):
    def describe(side, indices, spots):
        if not indices:
            return f"{side}: none"
        parts = [f"{spots[index]['area_mm2']:g} mm²" for index in indices]
        return f"{side}: {len(indices)} spot{'s' if len(indices) != 1 else ''} " + ", ".join(parts)
    return describe("first", left, first)+"; "+describe("second", right, second)


def _entry(ctx, cluster, status, left, right, first, second, pieces_a, pieces_b,
           low, high, pairing=None):
    members = [first[i] for i in left]+[second[j] for j in right]
    position = [float(np.mean([spot["panel_position"][0] for spot in members])),
                float(np.mean([spot["panel_position"][1] for spot in members]))]
    first_area = sum(first[i]["area_mm2"] for i in left)
    second_area = sum(second[j]["area_mm2"] for j in right)
    agreement = (rounded(min(first_area, second_area)/max(first_area, second_area))
                 if first_area and second_area else None)
    return {
        "cluster": cluster, "status": status,
        "count_low": int(low), "count_high": int(high),
        "count_definite": bool(low == high),
        "corroborated": bool(left and right),
        "panel_position": [rounded(value, 7) for value in position],
        "position_mm": [rounded(position[0]*ctx.scale.width_mm),
                        rounded(position[1]*ctx.scale.height_mm)],
        "spot_counts": [len(left), len(right)],
        "area_sum_mm2": [rounded(first_area), rounded(second_area)],
        "area_agreement": agreement,
        "pairing_distance_mm": rounded(pairing) if pairing is not None else None,
        "basis": _basis(left, right, first, second),
        "views": {VIEWS[0]: {"spots": [first[i] for i in left]},
                  VIEWS[1]: {"spots": [second[j] for j in right]}},
    }


def _census(ctx, first, second, tolerance):
    areas = [spot["area_mm2"] for spot in first+second]
    typical = float(np.median(areas)) if areas else None
    pieces_a = [_pieces(spot, typical) for spot in first]
    pieces_b = [_pieces(spot, typical) for spot in second]

    entries = []
    for cluster, (left, right) in enumerate(_clusters(first, second, tolerance, ctx.scale), 1):
        if not left or not right:
            pieces, side = (pieces_a, "first") if left else (pieces_b, "second")
            for index in (left or right):
                entries.append(_entry(ctx, cluster, f"{side}-only",
                                      [index] if side == "first" else [],
                                      [index] if side == "second" else [], first, second,
                                      pieces_a, pieces_b, 1, pieces[index]))
            continue
        if len(left) == len(right):
            for i, j, distance in _matched_pairs(left, right, first, second):
                possible = max(pieces_a[i], pieces_b[j])
                entries.append(_entry(ctx, cluster, "confirmed" if possible == 1 else "unresolved",
                                      [i], [j], first, second, pieces_a, pieces_b, 1, possible, distance))
            continue
        fused = any(pieces_a[i] > 1 for i in left) or any(pieces_b[j] > 1 for j in right)
        if fused:
            low = max(len(left), len(right))
            high = max(sum(pieces_a[i] for i in left), sum(pieces_b[j] for j in right), low)
            entries.append(_entry(ctx, cluster, "merged" if low == high else "unresolved",
                                  left, right, first, second, pieces_a, pieces_b, low, high))
            continue
        pieces_small = pieces_b if len(left) > len(right) else pieces_a
        pairs = _matched_pairs(left, right, first, second)
        used_a = {i for i, _, _ in pairs}
        used_b = {j for _, j, _ in pairs}
        for i, j, distance in pairs:
            entries.append(_entry(ctx, cluster, "confirmed", [i], [j], first, second,
                                  pieces_a, pieces_b, 1, 1, distance))
        extra = ([i for i in left if i not in used_a] if len(left) > len(right)
                 else [j for j in right if j not in used_b])
        side = "first" if len(left) > len(right) else "second"
        for index in extra:
            entries.append(_entry(ctx, cluster, f"{side}-only",
                                  [index] if side == "first" else [],
                                  [index] if side == "second" else [], first, second,
                                  pieces_a, pieces_b, 1, pieces_small[index]))
    entries.sort(key=lambda entry: (entry["position_mm"][1], entry["position_mm"][0]))
    for number, entry in enumerate(entries, 1):
        entry["group"] = number
    return entries, typical


def _box(spots, key="panel_bounds"):
    left = min(spot[key][0] for spot in spots)
    top = min(spot[key][1] for spot in spots)
    right = max(spot[key][2] for spot in spots)
    bottom = max(spot[key][3] for spot in spots)
    return left, top, right, bottom


def _label(entry):
    text = str(entry["group"])
    if entry["status"] == "unresolved":
        return text + f" [{entry['count_low']}-{entry['count_high']}?]"
    if entry["status"] == "merged":
        return text + f" [{entry['count_low']} in one spot]"
    if entry["status"].endswith("-only"):
        return text + " (one view)"
    return text


def _panel_view(frame, kept, entries, side):
    tinted = mask_overlay(frame.rgb, [(kept, (230, 199, 115), .45)], frame.present)
    shapes = []
    for entry in entries:
        spots = entry["views"][side]["spots"]
        if not spots:
            continue
        color = STATUS_COLORS[entry["status"]]
        left, top, right, bottom = _box(spots)
        shapes.append(([(left, top), (right, top), (right, bottom), (left, bottom), (left, top)],
                       color, _label(entry)))
    return mark_image(tinted, shapes)


def _panel_picture(ctx, view, side):
    return png(f"paired-census-{side}", view, ctx)


def _overview(frames, first_view, second_view, summary):
    tiles = []
    for view in (first_view, second_view):
        image = Image.fromarray(view)
        image.thumbnail((760, 620), Image.Resampling.LANCZOS)
        tiles.append(image)
    width = sum(image.width for image in tiles)+30
    height = max(image.height for image in tiles)+96
    sheet = Image.new("RGB", (width, height), (243, 238, 224))
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 10), "Paired bubble census | same reserved region, two captures", fill=(35, 42, 47))
    draw.text((14, 26), (f"{summary['objects_min']}-{summary['objects_max']} possible bubbles | "
                         f"{summary['confirmed_groups']} seen once in both | "
                         f"{summary['unresolved_groups']} open counts | "
                         f"{summary['first_only_groups']}+{summary['second_only_groups']} one-view"),
              fill=(61, 66, 70))
    x = 14
    for image, frame, side in zip(tiles, frames, VIEWS):
        sheet.paste(image, (x, 62))
        caption = f"{SIDE_CODES[0 if side == 'first' else 1]} | {frame.record['data']['title']}"
        draw.text((x, 46), caption[:60], fill=(35, 42, 47))
        x += image.width+16
    draw.text((14, height-22), "Green counted once | blue definite count inside a fused spot | "
                               "red open count range | amber one-view only", fill=(61, 66, 70))
    return png("paired-census-overview", np.asarray(sheet))


def _camera_canvas(contents, long_edge=1500):
    camera = open_camera(contents)
    preview = Image.open(BytesIO(camera.preview(limit=long_edge))).convert("RGB")
    return preview


def _camera_picture(contents, frame, entries, side, region, long_edge=1500):
    canvas = _camera_canvas(contents, long_edge)
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    scale = max(1, round(max(width, height)/1100))
    ring = frame.alignment.to_camera([region[:2], [region[2], region[1]],
                                      region[2:], [region[0], region[3]]])
    closing = [(x*width, y*height) for x, y in ring]
    draw.line(closing+[closing[0]], fill=(247, 244, 232), width=scale)
    for entry in entries:
        spots = entry["views"][side]["spots"]
        if not spots:
            continue
        color = STATUS_COLORS[entry["status"]]
        corners = []
        for spot in spots:
            left, top, right, bottom = spot["panel_bounds"]
            mapped = frame.alignment.to_camera([[left, top], [right, top], [right, bottom], [left, bottom]])
            for x, y in mapped:
                corners.append((x*width, y*height))
        left = min(point[0] for point in corners)
        top = min(point[1] for point in corners)
        right = max(point[0] for point in corners)
        bottom = max(point[1] for point in corners)
        if right-left < 8:
            left, right = (left+right)/2-4, (left+right)/2+4
        if bottom-top < 8:
            top, bottom = (top+bottom)/2-4, (top+bottom)/2+4
        draw.rectangle((left-2, top-2, right+2, bottom+2), outline=color, width=2*scale)
        draw.text((min(left+3, width-70), max(top-14, 2)), _label(entry), fill=color,
                  stroke_width=2, stroke_fill=(24, 27, 33))
    return png(f"paired-census-{side}-camera", np.asarray(canvas))


def paired_bubbles(ctx):
    first, second = ctx.two()
    opts = ctx.order.toolbox
    keys(opts, "radius_mm", "cutoff", "min_area", "max_area", "polarity", "match_mm")
    radius = scalar(opts.get("radius_mm", 1.5), "Glass neighborhood (mm)", .4, 12)
    cutoff = scalar(opts.get("cutoff", .042), "Local signal difference", .006, .6)
    low = scalar(opts.get("min_area", .27), "Minimum spot area (mm²)", .06, 50)
    high = scalar(opts.get("max_area", 30), "Maximum spot area (mm²)", low, 200)
    polarity = option(opts.get("polarity", "dark"), "Signal polarity", ("dark", "bright"))
    tolerance = scalar(opts.get("match_mm", 1.0), "Centroid match tolerance (mm)", .15, 6)

    first_spots, first_kept = _frame_spots(first, ctx, radius, cutoff, low, high, polarity)
    second_spots, second_kept = _frame_spots(second, ctx, radius, cutoff, low, high, polarity)
    first_spots.sort(key=lambda spot: (spot["centroid_mm"][1], spot["centroid_mm"][0]))
    second_spots.sort(key=lambda spot: (spot["centroid_mm"][1], spot["centroid_mm"][0]))
    entries, typical = _census(ctx, first_spots, second_spots, tolerance)
    for entry in entries:
        entry["views"]["first"]["capture"] = first.record["id"]
        entry["views"]["first"]["title"] = first.record["data"]["title"]
        entry["views"]["second"]["capture"] = second.record["id"]
        entry["views"]["second"]["title"] = second.record["data"]["title"]

    confirmed = [entry for entry in entries if entry["status"] == "confirmed"]
    merged = [entry for entry in entries if entry["status"] == "merged"]
    unresolved = [entry for entry in entries if entry["status"] == "unresolved"]
    first_only = [entry for entry in entries if entry["status"] == "first-only"]
    second_only = [entry for entry in entries if entry["status"] == "second-only"]
    shared = ctx.selection & first.measurable & second.measurable
    summary = dict(
        groups=len(entries),
        confirmed_groups=len(confirmed),
        merged_groups=len(merged),
        unresolved_groups=len(unresolved),
        first_only_groups=len(first_only), second_only_groups=len(second_only),
        objects_min=sum(entry["count_low"] for entry in entries),
        objects_max=sum(entry["count_high"] for entry in entries),
        first_candidates=len(first_spots), second_candidates=len(second_spots),
        typical_spot_area_mm2=rounded(typical) if typical else None,
        shared_support_fraction=ratio(shared, ctx.selection),
        retained_region_area_mm2=rounded(ctx.selection.sum()*ctx.scale.pixel_area),
        match_tolerance_mm=tolerance,
    )
    details = {
        **summary,
        "groups_detail": entries,
        "views": {
            "first": {"capture": first.record["id"], "title": first.record["data"]["title"],
                      "detected": len(first_spots)},
            "second": {"capture": second.record["id"], "title": second.record["data"]["title"],
                       "detected": len(second_spots)},
        },
        "region": ctx.order.region.public(),
        "polarity": polarity,
        "parameters": {"radius_mm": radius, "cutoff": cutoff, "min_area_mm2": low,
                       "max_area_mm2": high, "match_mm": tolerance},
        "note": ("Spots are paired in the registered panel plane when their centroids fall within the tolerance "
                 "or their recorded bounds overlap. A confirmed group is one bubble seen once in each capture and "
                 "is counted once. A merged group has a definite count, but at least one capture shows several bubbles as "
                 "one spot. An unresolved group keeps a possible count range because a large spot may hide more than "
                 "one bubble. A one-view group appears in only one capture and is listed without corroboration."),
    }
    group_table = csv_table(
        "bubble-register",
        ("group", "status", "count_low", "count_high", "count_definite", "first_spots", "second_spots",
         "x_mm", "y_mm", "first_area_mm2", "second_area_mm2", "area_agreement", "pairing_mm", "basis"),
        [(entry["group"], entry["status"], entry["count_low"], entry["count_high"],
          entry["count_low"] == entry["count_high"],
          *entry["spot_counts"], *entry["position_mm"], *entry["area_sum_mm2"], entry["area_agreement"],
          entry["pairing_distance_mm"], entry["basis"]) for entry in entries])
    members = []
    for entry in entries:
        for side in VIEWS:
            for number, spot in enumerate(entry["views"][side]["spots"], 1):
                members.append((entry["group"], side, entry["views"][side]["title"], number,
                                spot["area_mm2"], *spot["centroid_mm"], *spot["panel_position"],
                                *spot["camera_position"], *spot["camera_pixel"],
                                spot["near_unsupported_edge"], spot["at_region_edge"]))
    member_table = csv_table(
        "bubble-members",
        ("group", "view", "capture", "spot", "area_mm2", "x_mm", "y_mm", "panel_x", "panel_y",
         "camera_x", "camera_y", "camera_pixel_x", "camera_pixel_y", "near_unsupported_edge", "at_region_edge"),
        members)
    first_view = _panel_view(first, first_kept, entries, "first")
    second_view = _panel_view(second, second_kept, entries, "second")
    assets = (
        _overview((first, second), first_view, second_view, summary),
        _panel_picture(ctx, first_view, "first"),
        _panel_picture(ctx, second_view, "second"),
        _camera_picture(ctx.originals[0], first, entries, "first", ctx.order.region.public()),
        _camera_picture(ctx.originals[1], second, entries, "second", ctx.order.region.public()),
        group_table, member_table,
    )
    return Result("bubble-census", "Paired bubble census", summary, details, assets)
