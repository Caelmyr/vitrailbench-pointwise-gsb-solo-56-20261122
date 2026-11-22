from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion, convolve,
                           distance_transform_edt, find_objects, label as components)

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Problem, integer, keys, rounded, scalar
from .picture import colorize, csv_table, map_asset, mark_image, mask_overlay, png


CONNECTION = np.ones((3, 3), bool)
CCROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
LEAD = (236, 186, 92)
CENTERLINE = (236, 109, 63)


def neighbors(mask):
    kernel = np.ones((3, 3), dtype=np.int16)
    kernel[1, 1] = 0
    return convolve(mask.astype(np.int16), kernel, mode="constant", cval=0)


def clean_regions(mask, min_area):
    cc, count = components(mask, structure=CONNECTION)
    if count == 0:
        return mask.copy()
    keep = np.bincount(cc.ravel(), minlength=count+1) >= min_area
    keep[0] = False
    return keep[cc]


def _thinning_phase(mask, phase):
    u = np.pad(mask, 1, constant_values=False)
    p = [u[:-2, 1:-1], u[:-2, 2:], u[1:-1, 2:], u[2:, 2:], u[2:, 1:-1], u[2:, :-2], u[1:-1, :-2], u[:-2, :-2]]
    crossing = np.zeros(mask.shape, np.uint8)
    count = np.zeros(mask.shape, np.uint8)
    for i, bit in enumerate(p):
        count += bit
        crossing += (~bit) & p[(i+1) % 8]
    if phase == 0:
        pairs = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
    else:
        pairs = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
    return mask & (count >= 2) & (count <= 6) & (crossing == 1) & pairs


def centerline(mask, rounds=50):
    line = mask.copy()
    for _ in range(rounds):
        first = _thinning_phase(line, 0)
        line[first] = False
        second = _thinning_phase(line, 1)
        line[second] = False
        if not first.any() and not second.any():
            break
    return line


def _branch_pixels(line):
    return line & (neighbors(line) >= 3)


def _end_pixels(line):
    return line & (neighbors(line) == 1)


def _threshcut(frame, ctx, ceiling, radius, min_area, close_mm):
    light = frame.signal
    allowed = ctx.demand(ctx.measured(frame))
    sigma = max(radius/ctx.scale.step_y, .7), max(radius/ctx.scale.step_x, .7)
    surround, buffered = masked_blur(light, frame.measurable, sigma, floor=.55)
    # Consistent with a local silhouette under a bright transmitted background.
    dense = (light < ceiling) & (light < 0.62 * surround) & allowed & buffered
    if close_mm:
        count = min(8, max(1, round(close_mm / min(ctx.scale.step_x, ctx.scale.step_y))))
        dense = binary_closing(dense, structure=CCROSS, iterations=count) & allowed
    return clean_regions(dense, min_area), allowed


def _mm_xy(index, scale):
    y, x = index
    return ((x+.5)*scale.step_x, (y+.5)*scale.step_y)


def _point_index(y_x, scale):
    return [rounded((y_x[1]+.5)/scale.width_px, 7), rounded((y_x[0]+.5)/scale.height_px, 7)]


def _make_edges(line):
    indices = np.argwhere(line)
    grouped = {}
    for y,x in indices:
        grouped[(int(y), int(x))] = set()
    for point in grouped:
        y, x = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = (y+dy, x+dx)
                if q == point or q not in grouped:
                    continue
                if dy and dx and ((y, x+dx) in grouped or (y+dy, x) in grouped):
                    # A diagonal across another adjacent path is not a separate branch.
                    continue
                grouped[point].add(q)
    return grouped


def _edge(point, other):
    return min(point, other), max(point, other)


def _walk(start, other, graph, seen, junctions):
    path = [start]
    while True:
        point = path[-1]
        edge = _edge(point, other)
        if edge in seen:
            break
        seen.add(edge)
        path.append(other)
        if other == start or other in junctions:
            break
        choices = [candidate for candidate in graph[other] if _edge(other, candidate) not in seen]
        if len(choices) != 1:
            break
        other = choices[0]
    return path


def graph_paths(line, max_paths=3000):
    graph = _make_edges(line)
    junctions = {point for point, adj in graph.items() if len(adj) != 2}
    visited = set()
    paths = []
    for group in (sorted(junctions), sorted(set(graph)-junctions)):
        for node in group:
            for other in sorted(graph[node]):
                if _edge(node, other) in visited:
                    continue
                run = _walk(node, other, graph, visited, junctions)
                if len(run) > 1:
                    paths.append(run)
                    if len(paths) > max_paths:
                        raise Problem("The detected network is too fragmented. Tighten the region or threshold.", 422)
    return graph, paths


def _length(points, scale):
    mm = np.array([_mm_xy(p, scale) for p in points])
    return float(np.linalg.norm(np.diff(mm, axis=0), axis=1).sum())


def _average_direction(points, scale):
    physical = np.array([_mm_xy(p, scale) for p in points])
    center = physical - physical.mean(axis=0)
    cov = center.T @ center
    _, axes = np.linalg.eigh(cov)
    direction = axes[:, -1]
    return rounded(np.degrees(np.arctan2(direction[1], direction[0])) % 180)


def _compact(path, scale, max_points=26):
    indices = np.linspace(0, len(path)-1, min(len(path), max_points)).round().astype(int)
    return [_point_index(path[i], scale) for i in np.unique(indices)]


def _row(i, path, width, support, scale):
    xx, yy = zip(*[(p[1], p[0]) for p in path], strict=True)
    sampled = width[yy, xx]
    return {
        "segment": i, "length_mm": rounded(_length(path, scale)), "pixels": len(path),
        "width_mm": [rounded(value) for value in np.percentile(sampled, [10, 50, 90])],
        "orientation_degrees": _average_direction(path, scale),
        "polyline": _compact(path, scale), "near_unobserved": bool(np.any(~support[yy, xx])),
    }


@dataclass(frozen=True)
class LeadGraph:
    mask: np.ndarray
    visible: np.ndarray
    center: np.ndarray
    nodes: dict
    paths: tuple
    widths: np.ndarray
    interior: np.ndarray

    @classmethod
    def detect(cls, ctx, absolute, radius, cutoff_area, closing, min_length):
        frame = ctx.one()
        mask, visible = _threshcut(frame, ctx, absolute, radius, cutoff_area, closing)
        medial = centerline(mask)
        graph, paths = graph_paths(medial)
        scale = ctx.scale
        width = 2*distance_transform_edt(mask, sampling=(scale.step_y, scale.step_x))
        interior = binary_erosion(visible, structure=CONNECTION, iterations=1)
        filtered = tuple(p for p in paths if _length(p, scale) >= min_length)
        return cls(mask, visible, medial, graph, filtered, width, interior)


def _graph_options(opts, *extra):
    keys(opts, "absolute_dark", "local_radius_mm", "min_component", "close_mm", "min_run_mm", *extra)
    return (scalar(opts.get("absolute_dark", .10), "Absolute darkness limit", .006, .48),
            scalar(opts.get("local_radius_mm", 5), "Local radius in mm", .5, 26),
            integer(opts.get("min_component", 12), "Minimum dark component pixels", 4, 300),
            scalar(opts.get("close_mm", 0), "Closing distance in mm", 0, 2),
            scalar(opts.get("min_run_mm", 2.2), "Minimum path length", .3, 30))


def lead_network(ctx):
    opts = ctx.order.toolbox
    settings = _graph_options(opts, "include_ends")
    mode = opts.get("include_ends", "show")
    if mode not in ("show", "hide"):
        raise Problem("Endpoints choice must be show or hide.")
    bounds = LeadGraph.detect(ctx, *settings)
    runs = sorted(bounds.paths, key=lambda path: (-len(path), path[0]))
    if len(runs) > 800:
        raise Problem("There are more than 800 usable segments. Select a smaller window.", 422)
    segments = [_row(i+1, path, bounds.widths, bounds.interior, ctx.scale) for i,path in enumerate(runs)]
    ends = _end_pixels(bounds.center)
    junctions = _branch_pixels(bounds.center)
    picture = mask_overlay(ctx.one().rgb, [(bounds.mask, LEAD, .30), (bounds.center, CENTERLINE, .8),
                                           (junctions, (245, 242, 168), .9)], bounds.visible)
    if mode == "show":
        picture[binary_dilation(ends, structure=CCROSS)] = (105, 220, 224)
    charts = [png("lead-network", picture, ctx),
              map_asset("lead-width-mm", bounds.widths, bounds.mask, ctx, low=0, high=10)]
    charts.append(csv_table("segments", ("segment", "length_mm", "width_p10_mm", "width_median_mm", "width_p90_mm",
                                         "orientation_degrees", "touches_unobserved"),
                            [(row["segment"], row["length_mm"], *row["width_mm"], row["orientation_degrees"],
                              row["near_unobserved"]) for row in segments]))
    summary = dict(total_detected_length_mm=rounded(sum(row["length_mm"] for row in segments)),
                   segment_count=len(segments), opaque_fraction=ratio(bounds.mask, bounds.visible),
                   visible_fraction=ratio(bounds.visible, ctx.selection),
                   material_width_mm=percentiles(bounds.widths, bounds.center, [10, 50, 90]))
    details = {**summary, "segments": segments, "open_ends": int(ends.sum()),
               "junction_pixels": int(junctions.sum()),
               "note": "Dark continuous material is a lead-line candidate. Paint and opaque glass can also contribute."}
    return Result("leads", "Lead came network", summary, details, tuple(charts))


def _joint_support(lead, label_map, node, radius, scale):
    cy, cx = node
    ry, rx = max(2, int(radius/scale.step_y)), max(2, int(radius/scale.step_x))
    h, w = lead.center.shape
    x0,x1 = max(0,cx-rx),min(w,cx+rx+1)
    y0,y1 = max(0,cy-ry),min(h,cy+ry+1)
    yy,xx = np.mgrid[y0:y1,x0:x1]
    r = np.hypot((xx-cx)*scale.step_x, (yy-cy)*scale.step_y)
    annulus = (r >= radius * .72) & (r <= radius)
    actual = annulus & lead.visible[y0:y1,x0:x1]
    observed = np.sum(actual) / max(1, int(annulus.sum()))
    fragments, n = components(annulus & lead.center[y0:y1,x0:x1], structure=CONNECTION)
    axes = []
    for i in range(1, n+1):
        points = fragments == i
        x, y = float(np.mean(xx[points]))-cx, float(np.mean(yy[points]))-cy
        direction = np.degrees(np.arctan2(y*scale.step_y, x*scale.step_x)) % 360
        if np.any(label_map[y0:y1,x0:x1][points]) and r[points].max() >= radius*.85:
            axes.append(rounded(direction, 1))
    return sorted(axes), rounded(observed)


def solder_joints(ctx):
    opts = ctx.order.toolbox
    parameters = _graph_options(opts, "arm_radius_mm")
    radius = scalar(opts.get("arm_radius_mm", 5), "Arm check radius (mm)", 2, 22)
    lead = LeadGraph.detect(ctx, *parameters)
    pixels = _branch_pixels(lead.center)
    ids, count = components(pixels, structure=CONNECTION)
    if count > 500:
        raise Problem("Too many joint candidates. Use a tighter window or darker limit.", 422)
    path_fg, _ = components(lead.center, structure=CONNECTION)
    joints = []
    lines = []
    for i, bounds in enumerate(find_objects(ids), 1):
        if bounds is None:
            continue
        yy, xx = np.where(ids[bounds] == i)
        cy = int(round(yy.mean()))+bounds[0].start
        cx = int(round(xx.mean()))+bounds[1].start
        axes, observed = _joint_support(lead, path_fg, (cy,cx), radius, ctx.scale)
        if len(axes) < 3 and observed > .90:
            continue
        distances = [min((a-b)%360, (b-a)%360) for a,b in combinations(axes, 2)]
        status = "review" if observed < .91 else ("crowded" if min(distances or [360]) < 29 else "separate")
        center = _point_index((cy,cx), ctx.scale)
        joints.append(dict(number=len(joints)+1, panel_point=center, arms=axes, observed_ring=observed,
                           smallest_arm_angle=rounded(min(distances)) if len(distances) else None, status=status))
        hue = (100, 210, 186) if status == "separate" else (235, 115, 85)
        lines.append(([center], hue, str(len(joints))))
    photo = mask_overlay(ctx.one().rgb, [(lead.center, LEAD, .65)], lead.visible)
    asset = png("joint-overview", mark_image(photo, lines), ctx)
    summary = dict(candidates=len(joints), sufficiently_observed=sum(row["status"] != "review" for row in joints),
                   radius_mm=radius)
    return Result("joints", "Solder joint review", summary, {**summary, "joints": joints,
                  "note": "An arm must be visible near the outer check ring. An occluded ring calls for manual review."},
                  (asset,))
