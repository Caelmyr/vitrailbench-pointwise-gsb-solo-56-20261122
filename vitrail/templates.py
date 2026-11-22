from dataclasses import dataclass
from math import hypot
from xml.etree.ElementTree import Element, SubElement, tostring

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt

from .bench import Asset, Result
from .contracts import Problem, integer, keys, option, pair, rounded, scalar
from .leads import CCROSS, _graph_options
from .panes import PaneMap
from .picture import csv_table, mask_overlay, png


OBSERVED_COLOR = (94, 197, 178)
COMPLETED_COLOR = (239, 190, 92)
MANUAL_COLOR = (224, 86, 116)
NEIGHBOR_COLOR = (150, 168, 205)
PANE_TINT = (247, 226, 150)
MANUAL = "manual"
OBSERVED = "observed"


@dataclass(frozen=True)
class Ring:
    px: np.ndarray          # (y, x) pixel positions on the traced edge
    norm: np.ndarray        # normalized panel positions
    mm: np.ndarray          # physical positions

    def nearest(self, point):
        target = np.asarray(point)
        index = int(np.argmin(((self.norm - target) ** 2).sum(axis=1)))
        return index, float(hypot(*(self.norm[index] - target)))


def _trace_ring(mask):
    # Fill camera gaps inside the pane so its outer boundary is one closed ring.
    solid = binary_fill_holes(mask)
    if not solid.any():
        raise Problem("The pane has no observed interior.", 422)
    padded = np.pad(solid, 1, constant_values=False)
    ys, xs = np.nonzero(padded)
    start = (int(ys[0]), int(xs[0]))
    height, width = padded.shape

    def inside(y, x):
        return 0 <= y < height and 0 <= x < width and padded[y, x]

    # Moore neighborhood, clockwise from the west neighbor.
    directions = ((0, -1), (-1, -1), (-1, 0), (-1, 1),
                  (0, 1), (1, 1), (1, 0), (1, -1))
    order = [start]
    current = start
    incoming = 0  # the start was entered from its west (background) side
    guard = padded.sum() * 4 + 16
    for _ in range(guard):
        found = None
        for step in range(1, 9):
            k = (incoming + step) % 8
            dy, dx = directions[k]
            nxt = (current[0] + dy, current[1] + dx)
            if inside(*nxt):
                found = (nxt, k)
                break
        if found is None:
            raise Problem("The pane edge ring is broken. Tighten the region or lead closing.", 422)
        nxt, k = found
        incoming = (k + 4) % 8
        if nxt == start:
            break
        order.append(nxt)
        current = nxt
    else:
        raise Problem("The pane edge ring does not close. Choose an enclosed pane or tighten lead closing.", 422)
    ring = np.asarray(order, dtype=np.float64) - 1.0  # drop the one-pixel pad
    return ring


def _rdp(points, epsilon):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        seg = b - a
        length = hypot(seg[0], seg[1])
        if length < 1e-9:
            dist = np.hypot(*(points[i + 1:j] - a).T)
        else:
            dist = np.abs(seg[0] * (a[1] - points[i + 1:j, 1])
                          - seg[1] * (a[0] - points[i + 1:j, 0])) / length
        k = i + 1 + int(np.argmax(dist))
        if dist.max() > epsilon:
            keep[k] = True
            stack.extend([(i, k), (k, j)])
    return points[keep]


def _polyline_length(points):
    return float(np.hypot(*np.diff(points, axis=0).T).sum()) if len(points) > 1 else 0.0


def _simplify_polygon(points_mm, epsilon):
    points = _rdp(points_mm, epsilon)
    if np.hypot(*(points[0] - points[-1])) < epsilon:
        points = points[:-1]
    return points


def _rasterize(points_norm, width, height):
    image = Image.new("1", (width, height), 0)
    canvas = ImageDraw.Draw(image)
    canvas.polygon([(0.5 + width * x, 0.5 + height * y) for x, y in points_norm], fill=1)
    return np.asarray(image, dtype=bool)


def _dashed(draw, a, b, fill, width, dash=7, gap=5):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    span = hypot(*(b - a))
    if span <= dash:
        draw.line([tuple(a), tuple(b)], fill=fill, width=width)
        return
    steps = max(1, int(span // (dash + gap)))
    for i in range(steps + 1):
        t0 = i * (dash + gap) / span
        t1 = min(1.0, (i * (dash + gap) + dash) / span)
        if t0 >= 1:
            break
        draw.line([tuple(a + (b - a) * t0), tuple(a + (b - a) * t1)], fill=fill, width=width)


def _completion_curve(observed_mm, p1, p2, step_mm):
    """Smooth bridge through the blocked run, anchored exactly on the two visible points."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    chord = p2 - p1
    length = hypot(*chord)
    if length < 1e-9:
        raise Problem("The two visible edge points are too close.", 422)
    axis = chord / length
    normal = np.array([-axis[1], axis[0]])
    rel = observed_mm - p1
    t = rel @ axis / length
    side = rel @ normal
    window = (t > -1.25) & (t < 2.25) & (np.abs(side) < 9.0)
    fit_t, fit_s = t[window], side[window]
    count = max(20, int(round(length / max(step_mm, 0.12))))
    grid = np.linspace(0.0, 1.0, count)
    if len(fit_t) >= 6 and np.ptp(fit_t) > 0.4:
        degree = 3 if len(fit_t) >= 12 else 2
        coeff = np.polyfit(fit_t, fit_s, degree)
        predicted = np.polyval(coeff, grid)
        anchor_0, anchor_1 = np.polyval(coeff, 0.0), np.polyval(coeff, 1.0)
        corrected = predicted - ((1 - grid) * anchor_0 + grid * anchor_1)
        curve = p1 + np.outer(grid * length, axis) + np.outer(corrected, normal)
    else:
        left = observed_mm[np.argsort(np.abs(t + 0.0001))[:5]]
        right = observed_mm[np.argsort(np.abs(t - 1.0))[:5]]
        m1 = np.mean(left, axis=0) - p1
        m2 = p2 - np.mean(right, axis=0)
        m1 = m1 / max(hypot(*m1), 1e-9) * length
        m2 = m2 / max(hypot(*m2), 1e-9) * length
        h00, h10 = 2 * grid**3 - 3 * grid**2 + 1, grid**3 - 2 * grid**2 + grid
        h01, h11 = -2 * grid**3 + 3 * grid**2, grid**3 - grid**2
        curve = np.outer(h00, p1) + np.outer(h10, m1) + np.outer(h01, p2) + np.outer(h11, m2)
    curve[0], curve[-1] = p1, p2
    return curve


def _expand_paper(poly_norm, allowance, width, height, step_x, step_y):
    filled = _rasterize(poly_norm, width, height)
    distance = distance_transform_edt(~filled, sampling=(step_y, step_x))
    return distance <= allowance


def _distance_to_polyline(points, polyline):
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    best = np.full(len(points), np.inf)
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = b - a
        length2 = seg @ seg
        if length2 < 1e-12:
            distance = np.hypot(*(points - a).T)
        else:
            t = np.clip(((points - a) @ seg) / length2, 0.0, 1.0)
            distance = np.hypot(*(points - (a + np.outer(t, seg))).T)
        best = np.minimum(best, distance)
    return best


def _tag_vertices(vertices_mm, observed_src_mm, manual_src_mm, epsilon):
    # Tag each simplified edge whose midpoint follows the manual (obstructed)
    # source polyline within the expansion tolerance. Observed edges elsewhere
    # are far from that short polyline, so this does not need a relative test.
    edge_labels = [OBSERVED] * len(vertices_mm)
    if len(manual_src_mm) < 2:
        return edge_labels
    tolerance = epsilon * 4.0
    n = len(vertices_mm)
    for i in range(n):
        midpoint = (vertices_mm[i] + vertices_mm[(i + 1) % n]) * 0.5
        if _distance_to_polyline([midpoint], manual_src_mm)[0] <= tolerance:
            edge_labels[i] = MANUAL
    if MANUAL not in edge_labels:
        # Fall back to the single edge nearest the manual run.
        midpoints = (vertices_mm + np.roll(vertices_mm, -1, axis=0)) * 0.5
        nearest = int(np.argmin([_distance_to_polyline([mid], manual_src_mm)[0]
                                 for mid in midpoints]))
        edge_labels[nearest] = MANUAL
    vertex_labels = [OBSERVED] * n
    for i, label in enumerate(edge_labels):
        if label == MANUAL:
            vertex_labels[i] = MANUAL
            vertex_labels[(i + 1) % n] = MANUAL
    return vertex_labels


def _manual_segments(vertices_mm, origins):
    segments, run = [], []
    n = len(vertices_mm)
    for i in range(n + 1):
        origin = origins[i % n]
        if origin == MANUAL:
            run.append(vertices_mm[i % n])
        elif run:
            if len(run) >= 2:
                segments.append(np.asarray(run))
            run = []
    if len(run) >= 2 and len(segments) == 0:
        segments.append(np.asarray(run[:n]))
    return segments


def _paper_polygon(points_norm, observed_mm, manual_mm, allowance, ctx, epsilon):
    scale = ctx.scale
    paper = _expand_paper(points_norm, allowance, scale.width_px, scale.height_px,
                          scale.step_x, scale.step_y)
    ring_px = _trace_ring(paper)
    ring_norm = np.stack(((ring_px[:, 1] + 0.5) / scale.width_px,
                          (ring_px[:, 0] + 0.5) / scale.height_px), axis=1)
    ring_mm = ring_norm * np.array([scale.width_mm, scale.height_mm])
    vertices = _simplify_polygon(ring_mm, epsilon)
    origins = _tag_vertices(vertices, observed_mm, manual_mm, epsilon)
    return vertices, origins, paper


def _collision(paper_mask, neighbor_mask, ctx):
    overlap = int(np.count_nonzero(paper_mask & neighbor_mask))
    area = rounded(overlap * ctx.scale.pixel_area)
    if overlap:
        clearance = 0.0
    else:
        distance = distance_transform_edt(~neighbor_mask,
                                          sampling=(ctx.scale.step_y, ctx.scale.step_x))
        clearance = rounded(float(distance[paper_mask].min()))
    return area, clearance


def _variant_metrics(vertices, origins, paper_mask, neighbor_mask, ctx):
    segments = _manual_segments(vertices, origins)
    manual_len = sum(_polyline_length(part) for part in segments)
    perimeter = _polyline_length(np.vstack([vertices, vertices[:1]]))
    extent_x = float(np.ptp(vertices[:, 0]))
    extent_y = float(np.ptp(vertices[:, 1]))
    area_mm2 = rounded(float(np.count_nonzero(paper_mask)) * ctx.scale.pixel_area)
    collision, clearance = _collision(paper_mask, neighbor_mask, ctx)
    return dict(area_mm2=area_mm2, width_mm=rounded(extent_x), height_mm=rounded(extent_y),
                perimeter_mm=rounded(perimeter), observed_edge_mm=rounded(perimeter - manual_len),
                manual_edge_mm=rounded(manual_len), collision_mm2=collision,
                min_clearance_mm=clearance, vertices=len(vertices))


def _read_manual_edge(opts):
    raw = opts.get("manual_edge")
    if not isinstance(raw, list) or len(raw) != 2:
        raise Problem("Mark exactly two visible edge points around the obstructed run.")
    points = [pair(item, "Visible edge point") for item in raw]
    if hypot(*(np.asarray(points[1]) - np.asarray(points[0]))) < 0.001:
        raise Problem("The two visible edge points must be distinct.")
    return points


def _comparison_png(ctx, pane_mask, neighbor_mask, variants, manual_points):
    frame = ctx.one()
    scale = ctx.scale
    base = mask_overlay(frame.rgb,
                        [(pane_mask, PANE_TINT, 0.30), (neighbor_mask, NEIGHBOR_COLOR, 0.28)],
                        frame.present)
    canvas = Image.fromarray(base)
    draw = ImageDraw.Draw(canvas)
    h, w = pane_mask.shape

    def place(points_mm, color, width, dashed=False):
        norm = np.asarray(points_mm) / np.array([scale.width_mm, scale.height_mm])
        px = norm * np.array([w, h])
        closed = np.vstack([px, px[:1]])
        for a, b in zip(closed[:-1], closed[1:]):
            if dashed:
                _dashed(draw, a, b, color, width)
            else:
                draw.line([tuple(a), tuple(b)], fill=color, width=width)

    place(variants["observed"]["vertices"], OBSERVED_COLOR, 2)
    place(variants["completed"]["vertices"], COMPLETED_COLOR, 2)
    for label, vertices_mm, origins in (
        ("A", variants["observed"]["vertices"], variants["observed"]["origins"]),
        ("B", variants["completed"]["vertices"], variants["completed"]["origins"]),
    ):
        for segment in _manual_segments(vertices_mm, origins):
            place(segment, MANUAL_COLOR, 3, dashed=True)
    for i, point in enumerate(manual_points, 1):
        x, y = point[0] * w, point[1] * h
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=MANUAL_COLOR, width=2)
        draw.text((x + 7, y - 6), f"M{i}", fill=MANUAL_COLOR)
    collision = variants["completed"]["paper"] & neighbor_mask
    collision |= variants["observed"]["paper"] & neighbor_mask
    if collision.any():
        tint = np.zeros((h, w, 4), dtype=np.uint8)
        tint[collision] = (224, 86, 116, 170)
        canvas = Image.alpha_composite(canvas.convert("RGBA"), Image.fromarray(tint)).convert("RGB")
        draw = ImageDraw.Draw(canvas)
    legend = (
        "A observed-only (teal)   B trend-completed (amber)   red dashed = MANUAL obstructed run"
        "   blue = neighboring glass")
    text_width = draw.textlength(legend)
    draw.rectangle((6, 6, 6 + text_width + 16, 26), fill=(28, 32, 38))
    draw.text((12, 10), legend, fill=(245, 240, 226))
    return png("template-comparison", np.asarray(canvas), ctx, crop=False)


def _svg_path(parent, points, color, width, dashed=False):
    data = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in points)
    attrs = {"d": data, "fill": "none", "stroke": color, "stroke-width": f"{width:.2f}",
             "stroke-linejoin": "round", "stroke-linecap": "round"}
    if dashed:
        attrs["stroke-dasharray"] = "2.2 1.4"
    SubElement(parent, "path", attrs)


def paper_template_svg(variant_key, vertices_mm, origins, meta):
    margin = 15.0
    points = np.asarray(vertices_mm)
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)
    shifted = points - np.array([min_x, min_y]) + margin
    width = (max_x - min_x) + 2 * margin
    height = (max_y - min_y) + 2 * margin + 22
    svg = Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                          "width": f"{width:g}mm", "height": f"{height:g}mm",
                          "viewBox": f"0 0 {width:g} {height:g}"})
    SubElement(svg, "rect", {"x": "0", "y": "0", "width": f"{width:g}", "height": f"{height:g}",
                             "fill": "white"})
    closed = np.vstack([shifted, shifted[:1]])
    for i, (a, b) in enumerate(zip(closed[:-1], closed[1:])):
        manual = origins[i] == MANUAL or origins[(i + 1) % len(origins)] == MANUAL
        _svg_path(svg, [a, b], "#c03957" if manual else "#1f2a33", 0.35 if manual else 0.22,
                  dashed=manual)
    for i, (x, y) in enumerate(shifted):
        if origins[i] == MANUAL:
            SubElement(svg, "circle", {"cx": f"{x:.3f}", "cy": f"{y:.3f}", "r": "0.9",
                                       "fill": "#c03957"})
    title_y = height - 16
    text = SubElement(svg, "text", {"x": f"{margin}", "y": f"{title_y}", "fill": "#1f2a33",
                                    "font-family": "sans-serif", "font-size": "3.4"})
    variant_name = "observed edges only" if variant_key == "observed" else "trend-completed edges"
    text.text = (f"Pane {meta['pane']} · {variant_name} · allowance {meta['allowance_mm']:g} mm · "
                 f"{meta['width_mm']:g} x {meta['height_mm']:g} mm · {meta['area_mm2']:g} mm²")
    note = SubElement(svg, "text", {"x": f"{margin}", "y": f"{height - 10}", "fill": "#c03957",
                                    "font-family": "sans-serif", "font-size": "3.0"})
    note.text = "Red dashed = obstructed run, MANUAL decision / 红色虚线＝被扣件遮挡段，人工决定 · 1:1, print at 100% / 按 100% 原大打印，勿缩放"
    bar_x, bar_y, bar_len = margin, height - 6.5, min(40.0, width / 3)
    SubElement(svg, "line", {"x1": f"{bar_x}", "y1": f"{bar_y}", "x2": f"{bar_x + bar_len}",
                             "y2": f"{bar_y}", "stroke": "#1f2a33", "stroke-width": "0.3"})
    for step in range(0, int(bar_len) + 1, 5):
        SubElement(svg, "line", {"x1": f"{bar_x + step}", "y1": f"{bar_y - 1.2}",
                                 "x2": f"{bar_x + step}", "y2": f"{bar_y + 1.2}",
                                 "stroke": "#1f2a33", "stroke-width": "0.25"})
    tick = SubElement(svg, "text", {"x": f"{bar_x + bar_len + 1.5}", "y": f"{bar_y + 1.1}",
                                    "fill": "#1f2a33", "font-family": "sans-serif", "font-size": "2.6"})
    tick.text = f"{int(bar_len)} mm"
    return tostring(svg, encoding="utf-8", xml_declaration=True)


def paper_templates(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "pane", "allowance_mm", "manual_edge", "print_variant",
         "absolute_dark", "local_radius_mm", "min_component", "close_mm", "min_run_mm",
         "close_lead_gap_mm", "min_pane_mm2")
    pane_id = integer(opts.get("pane", 1), "Pane number", 1, 240)
    allowance = scalar(opts.get("allowance_mm", 1.5), "Lead-channel allowance (mm)", 0.1, 12)
    print_variant = option(opts.get("print_variant", "observed"), "Printed variant",
                           ("observed", "completed"))
    manual_points = _read_manual_edge(opts)
    lead_opts = {key: opts[key] for key in
                 ("absolute_dark", "local_radius_mm", "min_component", "close_mm",
                  "min_run_mm", "close_lead_gap_mm", "min_pane_mm2") if key in opts}
    graph_options = _graph_options(lead_opts, "close_lead_gap_mm", "min_pane_mm2")
    gap = scalar(lead_opts.get("close_lead_gap_mm", 0.8), "Preview gap closing (mm)", 0, 3)
    minimum = scalar(lead_opts.get("min_pane_mm2", 120), "Minimum pane area (mm²)", 4, 80000)
    panes = PaneMap.find(ctx, gap, minimum, graph_options)
    if pane_id > len(panes.entries):
        raise Problem(f"Pane {pane_id} is not in the catalog. This window has {len(panes.entries)} panes.", 404)
    pane_mask = panes.labels == pane_id
    if not pane_mask.any():
        raise Problem("The selected pane has no observed interior.", 422)
    neighbor_mask = (panes.labels > 0) & (panes.labels != pane_id)
    scale = ctx.scale

    ring_px = _trace_ring(pane_mask)
    ring_norm = np.stack(((ring_px[:, 1] + 0.5) / scale.width_px,
                          (ring_px[:, 0] + 0.5) / scale.height_px), axis=1)
    ring_mm = ring_norm * np.array([scale.width_mm, scale.height_mm])
    ring = Ring(ring_px, ring_norm, ring_mm)
    # The pane ring sits just inside the came. The clip/occlusion is a camera
    # fact (unmeasurable pixels), independent of the catalog's lead closing.
    # Grow the unmeasurable zone outward so an edge ring pixel next to the
    # clip is recognized as belonging to the obstructed run.
    clip_proximity = binary_dilation(~frame.measurable, structure=CCROSS, iterations=4)
    edge_blocked = clip_proximity

    i1, err1 = ring.nearest(manual_points[0])
    i2, err2 = ring.nearest(manual_points[1])
    tolerance_norm = max(3.0, allowance * 2.2) / min(scale.width_mm, scale.height_mm)
    if max(err1, err2) > tolerance_norm:
        raise Problem("Place both manual points directly on the selected pane's visible edge.")
    for point in manual_points:
        y, x = scale.point_px(point)
        if not frame.measurable[y, x]:
            raise Problem("Each manual point must sit on a visible, measured edge.")
    if i1 == i2:
        raise Problem("The two manual points land on the same edge pixel; spread them around the blocked run.")

    n = len(ring.px)
    first, last = sorted((i1, i2))
    arc_a = np.arange(first, last + 1)
    arc_b = np.arange(last, first + n + 1) % n

    def support(indices):
        yy = ring.px[indices, 0].astype(int)
        xx = ring.px[indices, 1].astype(int)
        return 1.0 - float(edge_blocked[yy, xx].mean())

    support_a, support_b = support(arc_a), support(arc_b)
    if support_a <= support_b:
        manual_arc, observed_arc = arc_a, arc_b
    else:
        manual_arc, observed_arc = arc_b, arc_a
    manual_blocked = np.flatnonzero(edge_blocked[ring.px[manual_arc, 0].astype(int),
                                                  ring.px[manual_arc, 1].astype(int)])
    if len(manual_blocked) < 4:
        raise Problem("No obstructed edge was found between the two points. Bracket the clip-covered run.")
    if len(manual_arc) > 0.6 * n:
        raise Problem("The blocked run is longer than the observed edge. Place the points closer together around it.")

    p1_mm = np.asarray(manual_points[0]) * np.array([scale.width_mm, scale.height_mm])
    p2_mm = np.asarray(manual_points[1]) * np.array([scale.width_mm, scale.height_mm])
    observed_norm = ring.norm[observed_arc]
    observed_mm = ring.mm[observed_arc]
    epsilon = max(0.25, allowance * 0.35)

    chord_manual = np.vstack([p1_mm, p2_mm])
    observed_poly = np.vstack([observed_norm, np.asarray(manual_points)])
    completion_mm = _completion_curve(observed_mm, p1_mm, p2_mm, min(scale.step_x, scale.step_y))
    completion_norm = completion_mm / np.array([scale.width_mm, scale.height_mm])
    completed_poly = np.vstack([observed_norm, completion_norm])

    variants = {}
    variants["observed"] = dict(
        poly=observed_poly, observed_src=observed_mm, manual_src=chord_manual,
        closure="straight chord between the two visible points")
    variants["completed"] = dict(
        poly=completed_poly, observed_src=observed_mm, manual_src=completion_mm,
        closure="smooth trend completion anchored at the two visible points")
    for key, item in variants.items():
        vertices, origins, paper = _paper_polygon(
            item["poly"], item["observed_src"], item["manual_src"], allowance, ctx, epsilon)
        item.update(vertices=vertices, origins=origins, paper=paper,
                    metrics=_variant_metrics(vertices, origins, paper, neighbor_mask, ctx))

    chosen = variants[print_variant]
    chosen_metrics = chosen["metrics"]
    comparison = _comparison_png(ctx, pane_mask, neighbor_mask, variants, manual_points)

    summary_rows = [(key, *[item["metrics"][field] for field in
                            ("area_mm2", "width_mm", "height_mm", "perimeter_mm",
                             "observed_edge_mm", "manual_edge_mm", "collision_mm2",
                             "min_clearance_mm", "vertices")])
                    for key, item in variants.items()]
    summary_csv = csv_table(
        "template-summary",
        ("variant", "area_mm2", "width_mm", "height_mm", "perimeter_mm", "observed_edge_mm",
         "manual_edge_mm", "collision_with_neighbor_glass_mm2", "min_clearance_mm", "vertices"),
        summary_rows)
    vertex_rows = []
    for key, item in variants.items():
        for i, ((x, y), origin) in enumerate(zip(item["vertices"], item["origins"]), 1):
            vertex_rows.append((key, i, rounded(x), rounded(y), origin))
    vertex_csv = csv_table("template-vertices",
                           ("variant", "vertex", "x_mm", "y_mm", "edge_origin"), vertex_rows)
    svg_bytes = paper_template_svg(
        print_variant, chosen["vertices"], chosen["origins"],
        {**chosen_metrics, "pane": pane_id, "allowance_mm": allowance})
    svg_asset = Asset("paper-template", "paper-template.svg", "image/svg+xml", svg_bytes)

    details = {
        "pane": pane_id, "allowance_mm": allowance, "printed_variant": print_variant,
        "manual_points": [[rounded(x, 7), rounded(y, 7)] for x, y in manual_points],
        "blocked_run_visible_support": rounded(support(manual_arc)),
        "variants": {
            key: {**item["metrics"], "closure": item["closure"],
                  "polygon_mm": [[rounded(x), rounded(y)] for x, y in item["vertices"]],
                  "edge_origin": item["origins"],
                  "manual_segments_mm": [[[rounded(x), rounded(y)] for x, y in segment]
                                         for segment in _manual_segments(item["vertices"], item["origins"])]}
            for key, item in variants.items()
        },
        "note": ("The obstructed run is not observed: its shape is a manual decision. The observed-only "
                 "variant closes it with a straight chord; the completed variant extends the confirmed local "
                 "trend. Compare collision with neighboring glass before printing the 1:1 sheet."),
    }
    summary = dict(pane=pane_id, allowance_mm=allowance, printed_variant=print_variant,
                   observed_collision_mm2=variants["observed"]["metrics"]["collision_mm2"],
                   completed_collision_mm2=variants["completed"]["metrics"]["collision_mm2"],
                   observed_manual_edge_mm=variants["observed"]["metrics"]["manual_edge_mm"],
                   completed_manual_edge_mm=variants["completed"]["metrics"]["manual_edge_mm"])
    return Result("template", "Replacement paper templates", summary, details,
                  (comparison, svg_asset, summary_csv, vertex_csv))
