from itertools import combinations
from math import atan2, hypot
import numpy as np
from PIL import Image, ImageDraw

from .bench import Result, ratio
from .contracts import Problem, integer, keys, label, object_data, pair, rounded
from .picture import csv_table, mark_image, mask_overlay, png
from .survey import _polygon_image


COVER_TINT = (232, 214, 150)
LEAD_SEEN = (104, 212, 207)
HIDDEN_INK = (238, 133, 92)
STAR_INK = (196, 150, 232)
MAX_ARMS = 8
MAX_HYPOTHESES = 10


def _on_segment(p, a, b, tol=1e-9):
    cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
    if abs(cross) > tol * (1 + abs(b[0]-a[0]) + abs(b[1]-a[1])):
        return False
    return (min(a[0], b[0])-tol <= p[0] <= max(a[0], b[0])+tol and
            min(a[1], b[1])-tol <= p[1] <= max(a[1], b[1])+tol)


def _in_poly(point, poly):
    # Points on the polygon boundary are treated as outside: clipped contact vertices sit on
    # an edge, and ray casting alone would classify them inconsistently.
    if any(_on_segment(point, a, b) for a, b in zip(poly, [*poly[1:], poly[0]])):
        return False
    inside = False
    x, y = point
    for a, b in zip(poly, [*poly[1:], poly[0]]):
        if (a[1] > y) != (b[1] > y):
            hit = a[0] + (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1])
            if hit > x:
                inside = not inside
    return inside


def _segment_cross(a, b, c, d):
    def side(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    s1, s2 = side(a, b, c), side(a, b, d)
    s3, s4 = side(c, d, a), side(c, d, b)
    return (s1 == 0 or s2 == 0 or (s1 > 0) != (s2 > 0)) and (s3 == 0 or s4 == 0 or (s3 > 0) != (s4 > 0))


def _seg_intersection(a, b, c, d):
    """Intersection of finite segments a-b and c-d; None if parallel or disjoint."""
    rx, ry = b[0]-a[0], b[1]-a[1]
    sx, sy = d[0]-c[0], d[1]-c[1]
    base = rx*sy - ry*sx
    if base == 0:
        return None
    qpx, qpy = c[0]-a[0], c[1]-a[1]
    u = (qpx*sy - qpy*sx) / base
    t = (qpx*ry - qpy*rx) / base
    if 0 <= u <= 1 and 0 <= t <= 1:
        return (a[0]+u*rx, a[1]+u*ry)
    return None


def _cross_hit(a, b, poly):
    for c, d in zip(poly, [*poly[1:], poly[0]]):
        hit = _seg_intersection(a, b, c, d)
        if hit is not None:
            return hit
    return None


def _clip_fragment(points, poly):
    """Keep only runs outside the occluder; hidden vertices are never returned.

    Also reports whether the fragment meets the label boundary, i.e. whether it is an arm
    of this joint rather than an unrelated lead line elsewhere on the panel.
    """
    runs, current, touches = [], [], False

    def close_run():
        nonlocal current
        if len(current) > 1:
            runs.append(current)
        current = []

    previous_outside = not _in_poly(points[0], poly)
    if previous_outside:
        current = [points[0]]
    for a, b in zip(points, points[1:]):
        a_out, b_out = not _in_poly(a, poly), not _in_poly(b, poly)
        if a_out and b_out:
            current.append(b)
            continue
        if not a_out and not b_out:
            close_run()
            continue
        hit = _cross_hit(a, b, poly)
        if hit is None:  # grazing on the boundary: treat by endpoint classification
            if b_out:
                current.append(b)
            else:
                close_run()
            continue
        touches = True
        if a_out:                # entering the label: finish the observed run at the edge
            current.append(hit)
            close_run()
        else:                    # leaving the label: start a new observed run at the edge
            current = [hit, b]
    close_run()
    return runs, touches


def _parse_cover(raw):
    data = object_data(raw, "Occluder label")
    keys(data, "title", "points")
    vertices = data.get("points")
    if not isinstance(vertices, list) or not 3 <= len(vertices) <= 40:
        raise Problem("The paper label needs a polygon with three to forty vertices.")
    poly = tuple(pair(item, "Label vertex") for item in vertices)
    if len(set(poly)) != len(poly):
        raise Problem("Label vertices must be distinct.")
    area = abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(poly, [*poly[1:], poly[0]]))) / 2
    if area < 0.000003:
        raise Problem("The paper label polygon is too narrow.")
    return label(data.get("title", "Paper label"), "Label title", 80), [list(p) for p in poly]


def _parse_fragments(raw):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 24:
        raise Problem("Draw one to twenty-four visible lead fragments.")
    fragments = []
    for i, entry in enumerate(raw, 1):
        data = object_data(entry, "Lead fragment")
        keys(data, "title", "points")
        vertices = data.get("points")
        if not isinstance(vertices, list) or not 2 <= len(vertices) <= 60:
            raise Problem(f"Lead fragment {i} needs two to sixty vertices.")
        line = [list(pair(item, "Lead point")) for item in vertices]
        if any(hypot(b[0]-a[0], b[1]-a[1]) < 0.0005 for a, b in zip(line, line[1:])):
            raise Problem(f"Lead fragment {i} has vertices too close to distinguish.")
        fragments.append({"title": label(data.get("title", f"Fragment {i}"), "Fragment title", 80), "points": line})
    return fragments


def _polyline_len(points):
    return sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))


def _direction(a, b):
    return atan2(b[1]-a[1], b[0]-a[0])


def _physical_len(points, scale):
    return sum(hypot((b[0]-a[0])*scale.width_mm, (b[1]-a[1])*scale.height_mm)
               for a, b in zip(points, points[1:]))


def _build_arms(ctx, cover, fragments):
    mask = _polygon_image((ctx.scale.width_px, ctx.scale.height_px), cover)
    frame = ctx.one()
    measured = frame.measurable
    stats = dict(observed_inside_fraction=ratio(measured & mask, mask),
                 clipped_inside_fraction=ratio(frame.clipped & mask, mask),
                 area_mm2=rounded(mask.sum() * ctx.scale.pixel_area))
    centroid = (float(np.mean([p[0] for p in cover])), float(np.mean([p[1] for p in cover])))
    on_boundary = lambda p: any(_on_segment(p, a, b) for a, b in zip(cover, [*cover[1:], cover[0]]))
    arms, ignored = [], []
    for fragment in fragments:
        clipped_runs, touches = _clip_fragment(fragment["points"], cover)
        touches = touches or any(on_boundary(p) for p in fragment["points"])
        kept = False
        for run in clipped_runs:
            if not touches or _physical_len(run, ctx.scale) < 0.5:
                continue
            # Clipping guarantees the run ends at the label boundary; the vertex nearest the
            # label centroid is therefore the observed contact point (no hidden pixels added).
            tip = min(run, key=lambda p: hypot(p[0]-centroid[0], p[1]-centroid[1]))
            outer = max(run, key=lambda p: hypot(p[0]-centroid[0], p[1]-centroid[1]))
            angle = (_direction(tip, outer) * 180 / np.pi) % 360
            ys = [min(ctx.scale.height_px-1, max(0, int(p[1]*ctx.scale.height_px))) for p in run]
            xs = [min(ctx.scale.width_px-1, max(0, int(p[0]*ctx.scale.width_px))) for p in run]
            support = float(np.mean(measured[ys, xs]))
            arms.append({"arm": len(arms)+1, "fragment": fragment["title"], "tip": [rounded(tip[0], 7), rounded(tip[1], 7)],
                         "outer_end": [rounded(outer[0], 7), rounded(outer[1], 7)],
                         "contact": [rounded(tip[0], 7), rounded(tip[1], 7)],
                         "approach_degrees": rounded(angle, 1),
                         "visible_length_mm": rounded(_physical_len(run, ctx.scale)),
                         "observed_fraction": rounded(support), "polyline": run})
            kept = True
        if not kept:
            ignored.append(fragment["title"])
    if len(arms) < 2:
        raise Problem("At least two visible lead fragments must reach the label edge.")
    if len(arms) > MAX_ARMS:
        raise Problem(f"Only the first {MAX_ARMS} arms at a label can be adjudicated. Tighten the region.")
    return arms, ignored, stats, mask


def _pairings(n):
    def build(indices):
        if not indices:
            return [[]]
        head, rest = indices[0], indices[1:]
        out = []
        for j, partner in enumerate(rest):
            tail = [x for k, x in enumerate(rest) if k != j]
            out.extend([[(head, partner), *pairs] for pairs in build(tail)])
        return out
    plans = [{"kind": "star", "pairs": []}]
    if n % 2 == 0:
        plans.extend({"kind": "paired", "pairs": pairs} for pairs in build(list(range(n))))
    else:
        for center in range(n):
            plans.append({"kind": "partial-star", "pairs": [], "center": center})
            plans.extend({"kind": "partial-paired", "pairs": pairs, "center": center}
                         for pairs in build([i for i in range(n) if i != center]))
    unique, seen = [], set()
    for plan in plans:
        token = (plan["kind"], plan.get("center"), tuple(sorted(plan["pairs"])))
        if token not in seen:
            seen.add(token)
            unique.append(plan)
    return unique[:MAX_HYPOTHESES]


def _bezier(p, q, ctrl_p, ctrl_q, steps=28):
    points = []
    for k in range(steps+1):
        t = k / steps
        x = (1-t)**3*p[0] + 3*(1-t)**2*t*ctrl_p[0] + 3*(1-t)*t**2*ctrl_q[0] + t**3*q[0]
        y = (1-t)**3*p[1] + 3*(1-t)**2*t*ctrl_p[1] + 3*(1-t)*t**2*ctrl_q[1] + t**3*q[1]
        points.append((x, y))
    return points


def _connector(arms, i, j):
    a, b = arms[i]["contact"], arms[j]["contact"]
    tip_a, tip_b = arms[i]["tip"], arms[j]["tip"]
    span = hypot(b[0]-a[0], b[1]-a[1]) or 0.01
    # Control handles follow each arm's observed approach direction into the label, so the
    # inferred link stays inside the occluder instead of bowing back over photographed glass.
    def inward(tip, contact):
        dx, dy = contact[0]-tip[0], contact[1]-tip[1]
        norm = hypot(dx, dy) or 1e-6
        reach = min(span * 0.7, 0.12)
        return (contact[0] + dx/norm*reach, contact[1] + dy/norm*reach)
    c1 = inward(tip_a, a)
    c2 = inward(tip_b, b)
    return _bezier(a, b, c1, c2)


def _cross_count(curves):
    total = 0
    for c1, c2 in combinations(curves, 2):
        total += sum(_segment_cross(a, b, c, d)
                     for a, b in zip(c1, c1[1:]) for c, d in zip(c2, c2[1:]))
    return total


def _continuity(arms, i, j, curve):
    incoming = _direction(arms[i]["tip"], arms[i]["contact"])
    outgoing = _direction(arms[j]["contact"], arms[j]["tip"])
    start = _direction(curve[0], curve[1])
    end = _direction(curve[-2], curve[-1])
    def turn(x, y):
        return abs((x-y+np.pi) % (2*np.pi) - np.pi)
    return (turn(incoming, start) + turn(end, outgoing)) / 2


def _score_plan(plan, arms, cover, curves):
    paired = plan["pairs"]
    hidden_pts = [p for curve in curves for p in curve]
    inside = np.mean([_in_poly(p, cover) for p in hidden_pts]) if hidden_pts else 0.0
    length = sum(_polyline_len(curve) for curve in curves)
    continuity = np.mean([_continuity(arms, i, j, curve) for (i, j), curve in zip(paired, curves)]) if paired else 0.0
    crosses = _cross_count(curves)
    lengths = [_polyline_len(curve) for curve in curves]
    spread = (max(lengths)-min(lengths)) / max(np.mean(lengths), 1e-9) if lengths else 0.0
    score = 100 - 42*crosses - 34*(1-inside) - 16*spread - 8*(continuity/np.pi)
    if plan["kind"] in ("star", "partial-star"):
        score -= 2.5
    return {"crossing_segments": crosses, "hidden_inside_label_fraction": rounded(inside),
            "inferred_length_norm": rounded(length, 5), "mean_turn_radians": rounded(float(continuity), 4),
            "length_spread": rounded(float(spread), 4), "score": round(float(max(0.0, min(100.0, score))), 2)}


def _dashed(draw, points, fill, width, dash=9, gap=6):
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        distance = hypot(dx, dy)
        if distance == 0:
            continue
        ux, uy = dx/distance, dy/distance
        travelled = 0
        while travelled < distance:
            end = min(travelled+dash, distance)
            draw.line((a[0]+ux*travelled, a[1]+uy*travelled, a[0]+ux*end, a[1]+uy*end), fill=fill, width=width)
            travelled += dash + gap


def _base_picture(ctx, cover, arms, mask):
    frame = ctx.one()
    picture = mask_overlay(frame.rgb, [(mask, COVER_TINT, 0.38)], frame.present)
    seen = [(arm["polyline"], LEAD_SEEN, str(arm["arm"])) for arm in arms]
    picture = mark_image(picture, seen)
    image = Image.fromarray(picture).convert("RGB")
    draw = ImageDraw.Draw(image)
    h, w = picture.shape[:2]
    border = [(p[0]*(w-1), p[1]*(h-1)) for p in [*cover, cover[0]]]
    draw.line(border, fill=COVER_TINT, width=max(2, int(round(max(w, h)/340))))
    return image, draw


def _hypothesis_image(ctx, cover, arms, mask, plan, curves, number, total):
    image, draw = _base_picture(ctx, cover, arms, mask)
    h, w = image.size[1], image.size[0]
    width = max(2, int(round(max(w, h)/300)))
    ink = STAR_INK if plan["kind"] in ("star", "partial-star") else HIDDEN_INK
    centroid = (np.mean([p[0] for p in cover]), np.mean([p[1] for p in cover]))
    if plan["kind"] in ("star", "partial-star"):
        center = arms[plan["center"]]["contact"] if plan["kind"] == "partial-star" else list(centroid)
        for i, arm in enumerate(arms):
            if plan["kind"] == "partial-star" and i == plan["center"]:
                continue
            curve = _bezier(arm["contact"], center, arm["contact"], center, 12)
            _dashed(draw, [(x*(w-1), y*(h-1)) for x, y in curve], ink, width)
    for curve in curves:
        _dashed(draw, [(x*(w-1), y*(h-1)) for x, y in curve], ink, width)
    draw.text((10, 9), f"Hypothesis {number}/{total} · {plan['kind']} · dashed = hidden, not photographed",
              fill=ink, stroke_width=2, stroke_fill=(26, 28, 35))
    return png(f"joint-hypothesis-{number}", np.asarray(image), ctx, crop=False)


def joint_adjudication(ctx):
    opts = ctx.order.toolbox
    keys(opts, "cover", "fragments", "chosen_hypothesis", "analyst_note")
    title, cover = _parse_cover(opts["cover"])
    fragments = _parse_fragments(opts.get("fragments"))
    arms, ignored, cover_stats, mask = _build_arms(ctx, cover, fragments)
    plans = _pairings(len(arms))
    hypotheses = []
    for number, plan in enumerate(plans, 1):
        curves = [_connector(arms, i, j) for i, j in plan["pairs"]]
        metrics = _score_plan(plan, arms, cover, curves)
        hypotheses.append({"number": number, "connection": plan["kind"],
                           "pairs": [[i+1, j+1] for i, j in plan["pairs"]],
                           "star_center_arm": plan["center"]+1 if "center" in plan else None,
                           **metrics})
    hypotheses.sort(key=lambda row: (-row["score"], row["number"]))
    for rank, row in enumerate(hypotheses, 1):
        row["rank"] = rank
        row["standing"] = "leading" if rank == 1 else ("close" if rank == 2 and
                          hypotheses[0]["score"]-row["score"] <= 8 else "alternative")
    chosen = None
    if opts.get("chosen_hypothesis") is not None:
        wanted = integer(opts["chosen_hypothesis"], "Chosen hypothesis", 1, len(plans))
        chosen = next(row for row in hypotheses if row["number"] == wanted)
    assets = []
    overview, _ = _base_picture(ctx, cover, arms, mask)
    assets.append(png("joint-overview", np.asarray(overview), ctx, crop=False))
    curve_lookup = {tuple(plan["pairs"]): [_connector(arms, i, j) for i, j in plan["pairs"]] for plan in plans}
    for row in hypotheses:
        plan = plans[row["number"]-1]
        assets.append(_hypothesis_image(ctx, cover, arms, mask, plan,
                                        curve_lookup[tuple(plan["pairs"])], row["number"], len(plans)))
    assets.append(csv_table("hypotheses",
                    ("rank", "hypothesis", "connection", "pairs", "score", "crossing_segments",
                     "hidden_inside_label_fraction", "inferred_length_norm", "mean_turn_radians", "length_spread"),
                    [(r["rank"], r["number"], r["connection"], r["pairs"], r["score"], r["crossing_segments"],
                      r["hidden_inside_label_fraction"], r["inferred_length_norm"], r["mean_turn_radians"],
                      r["length_spread"]) for r in hypotheses]))
    summary = dict(label=title, visible_arms=len(arms), hypotheses=len(hypotheses),
                   leading_score=hypotheses[0]["score"], label_observed_fraction=cover_stats["observed_inside_fraction"],
                   adjudicated=chosen is not None)
    result = {**summary, "occluder": {"title": title, "polygon": cover, **cover_stats},
              "visible_arms": [{k: v for k, v in arm.items() if k != "polyline"} for arm in arms],
              "ignored_fragments": ignored,
              "hypotheses": hypotheses,
              "chosen_hypothesis": chosen["number"] if chosen else None,
              "analyst_note": label(opts.get("analyst_note", ""), "Analyst note", 560, True),
              "note": "Solid cyan follows photographed lead. Dashed lines cross the paper label and are inferred only; "
                      "they are not observed pixels. Retain the record to store the manual determination."}
    return Result("joint-adjudication", "Labeled joint connection", summary, result, tuple(assets))
