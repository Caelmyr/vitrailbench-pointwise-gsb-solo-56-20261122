#!/usr/bin/env python3
"""Verification harness for the replacement paper-template tool.

Builds a synthetic 200x150 mm panel with a curved lead seam and a clip-sized
missing-camera zone (alpha hole), then exercises both paper variants end to end.
Run with isolated mode:  python3 -I run_verify.py
"""
import os
import sys
import tempfile
from io import BytesIO
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

sys.path.insert(0, "/workspace")
from vitrail.bench import Bench
from vitrail.contracts import Box, PanelSpec, PhotoSpec, Problem, WorkOrder
from vitrail.library import Library
from vitrail.leads import CCROSS, _graph_options
from vitrail.panes import PaneMap
from vitrail.templates import (_completion_curve, _manual_segments, _tag_vertices,
                               _trace_ring, paper_templates)
from vitrail.tools import tool_by_key

OUT = "/workspace/scratch_verify/out"
os.makedirs(OUT, exist_ok=True)

W, H = 800, 600
WIDTH_MM, HEIGHT_MM = 200.0, 150.0


def seam_u(v):
    return 0.52 + 0.10 * np.sin(2 * np.pi * 2.0 * v)


def seam_h(u):
    return 0.55 + 0.035 * np.sin(2 * np.pi * 1.5 * u)


def build_capture():
    yy, xx = np.mgrid[:H, :W]
    u, v = (xx + 0.5) / W, (yy + 0.5) / H
    vertical = seam_u(v)
    horizontal = seam_h(u)
    colors = np.array([[78, 128, 140], [140, 107, 90], [96, 142, 112], [150, 132, 84]],
                      dtype=np.uint8)
    column = (u >= vertical).astype(int)
    row = (v >= horizontal).astype(int)
    rgb = colors[2 * row + column].copy()
    lead_band = (np.abs(u - vertical) < 0.008) | (np.abs(v - horizontal) < 0.009)
    outer = (u < 0.03) | (v < 0.035) | (u > 0.97) | (v > 0.965)
    rgb[lead_band | outer] = (6, 6, 6)
    alpha = np.full((H, W), 255, dtype=np.uint8)
    # Clip-sized saturated zone on the curved vertical seam: narrow enough
    # across the came for the catalog's lead closing to bridge it (keeping
    # the panes separate), while several seam pixels stay genuinely unmeasured.
    clip = (np.abs(u - vertical) < 0.0075) & (v > 0.34) & (v < 0.40)
    rgb[clip] = (255, 255, 255)
    rgba = np.dstack([rgb, alpha])
    buf = BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def main():
    data_dir = tempfile.mkdtemp(prefix="vitrail-verify-")
    library = Library(data_dir)
    spec = PanelSpec("Template test panel", "VERIFY-T1", WIDTH_MM, HEIGHT_MM,
                     "Synthetic curved seam with clip obstruction", True)
    panel, created = library.create_panel(spec)
    assert created
    photo_spec = PhotoSpec("Transmitted", "Synthetic", "transmitted", 0, 0, "LED")
    photo = library.add_photo(panel["id"], "verify.png", build_capture(), photo_spec)
    base_order = WorkOrder(capture_ids=(photo["id"],), region=Box(0, 0, 1, 1),
                           toolbox={}, corrections={}, resolution=700)
    ctx = Bench.load(library, panel["id"], base_order)
    frame = ctx.one()

    # --- locate panes and the obstructed arc --------------------------------
    opts = _graph_options({}, "close_lead_gap_mm", "min_pane_mm2")
    panes = PaneMap.find(ctx, 0.8, 120, opts)
    assert len(panes.entries) >= 2, panes.entries
    # Upper-left enclosed pane: its right edge is the clipped curved seam.
    pane_id = next(row["pane"] for row in panes.entries
                   if row["enclosed"]
                   and row["center"][0] < seam_u(row["center"][1])
                   and row["center"][1] < seam_h(row["center"][0]))
    pane_mask = panes.labels == pane_id
    ring_px = _trace_ring(pane_mask)
    scale = ctx.scale
    ring_norm = np.stack(((ring_px[:, 1] + 0.5) / scale.width_px,
                          (ring_px[:, 0] + 0.5) / scale.height_px), axis=1)
    edge_blocked = binary_dilation(~frame.measurable, structure=CCROSS, iterations=4)
    yy = ring_px[:, 0].astype(int)
    xx = ring_px[:, 1].astype(int)
    blocked = np.flatnonzero(edge_blocked[yy, xx])
    assert len(blocked) > 5, "expected an obstructed seam segment"
    lo, hi = blocked.min(), blocked.max()
    # step one ring vertex away from the clip so the manual points stay measured
    p1 = ring_norm[(lo - 2) % len(ring_norm)]
    p2 = ring_norm[(hi + 2) % len(ring_norm)]
    py1, px1 = scale.point_px(p1)
    py2, px2 = scale.point_px(p2)
    assert frame.measurable[py1, px1] and frame.measurable[py2, px2]

    # --- happy path: both variants ------------------------------------------
    run_order = WorkOrder(
        capture_ids=(photo["id"],), region=Box(0, 0, 1, 1),
        toolbox={"pane": pane_id, "allowance_mm": 0.5, "manual_edge": [p1.tolist(), p2.tolist()],
                 "print_variant": "completed"}, corrections={}, resolution=700)
    result = paper_templates(Bench.load(library, panel["id"], run_order))
    assert result.kind == "template"
    variants = result.details["variants"]
    assert set(variants) == {"observed", "completed"}
    for key, item in variants.items():
        m = item
        assert m["area_mm2"] > 1000
        assert m["manual_edge_mm"] > 5, (key, m["manual_edge_mm"])
        assert m["observed_edge_mm"] > m["manual_edge_mm"]
        assert any(origin == "manual" for origin in m["edge_origin"])
        assert len(m["manual_segments_mm"]) >= 1
        # With a sub-mm allowance the paper stays inside the came gap.
        assert m["collision_mm2"] == 0, (key, m["collision_mm2"])
        assert m["min_clearance_mm"] is not None and m["min_clearance_mm"] >= 0
    # curved completion must bulge away from the straight chord
    completed_curve = np.asarray(variants["completed"]["manual_segments_mm"][0])
    a = completed_curve[0]
    b = completed_curve[-1]
    seg = b - a
    lateral = np.abs(seg[0] * (completed_curve[:, 1] - a[1])
                     - seg[1] * (completed_curve[:, 0] - a[0])) / np.linalg.norm(seg)
    assert lateral.max() > 0.2, f"completion too straight: {lateral.max()}"
    # chord variant manual segment must be straight: every point is collinear
    observed_manual = np.asarray(variants["observed"]["manual_segments_mm"][0])
    oa, ob = observed_manual[0], observed_manual[-1]
    oseg = ob - oa
    deviation = np.abs(oseg[0] * (observed_manual[:, 1] - oa[1])
                       - oseg[1] * (observed_manual[:, 0] - oa[0])) / np.linalg.norm(oseg)
    assert deviation.max() < 0.6, f"chord variant is not straight: {deviation.max()}"
    assert lateral.max() > 2.5 * deviation.max(), "completed curve must depart from the straight chord"
    chord_span_mm = np.linalg.norm(oseg)
    expected_span = np.hypot((p2[0] - p1[0]) * WIDTH_MM, (p2[1] - p1[1]) * HEIGHT_MM)
    assert abs(chord_span_mm - expected_span) < 6

    # --- collision path: large allowance reaches neighboring glass ----------
    collide_order = WorkOrder(
        capture_ids=(photo["id"],), region=Box(0, 0, 1, 1),
        toolbox={"pane": pane_id, "allowance_mm": 5.0, "manual_edge": [p1.tolist(), p2.tolist()],
                 "print_variant": "observed"}, corrections={}, resolution=700)
    collided = paper_templates(Bench.load(library, panel["id"], collide_order))
    hits = {k: v["collision_mm2"] for k, v in collided.details["variants"].items()}
    assert any(value > 0 for value in hits.values()), hits

    # --- validation errors ---------------------------------------------------
    def expect_problem(toolbox, fragment):
        bad = WorkOrder(capture_ids=(photo["id"],), region=Box(0, 0, 1, 1),
                        toolbox=toolbox, corrections={}, resolution=700)
        try:
            paper_templates(Bench.load(library, panel["id"], bad))
        except Problem as exc:
            assert fragment in str(exc), (str(exc), fragment)
            return
        raise AssertionError(f"expected Problem containing {fragment!r}")

    expect_problem({"pane": pane_id, "allowance_mm": 1.5,
                    "manual_edge": [p1.tolist()]}, "exactly two")
    expect_problem({"pane": pane_id, "allowance_mm": 1.5,
                    "manual_edge": [[0.05, 0.05], [0.08, 0.08]]}, "visible edge")
    expect_problem({"pane": 99, "allowance_mm": 1.5,
                    "manual_edge": [p1.tolist(), p2.tolist()]}, "not in the catalog")
    expect_problem({"pane": pane_id, "allowance_mm": 1.5,
                    "manual_edge": [p1.tolist(), p2.tolist()],
                    "print_variant": "guessed"}, "one of")
    expect_problem({"pane": pane_id, "allowance_mm": 1.5,
                    "manual_edge": [p1.tolist(), p1.tolist()]}, "distinct")

    # --- assets ---------------------------------------------------------------
    assets = {asset.key: asset for asset in result.assets}
    assert set(assets) == {"template-comparison", "paper-template",
                           "template-summary", "template-vertices"}
    Image.open(BytesIO(assets["template-comparison"].payload)).verify()
    svg = assets["paper-template"].payload.decode()
    root = ET.fromstring(svg)
    assert "#c03957" in svg and "1:1" in svg and "manual" in svg.lower()
    widths = float(root.attrib["width"].rstrip("mm"))
    assert widths > variants["completed"]["width_mm"]
    with open(os.path.join(OUT, "paper-template.svg"), "wb") as handle:
        handle.write(assets["paper-template"].payload)
    with open(os.path.join(OUT, "template-comparison.png"), "wb") as handle:
        handle.write(assets["template-comparison"].payload)
    for csv_key in ("template-summary", "template-vertices"):
        with open(os.path.join(OUT, csv_key + ".csv"), "wb") as handle:
            handle.write(assets[csv_key].payload)

    # --- unit-level helpers ---------------------------------------------------
    rect = np.array([[0.0, 0.0], [80.0, 0.0], [80.0, 60.0], [0.0, 60.0]])
    right_edge = np.array([[80.0, 0.0], [80.0, 60.0]])
    origins = _tag_vertices(rect, rect, right_edge, 0.4)
    assert origins.count("manual") == 2, origins
    assert len(_manual_segments(rect, origins)) == 1
    curve = _completion_curve(rect, np.array([80.0, 0.0]), np.array([80.0, 60.0]), 0.2)
    assert np.allclose(curve[0], [80.0, 0.0]) and np.allclose(curve[-1], [80.0, 60.0])

    # --- HTTP registration ----------------------------------------------------
    from vitrail.http import create_app
    app = create_app(data_dir)
    client = app.test_client()
    listed = client.get("/api/tools").get_json()["tools"]
    assert any(tool["key"] == "template" for tool in listed)

    print("SUMMARY")
    for key, item in variants.items():
        print(key, {k: item[k] for k in
                    ("area_mm2", "manual_edge_mm", "collision_mm2", "min_clearance_mm", "vertices")})
    print("collision run hits:", hits)
    print("all checks passed; assets written to", OUT)


if __name__ == "__main__":
    main()
