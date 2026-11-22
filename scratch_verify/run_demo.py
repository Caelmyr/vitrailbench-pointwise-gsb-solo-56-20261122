#!/usr/bin/env python3
"""Drive the template tool against the shipped synthetic demo specimens."""
import sys
import tempfile

sys.path.insert(0, "/workspace")

import numpy as np
from scipy.ndimage import binary_dilation

from vitrail.bench import Bench
from vitrail.contracts import Box, WorkOrder
from vitrail.demo import load_demo
from vitrail.library import Library
from vitrail.leads import CCROSS, _graph_options
from vitrail.panes import PaneMap
from vitrail.templates import _trace_ring, paper_templates
from vitrail.contracts import Problem

lib = Library(tempfile.mkdtemp(prefix="vitrail-demo-"))


def run_specimen(key):
    panel, _ = load_demo(lib, key)
    photo = lib.panel_summary(panel["id"])["photos"][0]
    ctx = Bench.load(lib, panel["id"],
                     WorkOrder((photo["id"],), Box(0, 0, 1, 1), {}, {}, 700))
    frame = ctx.one()
    opts = _graph_options({}, "close_lead_gap_mm", "min_pane_mm2")
    panes = PaneMap.find(ctx, 0.8, 120, opts)
    blocked_zone = binary_dilation(~frame.measurable, structure=CCROSS, iterations=4)
    print(f"== {key}: {len(panes.entries)} panes, unmeasured px {int((~frame.measurable).sum())}")
    used = 0
    for entry in panes.entries:
        pid = entry["pane"]
        ring = _trace_ring(panes.labels == pid)
        yy = ring[:, 0].astype(int)
        xx = ring[:, 1].astype(int)
        blocked = np.flatnonzero(blocked_zone[yy, xx])
        if len(blocked) < 4:
            continue
        lo, hi = blocked.min(), blocked.max()
        rn = np.stack(((ring[:, 1] + 0.5) / ctx.scale.width_px,
                       (ring[:, 0] + 0.5) / ctx.scale.height_px), axis=1)
        p1 = rn[(lo - 2) % len(rn)]
        p2 = rn[(hi + 2) % len(rn)]
        for variant in ("observed", "completed"):
            order = WorkOrder(
                (photo["id"],), Box(0, 0, 1, 1),
                {"pane": pid, "allowance_mm": 1.0,
                 "manual_edge": [p1.tolist(), p2.tolist()],
                 "print_variant": variant}, {}, 700)
            try:
                result = paper_templates(Bench.load(lib, panel["id"], order))
            except Problem as exc:
                # A pane whose obstruction dominates its ring is rejected by design.
                print(f"   pane {pid}: rejected as expected ({exc})")
                break
            v = result.details["variants"][variant]
            assert v["manual_edge_mm"] > 0
            assert any(o == "manual" for o in v["edge_origin"])
            svg = next(a for a in result.assets if a.key == "paper-template")
            assert svg.mime == "image/svg+xml" and b"viewBox" in svg.payload
        else:
            used += 1
            print(f"   pane {pid}: manual run mm="
                  f"{result.details['variants'][variant]['manual_edge_mm']}, ok")
    print(f"   panes with an obstructed edge: {used}")
    # validation still behaves
    try:
        paper_templates(Bench.load(
            lib, panel["id"],
            WorkOrder((photo["id"],), Box(0, 0, 1, 1),
                      {"pane": 50, "allowance_mm": 1.0,
                       "manual_edge": [[.1, .1], [.2, .2]]}, {}, 700)))
        raise AssertionError("expected catalog error")
    except Problem as exc:
        assert "not in the catalog" in str(exc)


for specimen in ("ivy", "lantern"):
    run_specimen(specimen)
print("demo specimens passed")
