#!/usr/bin/env python3
"""Full HTTP-path test and 1:1 scale verification for the template tool."""
import sys
import tempfile

sys.path.insert(0, "/workspace")

import numpy as np
from scipy.ndimage import binary_dilation

from vitrail.bench import Bench
from vitrail.contracts import Box, WorkOrder
from vitrail.library import Library
from vitrail.leads import CCROSS, _graph_options
from vitrail.panes import PaneMap
from vitrail.templates import _trace_ring
from run_verify import build_capture, seam_h, seam_u

lib = Library(tempfile.mkdtemp(prefix="vitrail-http-"))
panel, _ = lib.create_panel(__import__("vitrail.contracts", fromlist=["PanelSpec"]).PanelSpec(
    "HTTP test", "VER-HTTP", 200.0, 150.0, "x", True))
photo = lib.add_photo(panel["id"], "v.png", build_capture(),
                      __import__("vitrail.contracts", fromlist=["PhotoSpec"]).PhotoSpec(
                          "t", "s", "transmitted", 0, 0, "L"))
ctx = Bench.load(lib, panel["id"], WorkOrder((photo["id"],), Box(0, 0, 1, 1), {}, {}, 700))
frame = ctx.one()
panes = PaneMap.find(ctx, 0.8, 120, _graph_options({}, "close_lead_gap_mm", "min_pane_mm2"))
pid = next(e["pane"] for e in panes.entries if e["enclosed"]
           and e["center"][0] < seam_u(e["center"][1])
           and e["center"][1] < seam_h(e["center"][0]))
ring = _trace_ring(panes.labels == pid)
rn = np.stack(((ring[:, 1] + .5) / ctx.scale.width_px,
               (ring[:, 0] + .5) / ctx.scale.height_px), axis=1)
blocked = np.flatnonzero(binary_dilation(~frame.measurable, structure=CCROSS, iterations=4)
                         [ring[:, 0].astype(int), ring[:, 1].astype(int)])
p1 = rn[(blocked.min() - 2) % len(rn)].tolist()
p2 = rn[(blocked.max() + 2) % len(rn)].tolist()

from vitrail.http import create_app
client = create_app(lib.root).test_client()
payload = {"captures": [photo["id"]], "region": [0, 0, 1, 1], "resolution": 700,
           "options": {"pane": pid, "allowance_mm": 1.0,
                       "manual_edge": [p1, p2], "print_variant": "completed"}}
resp = client.post(f"/api/panels/{panel['id']}/tools/template", json=payload)
assert resp.status_code == 200, resp.get_data(as_text=True)
data = resp.get_json()
assert data["kind"] == "template"
svg = next(a for a in data["assets"] if a["key"] == "paper-template")
assert svg["mime"] == "image/svg+xml"

# 1:1 scale: SVG mm dimensions must match polygon extents plus the 15 mm margins.
from xml.etree import ElementTree as ET
root = ET.fromstring(svg["data"] and __import__("base64").b64decode(svg["data"]))
svg_w = float(root.attrib["width"].rstrip("mm"))
svg_h = float(root.attrib["height"].rstrip("mm"))
v = data["result"]["variants"]["completed"]
poly = np.asarray(v["polygon_mm"])
expected_w = np.ptp(poly[:, 0]) + 30.0
expected_h = np.ptp(poly[:, 1]) + 30.0 + 22.0
assert abs(svg_w - expected_w) < 0.01, (svg_w, expected_w)
assert abs(svg_h - expected_h) < 0.01, (svg_h, expected_h)
# viewBox matches physical mm units 1:1
vb = [float(x) for x in root.attrib["viewBox"].split()]
assert abs(vb[2] - svg_w) < 1e-9 and abs(vb[3] - svg_h) < 1e-9

# bad request: unknown option is rejected by the strict whitelist
bad = dict(payload, options={**payload["options"], "bogus": 1})
r2 = client.post(f"/api/panels/{panel['id']}/tools/template", json=bad)
assert r2.status_code == 400 and "Unknown field" in r2.get_json()["error"]

# the tool declares a single-capture contract; the order parser rejects dup ids first
r3 = client.post(f"/api/panels/{panel['id']}/tools/template",
                 json={**payload, "captures": [photo["id"], photo["id"]]})
assert r3.status_code == 400 and "occur only once" in r3.get_json()["error"]
from vitrail.tools import tool_by_key
assert tool_by_key("template").input_count == 1

print("HTTP path, 1:1 scale and validation all passed")
print(f"SVG {svg_w:.2f} x {svg_h:.2f} mm vs polygon extents "
      f"{np.ptp(poly[:,0]):.2f} x {np.ptp(poly[:,1]):.2f} mm")
