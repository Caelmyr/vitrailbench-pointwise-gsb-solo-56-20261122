from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .bench import Result, percentiles
from .camera import Alignment, sample_rgb, srgb_to_linear
from .contracts import Problem, keys, label, object_data, pair, rounded, scalar
from .picture import colorize, csv_table, mark_image, mask_overlay, png


MESH_X = 41
MESH_Y = 31
EDGE = 0.06
PIN_WEIGHT = 60.0
DATA_WEIGHT = 30.0
CENTER_WEIGHT = 90.0

STATUS_COLOR = {
    "within": (127, 216, 173),
    "frame-ok": (96, 190, 158),
    "disagreement": (244, 129, 110),
    "frame-conflict": (224, 84, 72),
    "unseen": (241, 200, 99),
}


@dataclass(frozen=True)
class LocalLine:
    title: str
    a: tuple
    b: tuple
    measured_mm: float
    on_frame: bool

    @classmethod
    def parse(cls, value, i):
        row = object_data(value, "Measured line")
        keys(row, "name", "from", "to", "measured_mm")
        name = label(row.get("name", f"Measured line {i}"), "Line label", 65)
        a = pair(row.get("from"), "First mark")
        b = pair(row.get("to"), "Second mark")
        if np.linalg.norm(np.asarray(a) - b) < .01:
            raise Problem("A measured line must span at least one percent of the panel image.")
        raw_length = row.get("measured_mm")
        if raw_length is None:
            length = None
        else:
            length = scalar(raw_length, "Measured segment (mm)", .5, 4000)
        return cls(name, a, b, length, _on_frame(a, b))

    def payload(self):
        return dict(name=self.title, **{"from": list(self.a), "to": list(self.b)},
                    measured_mm=self.measured_mm)


def _on_frame(a, b):
    pts = np.asarray((a, b))
    return (np.all(pts[:, 0] <= EDGE) or np.all(pts[:, 0] >= 1 - EDGE)
            or np.all(pts[:, 1] <= EDGE) or np.all(pts[:, 1] >= 1 - EDGE))


def parse_lines(raw, tolerance):
    if raw is None:
        raw = [
            {"name": "Middle horizontal", "from": [.34, .47], "to": [.66, .47],
             "measured_mm": None},
            {"name": "Middle vertical", "from": [.50, .35], "to": [.50, .64],
             "measured_mm": None},
        ]
    if not isinstance(raw, list) or not 1 <= len(raw) <= 40:
        raise Problem("Provide one to forty measured lines.")
    return [LocalLine.parse(entry, i) for i, entry in enumerate(raw, 1)]


def _base_length(line, width_mm, height_mm):
    return float(np.linalg.norm((np.asarray(line.b) - line.a) * [width_mm, height_mm]))


def _status(line, delta, tolerance, present):
    if line.on_frame:
        return "frame-conflict" if abs(delta) > tolerance else "frame-ok"
    if not present:
        return "unseen"
    return "within" if abs(delta) <= tolerance else "disagreement"


def _nearest_node(xs, ys, point):
    ix = int(np.clip(round(point[0] * (len(xs) - 1)), 0, len(xs) - 1))
    iy = int(np.clip(round(point[1] * (len(ys) - 1)), 0, len(ys) - 1))
    return iy, ix


def _line_node_chain(xs, ys, a, b):
    """Distinct nearest mesh nodes walking from a to b, with covered panel fractions."""
    start = _nearest_node(xs, ys, a)
    stop = _nearest_node(xs, ys, b)
    iy0, ix0 = start
    iy1, ix1 = stop
    steps = max(abs(ix1 - ix0), abs(iy1 - iy0))
    if steps == 0:
        raise Problem("A measured line is too short relative to the correction grid. Draw it longer.")
    nodes, fractions = [], []
    previous = None
    for k in range(steps + 1):
        ix = round(ix0 + (ix1 - ix0) * k / steps)
        iy = round(iy0 + (iy1 - iy0) * k / steps)
        node = (int(np.clip(iy, 0, len(ys) - 1)), int(np.clip(ix, 0, len(xs) - 1)))
        if node != previous:
            nodes.append(node)
            fractions.append(k / steps)
            previous = node
    if nodes[-1] != stop:
        nodes.append(stop)
        fractions.append(1.0)
    fractions[0] = 0.0
    return nodes, fractions


def fit_displacement(lines, width_mm, height_mm, smoothing):
    """Boundary-pinned weighted least-squares displacement field, solved in millimetres.

    Each interior measured line asks the warp to make its corrected physical span
    equal the ruler reading. Frame-edge lines only anchor the border and never drive
    the field, so the four-corner registration of the outer frame stays fixed.
    Unknowns are the x/y displacements in mm at every mesh node.
    """
    xs = np.linspace(0, 1, MESH_X)
    ys = np.linspace(0, 1, MESH_Y)
    ny, nx = MESH_Y, MESH_X
    drivers = [line for line in lines if not line.on_frame]
    if not drivers:
        raise Problem("Add at least one measured line inside the frame; frame edges are only anchors.")
    # Higher smoothing value -> stronger Laplacian penalty relative to the data.
    lam = 10.0 ** (2.2 * scalar(smoothing, "Smoothing", 0, 1) - 0.4)
    rows, cols, vals, rhs = [], [], [], []
    row_count = 0

    def node_var(node, component):
        iy, ix = node
        return 2 * (iy * nx + ix) + component

    def add(row_index, entries, value):
        for column, coefficient in entries:
            rows.append(row_index); cols.append(column); vals.append(coefficient)
        rhs.append(value)

    # Block 1: boundary nodes stay exactly on the four-corner registration.
    for iy in range(ny):
        for ix in range(nx):
            if ix in (0, nx - 1) or iy in (0, ny - 1):
                add(row_count, ((node_var((iy, ix), 0), PIN_WEIGHT),), 0.0)
                row_count += 1
                add(row_count, ((node_var((iy, ix), 1), PIN_WEIGHT),), 0.0)
                row_count += 1
    # Block 2: Laplacian smoothness of the mm displacement field (zero target).
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            for component in (0, 1):
                center = node_var((iy, ix), component)
                neighbours = (node_var((iy - 1, ix), component), node_var((iy + 1, ix), component),
                              node_var((iy, ix - 1), component), node_var((iy, ix + 1), component))
                entries = ((center, 4 * lam), *((nb, -lam) for nb in neighbours))
                add(row_count, entries, 0.0)
                row_count += 1
    # Block 3: each measured line contracts or expands symmetrically about its own
    # midpoint (the midpoint stays put), matching a pane squeezed inside a fixed frame.
    # Every mesh node on the line gets an absolute displacement target along the line;
    # the centre node is pinned hard so the correction cannot drift into a translation.
    for line in drivers:
        a, b = np.asarray(line.a), np.asarray(line.b)
        physical = (b - a) * [width_mm, height_mm]
        base_len = float(np.linalg.norm(physical))
        direction = physical / base_len
        shortfall = line.measured_mm - base_len
        nodes, fractions = _line_node_chain(xs, ys, a, b)
        center_node = _nearest_node(xs, ys, (a + b) / 2)
        add(row_count, ((node_var(center_node, 0), CENTER_WEIGHT),
                        (node_var(center_node, 1), CENTER_WEIGHT)), 0.0)
        row_count += 1
        for node, fraction in zip(nodes, fractions):
            target_mm = shortfall * (fraction - 0.5)
            entries = tuple((node_var(node, component), DATA_WEIGHT * direction[component])
                            for component in (0, 1))
            value = DATA_WEIGHT * target_mm * float(np.dot(direction, direction))
            add(row_count, entries, value)
            row_count += 1

    design = coo_matrix((vals, (rows, cols)), shape=(row_count, 2 * nx * ny)).tocsr()
    answer = lsqr(design, np.asarray(rhs), atol=1e-12, btol=1e-12, iter_lim=8000,
                  conlim=1e10)[0]
    ux = answer[0::2].reshape(ny, nx)
    uy = answer[1::2].reshape(ny, nx)
    # The frame never moves, even numerically.
    ux[[0, -1], :] = 0.0; ux[:, [0, -1]] = 0.0
    uy[[0, -1], :] = 0.0; uy[:, [0, -1]] = 0.0
    interp_x = RegularGridInterpolator((ys, xs), ux, bounds_error=False, fill_value=None)
    interp_y = RegularGridInterpolator((ys, xs), uy, bounds_error=False, fill_value=None)

    def warp(points):
        pts = np.clip(np.asarray(points, dtype=np.float64), 0, 1)
        query = pts[..., ::-1]  # interpolator axes are (y, x)
        shift_mm = np.stack((interp_x(query), interp_y(query)), axis=-1)
        return pts + shift_mm / [width_mm, height_mm]

    inverse_warp = _build_inverse(warp, xs, ys, width_mm, height_mm)
    return warp, inverse_warp, ux, uy, (xs, ys)


def _build_inverse(warp, xs, ys, width_mm, height_mm):
    """Numerical inverse: map a corrected panel point back to its pre-warp position."""
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    warped = warp(grid)
    # Displacement from corrected coordinates back to the original ones, sampled on the
    # regular panel grid so it can be interpolated against corrected positions.
    back_x = (grid[..., 0] - warped[..., 0]) * width_mm
    back_y = (grid[..., 1] - warped[..., 1]) * height_mm
    inv_x = RegularGridInterpolator((ys, xs), back_x, bounds_error=False, fill_value=None)
    inv_y = RegularGridInterpolator((ys, xs), back_y, bounds_error=False, fill_value=None)

    def inverse_warp(points):
        pts = np.clip(np.asarray(points, dtype=np.float64), 0, 1)
        query = pts[..., ::-1]
        shift_mm = np.stack((inv_x(query), inv_y(query)), axis=-1)
        return np.clip(pts + shift_mm / [width_mm, height_mm], 0, 1)

    return inverse_warp


def build_warp(local):
    panel = local["panel_mm"]
    lines = [LocalLine.parse(seg, i) for i, seg in enumerate(local["segments"], 1)]
    warp, inverse_warp, _, _, _ = fit_displacement(lines, panel[0], panel[1],
                                                    local.get("smoothing", .35))
    return warp, inverse_warp


def line_reports(lines, warp, width_mm, height_mm, tolerance, frame=None, scale=None):
    rows = []
    for line in lines:
        base_len = _base_length(line, width_mm, height_mm)
        corrected_a, corrected_b = warp((line.a, line.b))
        corrected_len = float(np.linalg.norm((corrected_b - corrected_a) * [width_mm, height_mm]))
        present = True
        if frame is not None and scale is not None:
            ax, ay = scale.point_px(line.a)
            bx, by = scale.point_px(line.b)
            present = bool(frame.present[ay, ax] and frame.present[by, bx])
        before = base_len - line.measured_mm
        after = corrected_len - line.measured_mm
        baseline_status = _status(line, before, tolerance, present)
        # Final status reflects whether the *corrected* registration matches the ruler.
        status = _status(line, after, tolerance, present)
        rows.append(dict(name=line.title, points=[list(line.a), list(line.b)],
                         measured_mm=line.measured_mm, baseline_mm=rounded(base_len),
                         corrected_mm=rounded(corrected_len),
                         baseline_difference_mm=rounded(before),
                         corrected_difference_mm=rounded(after),
                         baseline_status=baseline_status,
                         on_frame=bool(line.on_frame), endpoints_present=bool(present),
                         status=status,
                         used_in_fit=bool(not line.on_frame)))
    return rows


def _resample(frame, alignment, grid):
    mapped = alignment.to_camera(grid)
    code, present = sample_rgb(frame.camera, mapped)
    black = srgb_to_linear(frame.record["data"]["black_code"])
    rgb = np.maximum(srgb_to_linear(code) - black, 0) / 2 ** frame.record["data"]["exposure_ev"]
    return rgb.astype(np.float32), present


def _displacement_mesh(warp, width_px, height_px, width_mm, height_mm):
    xs = (np.arange(width_px) + .5) / width_px
    ys = (np.arange(height_px) + .5) / height_px
    grid = np.stack(np.meshgrid(xs, ys), axis=-1)
    shifted = warp(grid)
    dx = shifted[..., 0] - grid[..., 0]
    dy = shifted[..., 1] - grid[..., 1]
    magnitude = np.sqrt((dx * width_mm) ** 2 + (dy * height_mm) ** 2)
    return dx, dy, magnitude.astype(np.float32)


def local_payload(lines, width_mm, height_mm, tolerance, smoothing):
    return {"segments": [line.payload() for line in lines], "tolerance_mm": tolerance,
            "smoothing": smoothing, "panel_mm": [width_mm, height_mm],
            "note": "Frame edges stay on the four-corner registration; only interior measured lines move."}


def fit_local(ctx, lines=None, tolerance=None, smoothing=None):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "lines", "tolerance_mm", "smoothing")
    tolerance = scalar(tolerance if tolerance is not None else opts.get("tolerance_mm", 1.5),
                       "Allowed measurement difference (mm)", .05, 80)
    smoothing = smoothing if smoothing is not None else opts.get("smoothing", .35)
    width_mm, height_mm = ctx.scale.width_mm, ctx.scale.height_mm
    if lines is None:
        lines = parse_lines(opts.get("lines"), tolerance)
    missing = [line.title for line in lines if line.measured_mm is None]
    if missing:
        raise Problem(f"Fill the measured millimetres for {', '.join(missing)} or remove the line.")
    warp, inverse_warp, dx, dy, _ = fit_displacement(lines, width_mm, height_mm, smoothing)
    reports = line_reports(lines, warp, width_mm, height_mm, tolerance, frame, ctx.scale)
    base = frame.alignment
    corrected = Alignment(base.title + " · local measured lines", base.corners, "local",
                          base.matrix, local_payload(lines, width_mm, height_mm, tolerance, smoothing),
                          warp, inverse_warp)
    grid = ctx.scale.grid()
    rgb, present = _resample(frame, corrected, grid)
    shapes = [([row["points"][0], row["points"][1]], STATUS_COLOR[row["status"]],
               f"{row['name']} {row['status']}") for row in reports]
    registered = png("local-registered-view",
                     mark_image(mask_overlay(rgb, [], present), shapes), ctx, crop=False)
    _, _, magnitude = _displacement_mesh(warp, ctx.scale.width_px, ctx.scale.height_px, width_mm, height_mm)
    high = max(1.0, float(np.nanpercentile(magnitude, 99)) * 1.15)
    displacement = png("local-displacement-mm",
                       mark_image(colorize(magnitude, 0, high, "thermal", frame.present), shapes), ctx, crop=False)
    baseline_rgb = frame.rgb
    combined = baseline_rgb.copy()
    half = ctx.scale.width_px // 2
    combined[:, half:] = rgb[:, half:]
    comparison = png("local-before-after",
                     mark_image(mask_overlay(combined, [], frame.present), shapes), ctx, crop=False)
    table = csv_table("measured-lines",
                      ("name", "measured_mm", "baseline_mm", "corrected_mm",
                       "baseline_difference_mm", "corrected_difference_mm",
                       "baseline_status", "final_status", "used_in_fit"),
                      [(row["name"], row["measured_mm"], row["baseline_mm"], row["corrected_mm"],
                        row["baseline_difference_mm"], row["corrected_difference_mm"],
                        row["baseline_status"], row["status"], row["used_in_fit"]) for row in reports])
    mm_shift = np.sqrt(dx ** 2 + dy ** 2)
    interior = np.ones_like(mm_shift, dtype=bool)
    interior[[0, -1], :] = False; interior[:, [0, -1]] = False
    summary = dict(line_count=len(reports),
                   drivers=sum(row["used_in_fit"] for row in reports),
                   baseline_disagreements=sum(row["baseline_status"] == "disagreement" for row in reports),
                   resolved=sum(row["baseline_status"] == "disagreement" and row["status"] == "within"
                                for row in reports),
                   frame_conflicts=sum(row["status"] == "frame-conflict" for row in reports),
                   residual_disagreements=sum(row["status"] == "disagreement" for row in reports),
                   unseen=sum(row["status"] == "unseen" for row in reports),
                   max_shift_mm=rounded(mm_shift.max()),
                   median_shift_mm=rounded(np.median(mm_shift[interior])),
                   shift_percentiles_mm=percentiles(mm_shift, interior, (50, 90, 95)))
    details = {**summary, "lines": reports, "tolerance_mm": tolerance, "smoothing": smoothing,
               "local": corrected.local,
               "status_legend": {key: key for key in STATUS_COLOR},
               "note": "Frame lines only anchor the border; conflicting frame readings are excluded and flagged."}
    return Result("local-fit", "Local measured-line registration", summary, details,
                  (registered, displacement, comparison, table)), corrected


def local_fit(ctx):
    outcome, _ = fit_local(ctx)
    return outcome
