from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import sobel

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Problem, keys, pair, rounded, scalar
from .picture import coded, csv_table, map_asset, mark_image, mask_overlay, png


COARSE_INK = (251, 226, 140)
FINE_INK = (167, 215, 198)
UNRESOLVED_INK = (224, 111, 159)
PROBE_INK = (244, 222, 118)
DIM_INK = (43, 45, 50)
CONTRAST_FLOOR = .004


@dataclass(frozen=True)
class Band:
    tag: str
    spacing_mm: float
    orientation: np.ndarray
    coherence: np.ndarray
    contrast: np.ndarray
    valid: np.ndarray
    readable: np.ndarray


def _band_gain(freq, spacing):
    # Raised-cosine passband on wavelength, ±28% around the declared spacing.
    f_lo = 1/(spacing*1.32)
    f_hi = 1/(spacing*.78)
    taper = .18
    rise = (freq - f_lo*(1-taper)) / (2*taper*f_lo)
    fall = (f_hi*(1+taper) - freq) / (2*taper*f_hi)
    gain = np.clip(np.minimum(rise, fall), 0, 1)
    return .5 - .5*np.cos(np.pi*gain)


def _edge_taper(shape, scale, spacing):
    height, width = shape
    margin_y = min(max(2, int(round(2.*spacing/scale.step_y))), max(1, height//4))
    margin_x = min(max(2, int(round(2.*spacing/scale.step_x))), max(1, width//4))
    ramp_y = .5 - .5*np.cos(np.pi*np.arange(margin_y)/margin_y)
    ramp_x = .5 - .5*np.cos(np.pi*np.arange(margin_x)/margin_x)
    taper_y = np.ones(height)
    taper_x = np.ones(width)
    taper_y[:margin_y] = ramp_y
    taper_y[height-margin_y:] = ramp_y[::-1]
    taper_x[:margin_x] = ramp_x
    taper_x[width-margin_x:] = ramp_x[::-1]
    return taper_y[:, None]*taper_x[None, :]


def _spectral_band(signal, available, scale, spacing):
    fill, fill_ok = masked_blur(signal, available,
                                (spacing*.6/scale.step_y, spacing*.6/scale.step_x), floor=.5)
    taper = _edge_taper(signal.shape, scale, spacing)
    center = float(np.mean(fill[available])) if available.any() else 0.
    centered = center + (np.where(available, signal, fill) - center)*taper
    height, width = signal.shape
    spectrum = np.fft.rfft2(centered)
    fy = np.fft.fftfreq(height, d=scale.step_y)
    fx = np.fft.rfftfreq(width, d=scale.step_x)
    freq = np.hypot(fy[:, None], fx[None, :])
    band = np.fft.irfft2(spectrum*_band_gain(freq, spacing), s=(height, width))
    return band, available & fill_ok & (taper >= .98)


def _band(ctx, signal, available, spacing, neighborhood, tag, min_coherence):
    scale = ctx.scale
    band, known = _spectral_band(signal, available, scale, spacing)
    dy = sobel(band, axis=0, mode="nearest") / (8*scale.step_y)
    dx = sobel(band, axis=1, mode="nearest") / (8*scale.step_x)
    local = (neighborhood/scale.step_y, neighborhood/scale.step_x)
    a, ok_a = masked_blur(dx*dx, known, local)
    b, ok_b = masked_blur(dx*dy, known, local)
    c, ok_c = masked_blur(dy*dy, known, local)
    trace = a + c
    spread = np.hypot(a-c, 2*b)
    coherence = spread / np.maximum(trace, 1e-9)
    orientation = (np.degrees(.5*np.arctan2(2*b, a-c)) + 90) % 180
    energy, ok_e = masked_blur(band*band, known, local, floor=.52)
    contrast = np.sqrt(energy)
    valid = known & ok_a & ok_b & ok_c & ok_e & (trace > 1e-9)
    readable = valid & (coherence >= min_coherence) & (contrast > CONTRAST_FLOOR)
    return Band(tag, spacing, orientation, coherence, contrast, valid, readable)


def _gap(first, second):
    return np.abs((np.asarray(first) - np.asarray(second) + 90) % 180 - 90)


def _tick(ux, uy, degrees, neighborhood, scale, fraction):
    dx = np.cos(np.radians(degrees))*neighborhood*fraction/scale.width_mm
    dy = np.sin(np.radians(degrees))*neighborhood*fraction/scale.height_mm
    return [(ux-dx, uy-dy), (ux+dx, uy+dy)]


def _classify(coarse_ok, fine_ok, separation, agreement):
    if coarse_ok and fine_ok:
        return "both-agree" if separation <= agreement else "both-kept"
    if coarse_ok:
        return "coarse"
    if fine_ok:
        return "fine"
    return "unresolved"


def _panel_patch(ctx, frame, point, radius, coarse, fine):
    scale = ctx.scale
    rx, ry = radius/scale.width_mm, radius/scale.height_mm
    x0 = max(0, int(np.floor((point[0]-rx)*scale.width_px)))
    x1 = min(scale.width_px, int(np.ceil((point[0]+rx)*scale.width_px)))
    y0 = max(0, int(np.floor((point[1]-ry)*scale.height_px)))
    y1 = min(scale.height_px, int(np.ceil((point[1]+ry)*scale.height_px)))
    if x1-x0 < 2 or y1-y0 < 2:
        raise Problem("The review point is outside the measured panel area.")
    view = coded(frame.rgb[y0:y1, x0:x1], frame.present[y0:y1, x0:x1])
    image = Image.fromarray(view)
    factor = max(1, min(9, round(320/max(1, max(image.size)))))
    if factor > 1:
        image = image.resize((image.width*factor, image.height*factor), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    cx = (point[0]*scale.width_px - x0) * factor
    cy = (point[1]*scale.height_px - y0) * factor
    ex, ey = rx*scale.width_px*factor, ry*scale.height_px*factor
    draw.ellipse((cx-ex, cy-ey, cx+ex, cy+ey), outline=PROBE_INK, width=2)
    px, py = scale.point_px(point)
    for band, ink, width in ((coarse, COARSE_INK, 3), (fine, FINE_INK, 2)):
        if band.readable[py, px]:
            angle = np.radians(band.orientation[py, px])
            dx, dy = np.cos(angle)*ex*.8, np.sin(angle)*ey*.8
            draw.line((cx-dx, cy-dy, cx+dx, cy+dy), fill=ink, width=width)
    draw.line((cx-6, cy, cx+6, cy), fill=PROBE_INK, width=1)
    draw.line((cx, cy-6, cx, cy+6), fill=PROBE_INK, width=1)
    return np.asarray(image)


def _camera_patch(ctx, frame, camera, point, radius):
    scale = ctx.scale
    rx, ry = radius/scale.width_mm, radius/scale.height_mm
    corners = np.array([[point[0]-rx, point[1]-ry], [point[0]+rx, point[1]-ry],
                        [point[0]+rx, point[1]+ry], [point[0]-rx, point[1]+ry]])
    mapped = frame.alignment.to_camera(corners)
    center = frame.alignment.to_camera(np.array([point]))[0]
    width, height = camera.size
    pixels = mapped * np.array([width, height], dtype=float)
    lo, hi = pixels.min(axis=0), pixels.max(axis=0)
    span = np.maximum(hi-lo, 4)
    lo, hi = lo-span*.16-3, hi+span*.16+3
    info = dict(camera_location=[rounded(v, 7) for v in center],
                inside_camera=bool(0 <= center[0] <= 1 and 0 <= center[1] <= 1))
    if hi[0] < 0 or hi[1] < 0 or lo[0] > width or lo[1] > height:
        raise Problem("The review point falls outside the camera view.")
    x0 = int(np.clip(np.floor(lo[0]), 0, width-1))
    y0 = int(np.clip(np.floor(lo[1]), 0, height-1))
    x1 = int(np.clip(np.ceil(hi[0]), x0+1, width))
    y1 = int(np.clip(np.ceil(hi[1]), y0+1, height))
    info["camera_box_px"] = [x0, y0, x1, y1]
    crop = camera.image.crop((x0, y0, x1, y1))
    if crop.mode.startswith("I;16"):
        gray = np.asarray(crop, dtype=np.float32) / 65535
        image = Image.fromarray(np.rint(gray*255).astype(np.uint8)).convert("RGB")
    else:
        image = crop.convert("RGB")
    long = max(image.size)
    if long < 340:
        factor = max(1, min(9, round(340/max(1, long))))
        image = image.resize((image.width*factor, image.height*factor), Image.Resampling.NEAREST)
    elif long > 920:
        image.thumbnail((920, 920), Image.Resampling.LANCZOS)
    sx, sy = image.width/(x1-x0), image.height/(y1-y0)
    draw = ImageDraw.Draw(image)
    quad = [((p[0]-x0)*sx, (p[1]-y0)*sy) for p in pixels]
    draw.line(quad+[quad[0]], fill=PROBE_INK, width=2, joint="curve")
    ccx, ccy = (center[0]*width-x0)*sx, (center[1]*height-y0)*sy
    draw.line((ccx-7, ccy, ccx+7, ccy), fill=PROBE_INK, width=1)
    draw.line((ccx, ccy-7, ccx, ccy+7), fill=PROBE_INK, width=1)
    return np.asarray(image), info


def _probe_assets(ctx, frame, point, radius, coarse, fine, agreement, support):
    scale = ctx.scale
    px, py = scale.point_px(point)

    def read(band):
        if not band.valid[py, px]:
            return None
        return dict(degrees=rounded(band.orientation[py, px]), coherence=rounded(band.coherence[py, px]),
                    contrast=rounded(band.contrast[py, px]), readable=bool(band.readable[py, px]))

    coarse_ok, fine_ok = bool(coarse.readable[py, px]), bool(fine.readable[py, px])
    separation = float(_gap(coarse.orientation[py, px], fine.orientation[py, px])) if coarse_ok and fine_ok else None
    info = dict(panel=[rounded(v, 7) for v in point], neighborhood_mm=radius,
                supported=bool(support[py, px]),
                status=_classify(coarse_ok, fine_ok, separation, agreement) if coarse_ok or fine_ok else "unresolved",
                separation_degrees=rounded(separation) if separation is not None else None,
                coarse=read(coarse), fine=read(fine))
    panel_image = _panel_patch(ctx, frame, point, radius, coarse, fine)
    assets = [png("probe-panel-patch", panel_image)]
    if ctx.sources:
        camera_image, camera_info = _camera_patch(ctx, frame, ctx.sources[0], point, radius)
        info.update(camera_info)
        assets.append(png("probe-camera-source", camera_image))
    return info, assets


def streak_orientations(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "coarse_mm", "fine_mm", "neighborhood_mm", "min_coherence", "agreement_deg", "probe")
    coarse_mm = scalar(opts.get("coarse_mm", 3.2), "Coarse streak spacing (mm)", .9, 30)
    fine_mm = scalar(opts.get("fine_mm", .9), "Fine streak spacing (mm)", .3, 12)
    around = scalar(opts.get("neighborhood_mm", 9), "Sampling neighborhood (mm)", 2, 45)
    min_coh = scalar(opts.get("min_coherence", .34), "Minimum direction coherence", .04, .95)
    agree = scalar(opts.get("agreement_deg", 24), "Direction agreement (degrees)", 4, 90)
    probe = opts.get("probe")
    if probe is not None:
        probe = pair(probe, "Review point")
    if coarse_mm < 1.6*fine_mm:
        raise Problem("Keep the coarse spacing at least 1.6× the fine spacing so the two bands stay separate.")
    if around < coarse_mm:
        raise Problem("Use a sampling neighborhood at least as wide as the coarse streak spacing.")
    if fine_mm*.66 < 2*max(ctx.scale.step_x, ctx.scale.step_y):
        raise Problem("The working image cannot resolve the fine spacing. Increase preview resolution.")
    coarse = _band(ctx, frame.signal, frame.measurable, coarse_mm, around, "coarse", min_coh)
    fine = _band(ctx, frame.signal, frame.measurable, fine_mm, around, "fine", min_coh)
    support = ctx.demand((coarse.valid | fine.valid) & ctx.selection)
    separation = _gap(coarse.orientation, fine.orientation)
    both = support & coarse.readable & fine.readable
    kept = both & (separation > agree)
    energetic = (coarse.contrast > CONTRAST_FLOOR) | (fine.contrast > CONTRAST_FLOOR)
    unresolved = support & ~coarse.readable & ~fine.readable & energetic
    quiet = support & ~coarse.readable & ~fine.readable & ~energetic
    h, w = support.shape
    step_x = max(8, round(around/ctx.scale.step_x*1.65))
    step_y = max(8, round(around/ctx.scale.step_y*1.65))
    lines, crosses, samples = [], [], []
    counts = {"coarse": 0, "fine": 0, "both-agree": 0, "both-kept": 0, "unresolved": 0, "quiet": 0}
    for y in range(step_y//2, h, step_y):
        for x in range(step_x//2, w, step_x):
            if not support[y, x]:
                continue
            coarse_ok, fine_ok = bool(coarse.readable[y, x]), bool(fine.readable[y, x])
            gap = float(separation[y, x]) if coarse_ok and fine_ok else None
            if coarse_ok or fine_ok:
                status = _classify(coarse_ok, fine_ok, gap, agree)
            else:
                status = "unresolved" if energetic[y, x] else "quiet"
            counts[status] += 1
            ux, uy = (x+.5)/w, (y+.5)/h
            if coarse_ok:
                lines.append((_tick(ux, uy, coarse.orientation[y, x], around, ctx.scale, .42), COARSE_INK, ""))
            if fine_ok:
                lines.append((_tick(ux, uy, fine.orientation[y, x], around, ctx.scale, .32), FINE_INK, ""))
            if status == "unresolved":
                crosses.append(([(ux, uy)], UNRESOLVED_INK, ""))
            samples.append(dict(panel=[rounded(ux, 7), rounded(uy, 7)], status=status,
                                coarse_degrees=rounded(coarse.orientation[y, x]) if coarse_ok else None,
                                coarse_coherence=rounded(coarse.coherence[y, x]) if coarse.valid[y, x] else None,
                                fine_degrees=rounded(fine.orientation[y, x]) if fine_ok else None,
                                fine_coherence=rounded(fine.coherence[y, x]) if fine.valid[y, x] else None,
                                separation_degrees=rounded(gap) if gap is not None else None))
    dim = quiet | unresolved
    painted = mask_overlay(frame.rgb, [(dim, DIM_INK, .5)], frame.present)
    overlay = png("streak-directions", mark_image(painted, lines+crosses), ctx)
    coarse_map = map_asset("coarse-coherence", coarse.coherence, coarse.valid & ctx.selection, ctx,
                           low=0, high=1, palette="thermal")
    fine_map = map_asset("fine-coherence", fine.coherence, fine.valid & ctx.selection, ctx,
                         low=0, high=1, palette="thermal")
    assets = [overlay, coarse_map, fine_map]
    probe_info = None
    if probe is not None:
        probe_info, probe_assets = _probe_assets(ctx, frame, probe, around, coarse, fine, agree, support)
        assets.extend(probe_assets)
    table = csv_table("streak-samples",
                      ("panel_x", "panel_y", "status", "coarse_deg", "coarse_coherence",
                       "fine_deg", "fine_coherence", "separation_deg"),
                      [[*row["panel"], row["status"], row["coarse_degrees"], row["coarse_coherence"],
                        row["fine_degrees"], row["fine_coherence"], row["separation_degrees"]]
                       for row in samples])
    assets.append(table)
    summary = dict(coarse_spacing_mm=coarse_mm, fine_spacing_mm=fine_mm, neighborhood_mm=around,
                   measured_fraction=ratio(support, ctx.selection),
                   coarse_readable_fraction=ratio(coarse.readable & support, support),
                   fine_readable_fraction=ratio(fine.readable & support, support),
                   both_kept_fraction=ratio(kept, support),
                   unresolved_fraction=ratio(unresolved, support),
                   direction_gap_deg=percentiles(separation, both),
                   direction_samples=len(samples))
    details = {**summary, "minimum_coherence": min_coh, "agreement_degrees": agree,
               "status_counts": counts, "local_samples": samples[:400],
               "samples_stored": min(len(samples), 400), "probe": probe_info,
               "note": "Coarse and fine streaks are read as separate direction fields. Where both bands hold "
                       "different directions both are kept; locations that cannot be separated are flagged "
                       "unresolved. No averaged direction is produced. The probe ellipse marks one sampling "
                       "neighborhood around the review point."}
    return Result("streaks", "Two-scale streak orientation", summary, details, tuple(assets))
