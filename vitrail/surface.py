from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, label, find_objects, sobel

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Problem, keys, rounded, scalar
from .leads import CONNECTION, centerline, clean_regions, graph_paths
from .picture import csv_table, map_asset, mark_image, mask_overlay, png


def _ellipse_bounds(height, width, sx, sy):
    return [max(.5, sy/height), max(.5, sx/width)]


def _texture_details(signal, available, scale, wavelength, neighborhood):
    coarse = (wavelength*.7/scale.step_y, wavelength*.7/scale.step_x)
    fine = (wavelength*.19/scale.step_y, wavelength*.19/scale.step_x)
    wide, enough = masked_blur(signal, available, coarse, floor=.63)
    narrow, narrow_enough = masked_blur(signal, available, fine, floor=.67)
    band = narrow - wide
    known = enough & narrow_enough & available

    # Physical X/Y derivatives are deliberately in the same units.
    dy = sobel(band, axis=0, mode="nearest") / (8*scale.step_y)
    dx = sobel(band, axis=1, mode="nearest") / (8*scale.step_x)
    local = neighborhood/scale.step_y, neighborhood/scale.step_x
    a, ok_a = masked_blur(dx*dx, known, local)
    b, ok_b = masked_blur(dx*dy, known, local)
    c, ok_c = masked_blur(dy*dy, known, local)
    trace = a+c
    spread = np.hypot(a-c, 2*b)
    anisotropy = spread / np.maximum(trace, 1e-9)
    orientation = (np.degrees(.5*np.arctan2(2*b, a-c)) + 90) % 180
    energy, ok_e = masked_blur(band*band, known, local, floor=.52)
    valid = known & ok_a & ok_b & ok_c & ok_e & (trace > 1e-9)
    return band, np.sqrt(energy), np.sqrt(np.maximum(trace, 0)), anisotropy, orientation, valid


def texture(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "wavelength_mm", "neighborhood_mm", "min_coherence")
    wave = scalar(opts.get("wavelength_mm", 2.4), "Feature spacing (mm)", .45, 24)
    around = scalar(opts.get("neighborhood_mm", 7.5), "Texture neighborhood (mm)", 1, 45)
    min_coh = scalar(opts.get("min_coherence", .36), "Minimum directional coherence", .04, .95)
    if wave < 2*max(ctx.scale.step_x, ctx.scale.step_y):
        raise Problem("The working image cannot resolve that feature spacing. Increase preview resolution.")
    if around < wave:
        raise Problem("Use a neighborhood at least as wide as the feature spacing.")
    band, contrast, strength, anisotropy, orientation, valid = _texture_details(frame.signal,
                    frame.measurable, ctx.scale, wave, around)
    support = ctx.demand(valid & ctx.selection)
    reliable = support & (anisotropy >= min_coh) & (contrast > .0006)
    upper = max(.003, float(np.percentile(contrast[support], 95)))
    line_data = []
    h,w = support.shape
    step_x = max(8, round(around/ctx.scale.step_x*1.65))
    step_y = max(8, round(around/ctx.scale.step_y*1.65))
    samples = []
    for y in range(step_y//2, h, step_y):
        for x in range(step_x//2, w, step_x):
            if not reliable[y,x]:
                continue
            at = orientation[y,x]
            delta_x = np.cos(np.radians(at))*around*.42/ctx.scale.width_mm
            delta_y = np.sin(np.radians(at))*around*.42/ctx.scale.height_mm
            ux, uy = (x+.5)/w, (y+.5)/h
            line_data.append(([(ux-delta_x, uy-delta_y), (ux+delta_x, uy+delta_y)], (251,226,140), ""))
            samples.append(dict(panel=[rounded(ux, 7),rounded(uy,7)], degrees=rounded(at),
                                texture_contrast=rounded(contrast[y,x]), coherence=rounded(anisotropy[y,x]),
                                gradient_rms=rounded(strength[y,x])))
    painted = mask_overlay(frame.rgb, [(support & ~reliable, (43,45,50), .5)], frame.present)
    summary = dict(spacing_mm=wave, measured_fraction=ratio(support, ctx.selection),
                   directional_fraction=ratio(reliable, support),
                   band_contrast=percentiles(contrast, support), direction_samples=len(samples))
    result = (map_asset("texture-band", band, support, ctx, low=-upper, high=upper, palette="diverging"),
              map_asset("texture-energy", contrast, support, ctx, low=0, high=upper, palette="thermal"),
              png("texture-directions", mark_image(painted, line_data), ctx),
              csv_table("direction-samples", ("panel_x","panel_y","line_angle_deg","contrast","coherence","gradient_rms"),
                        [[*row["panel"],row["degrees"],row["texture_contrast"],row["coherence"],
                          row["gradient_rms"]] for row in samples]))
    return Result("texture", "Impressed glass texture", summary,
                  {**summary, "local_samples": samples, "minimum_coherence": min_coh,
                   "neighborhood_mm": around, "note": "Short lines follow the dominant local pattern direction."}, result)


@dataclass(frozen=True)
class Residue:
    detail: np.ndarray
    valid: np.ndarray
    positive: np.ndarray
    negative: np.ndarray

    @classmethod
    def detect(cls, ctx, frame, radius, amplitude, min_pixels):
        sigma = (radius/ctx.scale.step_y, radius/ctx.scale.step_x)
        background, fill = masked_blur(frame.signal, frame.measurable, sigma, floor=.61)
        allowed = frame.measurable & fill & ctx.selection
        detail = frame.signal - background
        return cls(detail, allowed, clean_regions(allowed & (detail > amplitude), min_pixels),
                   clean_regions(allowed & (detail < -amplitude), min_pixels))


def _segment_properties(segment, scale, detail):
    yy,xx = zip(*segment, strict=True)
    coords = np.column_stack(((np.asarray(xx)+.5)*scale.step_x, (np.asarray(yy)+.5)*scale.step_y))
    distance = np.linalg.norm(np.diff(coords, axis=0), axis=1).sum()
    displace = np.linalg.norm(coords[0]-coords[-1])
    along = np.linspace(0,len(segment)-1,min(len(segment),18)).round().astype(int)
    return dict(length_mm=rounded(distance), straight_ratio=rounded(displace/max(distance, .00001)),
                peak=rounded(np.max(np.abs(detail[yy,xx]))), endpoints=coords[[0,-1]].round(5).tolist(),
                line=[[(segment[i][1]+.5)/scale.width_px,(segment[i][0]+.5)/scale.height_px] for i in along])


def crack_candidates(ctx):
    opts = ctx.order.toolbox
    frame = ctx.one()
    keys(opts, "radius_mm", "amplitude", "min_length_mm", "polarity")
    radius = scalar(opts.get("radius_mm", 1.4), "Background radius (mm)", .3, 9)
    amplitude = scalar(opts.get("amplitude", .052), "Signal change", .003, .5)
    minimum = scalar(opts.get("min_length_mm", 5), "Minimum visible run (mm)", 1, 80)
    polarity = opts.get("polarity", "bright")
    if polarity not in ("bright", "dark", "both"):
        raise Problem("Polarity can be bright, dark or both.")
    residue = Residue.detect(ctx, frame, radius, amplitude, 4)
    usable = ctx.demand(residue.valid)
    paths = []
    colors = []
    direct = {"bright": (247,219,121), "dark": (224,111,159)}
    for field, layer in (("bright", residue.positive), ("dark", residue.negative)):
        if polarity != "both" and polarity != field:
            continue
        line = centerline(layer)
        _, runs = graph_paths(line)
        selected = np.zeros(line.shape, dtype=bool)
        for run in runs:
            if len(run) < 4:
                continue
            row = _segment_properties(run, ctx.scale, residue.detail)
            if row["length_mm"] < minimum:
                continue
            row["polarity"] = field
            paths.append(row)
            yy,xx=zip(*run,strict=True)
            selected[yy,xx]=True
        if selected.any():
            colors.append((selected,direct[field],.85))
    paths.sort(key=lambda path: -path["length_mm"])
    summaries = paths[:100]
    image = mask_overlay(frame.rgb, colors[:2], frame.present)
    marks = [(row["line"], direct[row["polarity"]], str(i+1)) for i,row in enumerate(summaries)]
    summary = dict(candidates=len(paths), highlighted=min(100,len(paths)),
                   max_length_mm=summaries[0]["length_mm"] if summaries else None,
                   measured_fraction=ratio(usable, ctx.selection))
    return Result("cracks", "Crack candidate tracing", summary,
                  {**summary,"candidates":summaries,"polarity":polarity,
                   "note":"Local light ridges and troughs can also reflect pigment or surface glare."},
                  (png("traces",mark_image(image,marks),ctx),
                   map_asset("local-departure",residue.detail,usable,ctx,low=-amplitude*3,high=amplitude*3,palette="diverging")))


def _piece_index(mask, minimum):
    tokens, count = label(mask, structure=CONNECTION)
    weights = np.bincount(tokens.ravel(), minlength=count+1)
    good = weights >= minimum
    good[0] = False
    return tokens, weights, good


def glare(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "clip_code", "halo_radius_mm", "min_patch_mm2")
    threshold = scalar(opts.get("clip_code", .985), "Clip code", .82, 1)
    radius = scalar(opts.get("halo_radius_mm", 1.6), "Halo width (mm)", .2, 15)
    minimum = scalar(opts.get("min_patch_mm2",1.15), "Minimum patch area (mm²)",.05,100)
    signal = frame.signal
    valid = ctx.demand(ctx.measured(frame, allow_clipped=True))
    hard = (np.max(frame.code,axis=2) >= threshold) & valid
    labels, counts, members = _piece_index(hard, round(minimum/ctx.scale.pixel_area))
    kept = members[labels]
    aura, enough = masked_blur(kept.astype(np.float32), valid,
                              (radius/ctx.scale.step_y,radius/ctx.scale.step_x),floor=.51)
    halo = ~kept & valid & enough & (aura>.07)
    details = []
    marks = []
    h,w = kept.shape
    near=~binary_erosion(valid,structure=CONNECTION)
    for i, bounds in enumerate(find_objects(labels),1):
        if bounds is None or not members[i]:
            continue
        y,x = np.where(labels[bounds]==i)
        y,x = y+bounds[0].start,x+bounds[1].start
        center = [(float(x.mean())+.5)/w,(float(y.mean())+.5)/h]
        lower,upper = [int(x.min()),int(y.min())],[int(x.max()+1),int(y.max()+1)]
        details.append(dict(patch=len(details)+1, panel_center=center, size_mm2=rounded(counts[i]*ctx.scale.pixel_area),
                            width_mm=rounded((upper[0]-lower[0])*ctx.scale.step_x),
                            height_mm=rounded((upper[1]-lower[1])*ctx.scale.step_y),
                            saturating_channels=[int(np.sum(frame.code[...,c][y,x]>=threshold)) for c in range(3)],
                            near_unobserved=bool(np.any(near[y,x]))))
        if len(details)<=175:
            marks.append(([center],(250,221,143),str(len(details))))
    picture = mask_overlay(frame.rgb,[(kept,(239,76,65),.80),(halo,(248,186,73),.53)],frame.present)
    summary = dict(overexposed_patches=len(details), clipped_fraction=ratio(kept,valid),
                   halo_fraction=ratio(halo,valid), measured_fraction=ratio(valid,ctx.selection))
    return Result("glare","Saturated highlights",summary,{**summary,"patches":details[:175],
                   "clip_code":threshold,"note":"Brightness and channel clipping are directly supported. Halo is a vicinity flag."},
                  (png("highlight-and-halo",mark_image(picture,marks),ctx),
                   map_asset("field-strength",signal,valid,ctx,low=0,high=1)))


def edge_roughness(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts, "smoothing_mm", "gradient_floor")
    smooth = scalar(opts.get("smoothing_mm", 1.4), "Smoothing (mm)", .25, 18)
    limit = scalar(opts.get("gradient_floor", .12), "Strong gradient (linear/mm)", .01, 1)
    luma = frame.signal
    sigma = smooth/ctx.scale.step_y,smooth/ctx.scale.step_x
    local, valid = masked_blur(luma, frame.measurable, sigma)
    usable = ctx.demand(valid & ctx.measured(frame))
    dx = sobel(local,axis=1,mode="nearest") / (8*ctx.scale.step_x)
    dy = sobel(local,axis=0,mode="nearest") / (8*ctx.scale.step_y)
    gradient = np.hypot(dx,dy)
    extended = binary_erosion(usable,structure=CONNECTION)
    strong = (gradient >= limit) & extended
    raw_high = np.abs(luma-local)
    summary = dict(gradient_percentiles_per_mm=percentiles(gradient,extended),
                   mean_absolute_detail=rounded(np.mean(raw_high[extended])) if extended.any() else None,
                   strong_fraction=ratio(strong,extended))
    return Result("edges","Edge and paint-detail survey",summary,{**summary,"smoothing_mm":smooth,
                   "strong_threshold":limit,"note":"Stronger edges can come from solder, paint or hard shadows."},
                  (map_asset("gradient",gradient,extended,ctx,low=0,high=limit*2,palette="thermal"),
                   png("edges",mask_overlay(frame.rgb,[(strong,(247,217,129),.71)],frame.present),ctx)))
