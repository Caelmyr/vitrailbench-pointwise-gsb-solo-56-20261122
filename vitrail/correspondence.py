from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.signal import fftconvolve

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, label, object_data, pair, rounded, scalar
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


@dataclass(frozen=True)
class ControlPoint:
    name: str
    first: tuple
    second: tuple

    @classmethod
    def read(cls,value,index):
        raw=object_data(value,"Matching marks")
        keys(raw,"name","first","second")
        name=label(raw.get("name",f"Mark {index}"),"Mark name",64)
        return cls(name,pair(raw.get("first"),"First capture point"),pair(raw.get("second"),"Second capture point"))


def _edge_features(frame,sigma,minimum):
    source=frame.signal
    usable=frame.measurable
    average,support=masked_blur(source,usable,sigma,floor=.60)
    fy=np.gradient(average,axis=0)
    fx=np.gradient(average,axis=1)
    signal=np.hypot(fx,fy)
    return signal,usable&support&(signal>=minimum)


def _point_offsets(first,second,control,ctx):
    a=np.array(control.first)
    b=np.array(control.second)
    reference=first.alignment.to_panel([a])[0]
    comparison=second.alignment.to_panel([b])[0]
    u,v=reference
    s,t=comparison
    posed=all(0<=x<=1 for x in (u,v,s,t))
    valid=False
    if posed:
        ra,ca=ctx.scale.point_px(reference)
        rb,cb=ctx.scale.point_px(comparison)
        valid=first.present[ca,ra] and second.present[cb,rb]
    dims=np.array([ctx.scale.width_mm,ctx.scale.height_mm])
    displacement=(comparison-reference)*dims
    return {"name":control.name,"first_camera":list(a),"second_camera":list(b),
            "first_panel":[rounded(x) for x in reference],"second_panel":[rounded(x) for x in comparison],
            "drift_mm":[rounded(x) for x in displacement],"distance_mm":rounded(np.linalg.norm(displacement)),
            "relation":"present" if valid else ("off-panel" if not posed else "missing")}


def _spatial_variation(steps):
    pairs=[]
    for left,right in combinations(steps,2):
        diff=np.asarray(left["drift_mm"])-right["drift_mm"]
        d=float(np.linalg.norm(diff))
        pairs.append((d,left["name"],right["name"]))
    if not pairs:
        return None,None
    return rounded(np.median([item[0] for item in pairs])),[name for name in max(pairs)[1:]]


def pair_control(ctx):
    first,second=ctx.two()
    opts=ctx.order.toolbox
    keys(opts,"controls","tolerance_mm")
    tolerance=scalar(opts.get("tolerance_mm",2),"Allowed displacement (mm)",.1,80)
    raw=opts.get("controls",[
        {"name":"Upper left","first":[.2,.24],"second":[.2,.24]},
        {"name":"Center","first":[.48,.48],"second":[.48,.48]},
        {"name":"Lower right","first":[.69,.76],"second":[.69,.76]}])
    if not isinstance(raw,list) or not 1<=len(raw)<=80:
        raise Problem("Provide one to eighty matching pairs.")
    points=[ControlPoint.read(row,index) for index,row in enumerate(raw,1)]
    if len({point.name for point in points})!=len(points):
        raise Problem("Matching marks require different names.")
    rows=[_point_offsets(first,second,p,ctx) for p in points]
    artwork=[]
    for row in rows:
        difference=row["distance_mm"]
        color=(137,222,190) if row["relation"]=="present" and difference<tolerance else (242,133,113)
        artwork.append(([row["first_panel"],row["second_panel"]],color,row["name"]))
    seen=[row for row in rows if row["relation"]=="present"]
    scores=np.asarray([row["distance_mm"] for row in seen])
    typical,worst=_spatial_variation(seen)
    output=mask_overlay(first.rgb,[(second.present&~first.present,(163,160,209),.51)],first.present)
    overview=mark_image(output,artwork)
    summary=dict(pairs=len(rows),observed_pairs=len(seen),residual_mm=rounded(np.sqrt(np.mean(scores**2))) if len(scores) else None,
                 over_limit=sum(row["distance_mm"]>tolerance for row in seen),median_change_between_pairs_mm=typical)
    table=csv_table("correspondence",("name","left_x","left_y","right_x","right_y","drift_x_mm","drift_y_mm","relation"),
                    [(row["name"],*row["first_panel"],*row["second_panel"],*row["drift_mm"],row["relation"]) for row in rows])
    return Result("matching-pairs","Paired capture mark residuals",summary,
                  {**summary,"points":rows,"largest_drift_between":worst,"tolerance_mm":tolerance,
                   "note":"Manual corresponding spots measure an alignment. The report does not move either original."},
                  (png("position-residuals",overview,ctx,crop=False),table))


def _local_rms(value,mask):
    data=value[mask]
    if len(data)<24:
        return None
    return rounded(np.sqrt(np.mean(data**2)))


def _focus_drift(a,b,valid):
    ab=valid&(a>.000001)&(b>.000001)
    if ab.sum()<24:
        return None
    db=20*np.log10(np.maximum(b[ab],.000001)/np.maximum(a[ab],.000001))
    return rounded(np.median(db))


def _alignment_difference(ctx,first,second,scale,threshold):
    radius=(max(.7,scale/ctx.scale.step_y),max(.7,scale/ctx.scale.step_x))
    features_a,valid_a=_edge_features(first,radius,threshold)
    features_b,valid_b=_edge_features(second,radius,threshold)
    ratio_field=(features_b-features_a)/np.maximum(features_a+features_b,.0005)
    shared=ctx.shared()
    automatic=ctx.demand(shared&(valid_a|valid_b),floor=.005,pixels=32)
    both=shared&valid_a&valid_b
    return features_a,features_b,ratio_field,automatic,both,shared


def registered_edges(ctx):
    first,second=ctx.two()
    opts=ctx.order.toolbox
    keys(opts,"analysis_scale_mm","threshold","only_one_amplitude")
    scale=scalar(opts.get("analysis_scale_mm",1.3),"Edge neighborhood (mm)",.24,12)
    floor=scalar(opts.get("threshold",.007),"Minimum edge energy",.0008,.5)
    exclusive=scalar(opts.get("only_one_amplitude",.55),"Single-image edge strength",.15,.95)
    a,b,variation,automatic,both,shared=_alignment_difference(ctx,first,second,scale,floor)
    source_only=automatic&(variation<=-exclusive)
    comparison_only=automatic&(variation>=exclusive)
    unchanged=automatic&~source_only&~comparison_only
    layout=mask_overlay(first.rgb,[(source_only,(228,133,116),.66),(comparison_only,(130,180,238),.68),
                                    (unchanged&(variation<.10)&(variation>-.10),(147,207,156),.42)],first.present)
    summary=dict(shared=ratio(shared,ctx.selection),strong_edge_support=ratio(automatic,shared),
                 shared_edge_response=ratio(both,automatic),reference_only=ratio(source_only,automatic),
                 comparison_only=ratio(comparison_only,automatic),
                 typical_drift_db=_focus_drift(a,b,both),strong_deviation_rms=_local_rms(variation,automatic))
    return Result("paired-edges","Registered edge comparison",summary,{**summary,"scale_mm":scale,"threshold":floor,
                  "note":"Similar edge strengths can occur in unrelated patterns. Differences are cues for alignment review."},
                  (png("paired-edge-overview",layout,ctx),
                   map_asset("edge-change",variation,automatic,ctx,low=-1,high=1,palette="diverging")))


def _scalar_autocorrelation(values,mask,limit):
    window=np.asarray(values,dtype=np.float64)
    valid=mask.astype(np.float64)
    center=np.sum(window*valid)/max(1,valid.sum())
    window=(window-center)*valid
    power=float(np.sum(window*window))
    if power<1e-7:
        raise Problem("The selected region contains too little visual variation.",422)
    alignments=fftconvolve(valid,valid[::-1,::-1],mode="full")
    numerator=fftconvolve(window,window[::-1,::-1],mode="full")
    h,w=window.shape
    by,bx=min(limit,h-2),min(limit,w-2)
    cuts=(slice(h-1-by,h+by),slice(w-1-bx,w+bx))
    coverage=np.maximum(np.rint(alignments[cuts]),0)/valid.sum()
    strength=numerator[cuts]/power/np.maximum(coverage,.0001)
    return np.clip(strength,-1,1),coverage,by,bx


def pattern_spacing(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"step_budget_px","display_limit","min_overlap")
    limit=integer(opts.get("step_budget_px",65),"Maximum pattern lag (working pixels)",10,145)
    display=scalar(opts.get("display_limit",.70),"Correlation display amplitude",.15,1)
    floor=scalar(opts.get("min_overlap",.66),"Minimum image support at shift",.25,.95)
    x0,y0,x1,y1=ctx.order.region.indices(ctx.scale.width_px,ctx.scale.height_px)
    if min(x1-x0,y1-y0)<14:
        raise Problem("Use a larger patch for pattern-spacing measurement.")
    bounds=np.s_[y0:y1,x0:x1]
    observed=frame.measurable[bounds]
    if observed.mean()<.50:
        raise Problem("At least half of this window needs measured pixels.",422)
    relief=frame.signal[bounds]
    strength,overlap,by,bx=_scalar_autocorrelation(relief,observed,limit)
    af=overlap>=floor
    candidate=(strength>=.36)&af
    cy,cx=np.mgrid[-by:by+1,-bx:bx+1]
    distance=np.hypot(cx*ctx.scale.step_x,cy*ctx.scale.step_y)
    candidate&=(distance>2.2)
    from scipy.ndimage import maximum_filter
    peaks=candidate&(strength==maximum_filter(strength,size=5,mode="nearest"))
    order=np.argsort(-strength[peaks])[:24]
    offsets=list(zip(cx[peaks][order],cy[peaks][order],strict=True))
    matches=[]
    for x,y in offsets:
        if y<0 or y==0 and x<0:
            continue
        mm=[rounded(x*ctx.scale.step_x),rounded(y*ctx.scale.step_y)]
        matches.append(dict(lag_px=[int(x),int(y)],lag_mm=mm,correlation=rounded(strength[y+by,x+bx]),
                            supporting_overlap=rounded(overlap[y+by,x+bx])))
        if len(matches)>=12:
            break
    from .picture import colorize
    picture=png("signed-autocorrelation",colorize(strength,-display,display,"diverging",af))
    hline=chart("horizontal-pattern",cx[by]*ctx.scale.step_x,[("Correlation",(145,90,144),strength[by])],
                "Horizontal shift response","Correlation")
    cells=csv_table("repeat-distances",("lag_x_mm","lag_y_mm","correlation","observed_overlap"),
                    [(*entry["lag_mm"],entry["correlation"],entry["supporting_overlap"]) for entry in matches])
    summary=dict(measured_fraction=rounded(observed.mean()),periodic_candidates=len(matches),max_lag_px=[bx,by])
    return Result("periodic","Pattern spacing survey",summary,{**summary,"offsets":matches,"min_overlap":floor},
                  (picture,hline,cells))
