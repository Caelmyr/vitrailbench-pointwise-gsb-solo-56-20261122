from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label, find_objects
from scipy.stats import spearmanr

from .bench import Result, masked_blur, percentiles, ratio
from .camera import brightness
from .contracts import Box, Problem, integer, keys, rounded, scalar
from .leads import CONNECTION
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


_MIN=.001


def _repeat_support(ctx,frames,ref):
    shared=ctx.selection.copy()
    source_selection=ref.mask(ctx.scale.width_px,ctx.scale.height_px)
    for capture in frames:
        shared &= capture.measurable & (capture.signal>_MIN)
        source_selection &= capture.measurable & (capture.signal>_MIN)
    ctx.demand(shared,floor=.1,pixels=40)
    if source_selection.sum()<65:
        raise Problem("The repeat comparison needs 65 commonly observed reference pixels.",422)
    return shared,source_selection


@dataclass(frozen=True)
class RepeatRanges:
    scale: np.ndarray
    absolute: np.ndarray
    noise: np.ndarray
    support: np.ndarray
    obs_counts: np.ndarray
    ids: tuple

    @classmethod
    def solve(cls,ctx,frames,reference,normalize):
        common, measured_ref=_repeat_support(ctx,frames,reference)
        signals=np.stack([capture.signal for capture in frames],axis=0)
        if normalize:
            logs=np.log2(np.maximum(signals,_MIN))
            values=np.median(logs[:,measured_ref],axis=1)
            shift=values-values[0]
        else:
            shift=np.zeros(len(frames),dtype=np.float32)
        adjusted=signals/(2**shift[:,None,None])
        center=np.median(adjusted,axis=0)
        rss=np.subtract(adjusted,center[None,...])
        sigma=np.sqrt(np.mean(rss**2,axis=0))
        greatest=np.max(np.abs(rss),axis=0)
        observed=np.stack([frame.measurable for frame in frames]).sum(axis=0)
        return cls(shift,greatest,sigma,common,observed,tuple(capture.record["id"] for capture in frames))


def repeatability(ctx):
    opts=ctx.order.toolbox
    keys(opts,"reference","threshold","normalize")
    if len(ctx.frames)<2:
        raise Problem("Select at least two repeated captures of the panel.")
    tolerance=scalar(opts.get("threshold",.028),"Absolute variation limit",.003,.3)
    reference=Box.parse(opts.get("reference",[.4,.3,.6,.7]),"Repeated reference")
    compare=RepeatRanges.solve(ctx,ctx.frames,reference,bool(opts.get("normalize",False)))
    abnormal=compare.support&(compare.absolute>tolerance)
    obs=ctx.selection&~compare.support
    ref=ctx.frames[0]
    picture=mask_overlay(ref.rgb,[(abnormal,(227,123,165),.60),(obs,(171,155,113),.47)],ref.present)
    counts=[{"captures_used":i,"pixels":int(np.sum(ctx.selection&(compare.obs_counts==i)))}
            for i in range(len(ctx.frames)+1)]
    table=csv_table("repeatability-support",("capture_count","pixels"),
                    [(row["captures_used"],row["pixels"]) for row in counts])
    anchors=[dict(capture=fid,offset_ev=rounded(offset)) for fid,offset in zip(compare.ids,compare.scale,strict=True)]
    summary=dict(selected_captures=len(ctx.frames),comparison_pixels=int(compare.support.sum()),
                 out_of_tolerance_fraction=ratio(abnormal,compare.support),
                 variation_levels=percentiles(compare.noise,compare.support,(10,50,80,95)))
    return Result("repeat","Capture repeatability",summary,
                  {**summary,"signed_offsets":anchors,"available_captures":counts,"reference":reference.public(),
                   "threshold":tolerance,"note":"Comparisons only use registrations shared by the selected input stack."},
                  (map_asset("relative-variation",compare.absolute,compare.support,ctx,low=0,high=tolerance*2,palette="thermal"),
                   png("coverage-of-repeats",picture,ctx),table))


def _log_slope(codes_a,codes_b,intervals,max_sample=4350):
    subset=np.linspace(0,len(codes_a)-1,min(len(codes_a),max_sample)).astype(int)
    a=codes_a[subset]
    b=codes_b[subset]
    if float(np.std(a))<1e-8 or float(np.std(b))<1e-8:
        order=None
    else:
        order=rounded(spearmanr(a,b).statistic)
    quantiles_a=np.quantile(a,intervals)
    quantiles_b=np.quantile(b,intervals)
    return order,quantiles_a,quantiles_b


def exposure_check(ctx):
    first,second=ctx.two()
    opts=ctx.order.toolbox
    keys(opts,"minimum_linear","interval_count","max_channel_drift")
    signal=scalar(opts.get("minimum_linear",.020),"Lowest reliable exposure signal",.002,.25)
    bins=integer(opts.get("interval_count",16),"Number of distribution bins",6,33)
    ev_limit=scalar(opts.get("max_channel_drift",.28),"Channel mismatch threshold (EV)",.02,2)
    first_signal=first.signal
    second_signal=second.signal
    measure=ctx.demand(ctx.shared()&(first_signal>signal)&(second_signal>signal))
    n=int(measure.sum())
    channels=["red","green","blue"]
    probs=np.linspace(.035,.965,bins)
    stats=[]
    profiles=[]
    colors=((183,89,74),(102,152,111),(76,128,179))
    shifts=np.log2(np.maximum(second.rgb,_MIN)/np.maximum(first.rgb,_MIN))
    center=np.median(shifts[measure],axis=0)
    hue_shift=np.max(shifts-center,axis=2)-np.min(shifts-center,axis=2)
    divergent=measure&(hue_shift>=ev_limit)
    for band,name in enumerate(channels):
        a,b=first.rgb[...,band][measure],second.rgb[...,band][measure]
        corr,q1,q2=_log_slope(a,b,probs)
        ev=np.log2(np.maximum(q2,_MIN)/np.maximum(q1,_MIN))
        stats.append(dict(channel=name,rank_agreement=corr,ev_at_quantile=[rounded(x) for x in ev],
                          first=[rounded(x) for x in q1],second=[rounded(x) for x in q2],
                          median_ev=rounded(center[band])))
        profiles.append((name,colors[band],ev))
    overlay=mask_overlay(first.rgb,[(divergent,(248,162,113),.64)],first.present)
    summary=dict(measured_pixels=n,measured_fraction=ratio(measure,ctx.selection),
                 diverging_pixels=int(divergent.sum()),
                 per_channel_median_ev=[rounded(x) for x in center])
    graphic=chart("exposure-trends",probs,profiles,"Exposure offset across percentiles","EV",x_caption="Signal quantile")
    table=csv_table("channel-differences",("channel","quantile","first","second","ev"),
                  [(name,rounded(q),first_val,second_val,diff) for name,item in zip(channels,stats,strict=True)
                    for q,first_val,second_val,diff in zip(probs,item["first"],item["second"],item["ev_at_quantile"],strict=True)])
    return Result("exposure","Exposure consistency check",summary,{**summary,"channels":stats,"threshold":ev_limit,
                  "note":"This report can identify color-dependent drift across two observations. It retains no corrected capture."},
                  (graphic,png("divergent-exposure",overlay,ctx),table))


def _sampling_bands(scale,count):
    w,h=scale.width_px,scale.height_px
    yy,xx=np.mgrid[:h,:w]
    limits=np.minimum(np.minimum((xx+.5)*scale.step_x,(w-.5-xx)*scale.step_x),
                      np.minimum((yy+.5)*scale.step_y,(h-.5-yy)*scale.step_y))
    maximum=min(scale.width_mm,scale.height_mm)/2
    ends=np.linspace(0,maximum,count+1)
    return limits,ends


def border_shadow(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"zones","darkness_limit","middle")
    n=integer(opts.get("zones",24),"Number of inset steps",6,80)
    allowable=scalar(opts.get("darkness_limit",.65),"Low signal relative to middle",.10,.95)
    core=Box.parse(opts.get("middle",[.32,.31,.68,.70]),"Central comparison")
    good=ctx.demand(ctx.measured(frame))
    basis=ctx.demand(good&core.mask(ctx.scale.width_px,ctx.scale.height_px),floor=0,pixels=45)
    signal=frame.signal
    midpoint=np.median(signal[basis])
    if midpoint<_MIN:
        raise Problem("The center reference is too dark for a margin comparison.",422)
    metric=signal/midpoint
    x,ends=_sampling_bands(ctx.scale,n)
    rows=[]
    for j in range(n):
        p,q=ends[j:j+2]
        selection=(x>=p)&(x<q)&ctx.selection
        selected=good&selection
        num=int(selected.sum())
        if num<22:
            levels=[None]*3
        else:
            levels=percentiles(metric,selected,(10,50,90),22)
        rows.append(dict(distance_mm=rounded((p+q)/2),coverage=ratio(selected,selection),samples=num,
                         ranges=levels,low_share=ratio(selected&(metric<allowable),selected) if num>=22 else None))
    shadow=good&(metric<allowable)
    art=mask_overlay(frame.rgb,[(shadow,(124,164,229),.52)],frame.present)
    plot=chart("border-falloff",[item["distance_mm"] for item in rows],
               [("median",(173,100,64),[item["ranges"][1] for item in rows]),
                ("P10",(87,137,159),[item["ranges"][0] for item in rows])],
               "Signal with distance from panel edge","Ratio")
    summary=dict(middle_median=rounded(midpoint),poor_edge_fraction=ratio(shadow,good),
                 measured_fraction=ratio(good,ctx.selection),inset_sections=n)
    return Result("border-shadow","Panel edge shading",summary,{**summary,"bands":rows,"fraction_limit":allowable},
                  (plot,png("low-signal",art,ctx),
                   csv_table("inset-bands",("distance_mm","usable_pixels","coverage","p10","p50","p90","low"),
                             [(r["distance_mm"],r["samples"],r["coverage"],*r["ranges"],r["low_share"]) for r in rows])))


def _hist_channel(source,valid,index,bins):
    measured=source[...,index][valid]
    counts,ends=np.histogram(measured,range=(0,1),bins=bins)
    above=int(np.sum(measured>.995))
    darkness=int(np.sum(measured<.013))
    return counts,ends,above,darkness


def channel_clip(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"clip_code","bins","shadow_code")
    limit=scalar(opts.get("clip_code",.99),"Near-clipping code",.85,1)
    shadow=scalar(opts.get("shadow_code",.04),"Dark code",.001,.25)
    bins=integer(opts.get("bins",40),"Histogram bins",10,100)
    measured=ctx.demand(ctx.measured(frame,allow_clipped=True))
    palettes=[(235,117,101),(110,207,148),(107,165,233)]
    seq=[]
    channel_stats=[]
    csv_rows=[]
    serial=("red","green","blue")
    for i,name in enumerate(serial):
        d=frame.code[...,i]
        mask=measured&(d>=limit)
        seq.append((mask,palettes[i],.72))
        counts,ends,peak,deep=_hist_channel(frame.code,measured,i,bins)
        row=dict(channel=name,near_clip=int(mask.sum()),dark=int(np.sum(measured&(d<shadow))),
                 precise_high=peak,precise_low=deep,codes=percentiles(d,measured),histogram=[int(x) for x in counts])
        channel_stats.append(row)
        csv_rows.extend([(name,rounded(ends[j]),rounded(ends[j+1]),int(c)) for j,c in enumerate(counts)])
    color=mask_overlay(frame.rgb,seq,frame.present)
    x=(ends[1:]+ends[:-1])/2
    signal=chart("histograms",x,[(n,p,[int(z) for z in row["histogram"]]) for (n,row,p) in zip(serial,channel_stats,palettes,strict=True)],
                 "Camera code distributions","Pixels",x_caption="Encoded channel level")
    summary=dict(capture_px=int(measured.sum()),pixels_with_clipping=int((measured&np.any(frame.code>=limit,axis=-1)).sum()),
                 lowest_code=shadow,threshold=limit)
    return Result("channels","Camera channel clipping",summary,{**summary,"channels":channel_stats},
                  (signal,png("clip-colors",color,ctx),
                   csv_table("code-histograms",("channel","code_lower","code_upper","pixels"),csv_rows)))
