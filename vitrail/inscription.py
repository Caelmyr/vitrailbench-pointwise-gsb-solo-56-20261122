from dataclasses import dataclass
import math

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, gaussian_filter, label, find_objects

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, option, rounded, scalar
from .leads import CONNECTION, CCROSS
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


def _supported_derivative(field,usable,axis):
    if axis==0:
        intervals=.5*(field[2:,:]-field[:-2,:])
        shape=np.pad(intervals,((1,1),(0,0)),mode="constant")
        keep=usable&np.roll(usable,1,axis=0)&np.roll(usable,-1,axis=0)
        keep[0,:]=keep[-1,:]=False
    else:
        intervals=.5*(field[:,2:]-field[:,:-2])
        shape=np.pad(intervals,((0,0),(1,1)),mode="constant")
        keep=usable&np.roll(usable,1,axis=1)&np.roll(usable,-1,axis=1)
        keep[:,0]=keep[:,-1]=False
    return shape,keep


@dataclass(frozen=True)
class StrokeField:
    ink: np.ndarray
    contrast: np.ndarray
    tangent: np.ndarray
    confident: np.ndarray
    good: np.ndarray

    @classmethod
    def sense(cls,ctx,frame,radius,kind,minimum):
        data=frame.signal
        allowed=frame.measurable
        sx=radius/ctx.scale.step_x
        sy=radius/ctx.scale.step_y
        trace,wide=masked_blur(data,allowed,(sy,sx),floor=.6)
        narrow,sharp=masked_blur(data,allowed,(.5,.5),floor=.7)
        mask=wide&sharp&allowed
        delta=trace-narrow if kind=="dark" else narrow-trace
        base=np.maximum(trace,.002)
        measure=np.divide(delta,base,out=np.zeros_like(delta),where=mask)
        dy,known_y=_supported_derivative(narrow,mask,0)
        dx,known_x=_supported_derivative(narrow,mask,1)
        dy=dy/ctx.scale.step_y
        dx=dx/ctx.scale.step_x
        known=known_x&known_y
        cap=(max(1,sy/3),max(1,sx/3))
        Jxx,m1=masked_blur(dx*dx,known,cap,floor=.61)
        Jxy,m2=masked_blur(dx*dy,known,cap,floor=.61)
        Jyy,m3=masked_blur(dy*dy,known,cap,floor=.61)
        ph=np.arctan2(2*Jxy,Jxx-Jyy)/2
        tangent=(np.degrees(ph)+90)%180
        reliability=np.hypot(Jxx-Jyy,2*Jxy)/np.maximum(Jxx+Jyy,1e-10)
        settled=known&m1&m2&m3&mask
        return cls((measure>=minimum)&settled,measure,tangent,reliability,settled)


def _traverse_boundaries(ctx,mask,allow):
    # Separate observed outer boundaries from bands that meet an unobserved sample.
    support=ctx.scale
    ly=support.step_y
    lx=support.step_x
    pad=np.pad(mask,1,constant_values=False)
    visible=np.pad(allow,1,constant_values=False)
    sides=[]
    for y,x,weight in ((0,1,lx),(2,1,lx),(1,0,ly),(1,2,ly)):
        delta=mask&~pad[y:y+mask.shape[0],x:x+mask.shape[1]]
        known=delta&visible[y:y+mask.shape[0],x:x+mask.shape[1]]
        sides.append((float(known.sum()*weight),float((delta&~known).sum()*weight)))
    return tuple(sum(side[i] for side in sides) for i in range(2))


def _glyph_confidence(detail,tangent,usable):
    strength=detail[usable]
    direction=tangent[usable]
    if not len(strength):
        return None,None
    angle=np.radians(direction)
    vector=np.mean(np.exp(2j*angle))
    result=(np.degrees(np.angle(vector))/2)%180
    return rounded(result),rounded(abs(vector))


def _illustration(ctx,items):
    color=(252,225,146)
    path=[]
    for item in items:
        left,top,right,bottom=item["box"]
        path.append(([(left,top),(right,top),(right,bottom),(left,bottom),(left,top)],color,str(item["fragment"])))
    return path


def inscription_survey(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"polarity","radius_mm","relative_contrast","min_area_mm2","closing_mm","coherence")
    kind=option(opts.get("polarity","dark"),"Inscription polarity",("dark","bright"))
    radius=scalar(opts.get("radius_mm",1.2),"Character neighborhood (mm)",.4,19)
    epsilon=scalar(opts.get("relative_contrast",.15),"Minimum local contrast",.02,.9)
    minimum=scalar(opts.get("min_area_mm2",.8),"Minimum mark size (mm²)",.08,150)
    bridging=scalar(opts.get("closing_mm",.3),"Fragment closing distance (mm)",0,3.8)
    directional=scalar(opts.get("coherence",.26),"Direction certainty",.1,.97)
    surface=StrokeField.sense(ctx,frame,radius,kind,epsilon)
    measured=ctx.demand(surface.good&ctx.selection)
    active=surface.ink&measured
    if bridging:
        steps=min(8,max(1,int(round(bridging/max(ctx.scale.step_x,ctx.scale.step_y)))))
        linked=(binary_closing(active,structure=CCROSS,iterations=steps)|active)&measured
    else:
        linked=active
    classified,n=label(linked,structure=CONNECTION)
    tally=np.bincount(classified.ravel(),minlength=n+1)
    accepted=(tally>=max(2,minimum/ctx.scale.pixel_area))
    accepted[0]=False
    parts=find_objects(classified)
    stats=[]
    for idx,bounds in enumerate(parts,1):
        if not accepted[idx] or bounds is None:
            continue
        y0,y1=bounds[0].start,bounds[0].stop
        x0,x1=bounds[1].start,bounds[1].stop
        fragment=classified[bounds]==idx
        core=fragment&active[bounds]
        support=fragment&(surface.confident[bounds]>=directional)
        heading,coherence=_glyph_confidence(surface.contrast[bounds],surface.tangent[bounds],support)
        label_box=[rounded(x0/ctx.scale.width_px,6),rounded(y0/ctx.scale.height_px,6),
                   rounded(x1/ctx.scale.width_px,6),rounded(y1/ctx.scale.height_px,6)]
        real_share=ratio(core,fragment)
        neighbor=measured[max(0,y0-1):y1+1,max(0,x0-1):x1+1]
        stats.append(dict(fragment=len(stats)+1,box=label_box,
                          area_mm2=rounded(tally[idx]*ctx.scale.pixel_area),actual_share=real_share,
                          expected_angle=heading,coherence=coherence,observed_neighborhood=rounded(neighbor.mean()),
                          contrast=percentiles(surface.contrast[bounds],core,(10,50,90),4)))
        if len(stats)>400:
            raise Problem("Inspect a smaller panel area or increase minimum mark area.",422)
    visible=accepted[classified]
    boundary=_traverse_boundaries(ctx,visible,measured)
    painted=mask_overlay(frame.rgb,[(visible&active,(136,216,187),.53),(visible&~active,(239,186,111),.61)],frame.present)
    summary=dict(ink_candidates=len(stats),candidate_area_mm2=rounded(visible.sum()*ctx.scale.pixel_area),
                 support_coverage=ratio(measured,ctx.selection),observed_outline_mm=rounded(boundary[0]),
                 uncertain_outline_mm=rounded(boundary[1]))
    table=csv_table("mark-fragments",("fragment","area_mm2","observed_proportion","direction_degrees","coherence"),
                    [(item["fragment"],item["area_mm2"],item["actual_share"],item["expected_angle"],item["coherence"]) for item in stats])
    return Result("inscription","Painted mark legibility",summary,{**summary,"fragments":stats,"polarity":kind,
                  "note":"Only the image evidence is assessed. The report makes no claim about words or authorship."},
                  (png("mark-contrast",mark_image(painted,_illustration(ctx,stats)),ctx),
                   map_asset("contrast",surface.contrast,measured,ctx,low=0,high=epsilon*3,palette="ink"),table))


def _tile_distribution(coords,values,valid,length,axis):
    response=[]
    for row in range(length):
        along=(coords>=row/length)&(coords<(row+1)/length)
        allowed=along&valid
        metrics=percentiles(values,allowed,(20,50,80),15)
        response.append(dict(block=row+1,span=[rounded(row/length),rounded((row+1)/length)],
                             valid_pixels=int(np.sum(allowed)),of_region=ratio(allowed,along),
                             percentiles=metrics,direction=axis))
    return response


def frosted_haze(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"reference","broad_mm","fine_mm","strips")
    narrow=scalar(opts.get("fine_mm",1.3),"Fine texture diameter (mm)",.30,12)
    broad=scalar(opts.get("broad_mm",7.5),"Haze neighborhood (mm)",narrow,31)
    reference=Box.parse(opts.get("reference",[.38,.13,.58,.30]),"Clean reference")
    bins=integer(opts.get("strips",18),"Summary strips",4,90)
    present=frame.measurable
    luma=frame.signal
    scale=ctx.scale
    median,central=masked_blur(luma,present,(broad/scale.step_y,broad/scale.step_x),floor=.66)
    texture,confident=masked_blur(luma,present,(narrow/scale.step_y,narrow/scale.step_x),floor=.70)
    magnitude,observed=masked_blur((luma-texture)**2,confident&present,(broad/scale.step_y,broad/scale.step_x),floor=.55)
    relative=np.sqrt(np.maximum(magnitude,0))/np.maximum(median,.012)
    active=central&observed&present
    reference_mask=reference.mask(scale.width_px,scale.height_px)
    ctx.demand(active&reference_mask,floor=0,pixels=30)
    basis=float(np.median(relative[reference_mask&active]))
    eligible=ctx.demand(active&ctx.selection)
    contrast_drop=20*np.log10(np.maximum(relative,.0008)/max(.0008,basis))
    grid=scale.grid()
    slices=_tile_distribution(grid[...,0],contrast_drop,eligible,bins,"horizontal")
    data=chart("detail-loss-profile",[(r["block"]-.5)*scale.width_mm/bins for r in slices],
               [("Median",(135,89,161),[r["percentiles"][1] for r in slices])],"Relative fine-detail loss","dB")
    summary=dict(reference_contrast=rounded(basis),observed_fraction=ratio(eligible,ctx.selection),
                 variation_db=percentiles(contrast_drop,eligible),
                 lower_detail_fraction=ratio(eligible&(contrast_drop<-4),eligible))
    return Result("haze","Clouded glass detail survey",summary,
                  {**summary,"profile":slices,"clean_reference":reference.public(),
                   "note":"Texture loss is a visual clue. Glass color, cast details and surface haze can produce the same result."},
                  (map_asset("detail-departure-db",contrast_drop,eligible,ctx,low=-15,high=15,palette="diverging"),data))
