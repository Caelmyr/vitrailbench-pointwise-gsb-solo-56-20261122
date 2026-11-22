from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, label, find_objects

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, option, rounded, scalar
from .leads import CONNECTION, CCROSS, centerline
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


@dataclass(frozen=True)
class PatchShape:
    centroid: tuple
    extent: tuple
    axes_mm: tuple
    angle: float
    perimeter_mm: float
    normalized_bbox: tuple

    @classmethod
    def measure(cls,region,window,scale):
        ys,xs=np.where(region)
        x0,y0,x1,y1=window
        point=np.stack(((xs+x0+.5)*scale.step_x,(ys+y0+.5)*scale.step_y),axis=1)
        center=point.mean(axis=0)
        offsets=point-center[None,:]
        cov=offsets.T@offsets/max(1,len(xs))
        val,vec=np.linalg.eigh(cov)
        axes=4*np.sqrt(np.maximum(val,0))
        z=vec[:,-1]
        theta=degrees(atan2(z[1],z[0]))%180
        buf=np.pad(region,1)
        perimeter=float(np.count_nonzero(np.diff(buf,axis=0))*scale.step_x+
                        np.count_nonzero(np.diff(buf,axis=1))*scale.step_y)
        total_width=scale.width_px
        total_height=scale.height_px
        return cls((float(center[0]),float(center[1])),(x1-x0,y1-y0),(float(axes[-1]),float(axes[0])),float(theta),perimeter,
                   ((x0+float(xs.min()))/total_width,(y0+float(ys.min()))/total_height,
                    (x0+float(xs.max()+1))/total_width,(y0+float(ys.max()+1))/total_height))

    def public(self):
        return dict(centroid_mm=[rounded(x) for x in self.centroid],extent_px=list(self.extent),
                    fitted_axes_mm=[rounded(x) for x in self.axes_mm],angle_degrees=rounded(self.angle,2),
                    perimeter_mm=rounded(self.perimeter_mm),bounds=self.normalized_bbox)


def _local_lighting(frame,ctx,radius,min_support):
    sigma=radius/ctx.scale.step_y,radius/ctx.scale.step_x
    baseline,support=masked_blur(frame.signal,frame.measurable,sigma,floor=min_support)
    return frame.signal-baseline,ctx.measured(frame)&support


def _spot_region_info(coded,num,bounds,scale,residue,present):
    if bounds is None:
        return None
    y0,y1=bounds[0].start,bounds[0].stop
    x0,x1=bounds[1].start,bounds[1].stop
    item=coded[bounds]==num
    stats=PatchShape.measure(item,(x0,y0,x1,y1),scale)
    yy,xx=np.where(item)
    near_loss=bool((~present[yy+y0,xx+x0]).any())
    return item,stats,near_loss


def seed_inclusions(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"radius_mm","cutoff","min_area","max_area","polarity")
    radius=scalar(opts.get("radius_mm",1.5),"Glass neighborhood (mm)",.4,12)
    cutoff=scalar(opts.get("cutoff",.042),"Local signal difference",.006,.6)
    low=scalar(opts.get("min_area",.27),"Minimum spot area (mm²)",.06,50)
    high=scalar(opts.get("max_area",10),"Maximum spot area (mm²)",low,200)
    side=option(opts.get("polarity","dark"),"Signal polarity",("dark","bright"))
    residue,supported=_local_lighting(frame,ctx,radius,.66)
    floor=max(4,int(low/ctx.scale.pixel_area))
    ceiling=max(floor,int(high/ctx.scale.pixel_area))
    required=ctx.demand(supported)
    candidate=required&((-residue if side=="dark" else residue)>cutoff)
    codes,count=label(candidate,structure=CONNECTION)
    count_sizes=np.bincount(codes.ravel(),minlength=count+1)
    keep=(count_sizes>=floor)&(count_sizes<=ceiling)
    keep[0]=False
    suitable=keep[codes]
    outlines=binary_erosion(frame.measurable,structure=CONNECTION)
    rows=[]
    shapes=[]
    for index,bounds in enumerate(find_objects(codes),1):
        if not keep[index]:
            continue
        component,selection,close_to_loss=_spot_region_info(codes,index,bounds,ctx.scale,residue,outlines)
        axes=selection.axes_mm
        thinness=axes[-1]/max(axes[0],.000001)
        if thinness < .21:
            continue
        shape=selection.public()
        if len(rows)>=300:
            raise Problem("The window produces more than 300 candidate spots. Select a smaller region.",422)
        center=(selection.centroid[0]/ctx.scale.width_mm,selection.centroid[1]/ctx.scale.height_mm)
        shape.update(id=len(rows)+1,area_mm2=rounded(component.sum()*ctx.scale.pixel_area),
                     median_signal_difference=rounded(np.median(residue[bounds][component])), uncertain_border=close_to_loss,
                     centered_panel=[rounded(p,7) for p in center])
        rows.append(shape)
        shapes.append(([center],(245,194,114) if not close_to_loss else (231,108,113),str(len(rows))))
    painted=mask_overlay(frame.rgb,[(suitable,(230,199,115),.50)],frame.present)
    positions=[item["centered_panel"] for item in rows]
    summary=dict(spots=len(rows),component_pixels=int(candidate.sum()),usable_fraction=ratio(required,ctx.selection),
                 reported_area_mm2=rounded(sum(item["area_mm2"] for item in rows)))
    table=csv_table("inclusion-candidates",("spot","area_mm2","x_mm","y_mm","major_mm","minor_mm","confidence","difference"),
                    [(row["id"],row["area_mm2"],*row["centroid_mm"],*row["fitted_axes_mm"],"boundary" if row["uncertain_border"] else "clear",
                       row["median_signal_difference"]) for row in rows])
    return Result("inclusions","Bubble and inclusion candidates",summary,
                  {**summary,"candidates":rows,"area_window":[low,high],"polarity":side,
                   "note":"A compact departure is an image candidate. Dust and bubbles require visual review."},
                  (png("spot-locations",mark_image(painted,shapes),ctx),table))


def _nearest_boundary_stats(ctx,frame,zone,good,texture_size):
    size=ctx.scale
    dist=distance_transform_edt(zone,sampling=(size.step_y,size.step_x))
    near=zone&(dist<=texture_size)&good
    far=zone&(dist>=texture_size*2)&good
    luma=frame.signal
    return {
        "near_px":int(near.sum()),"interior_px":int(far.sum()),
        "near_luma":percentiles(luma,near,(10,50,90),35),
        "interior_luma":percentiles(luma,far,(10,50,90),35),
        "near_coverage":ratio(near,zone&(dist<=texture_size)),
        "interior_coverage":ratio(far,zone&(dist>=texture_size*2)),
    },dist,near,far


def boundary_intensity(ctx):
    from .survey import Mark, _polygon_image
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"boundary","edge_mm","baseline")
    geometry=opts.get("boundary",[[.38,.38],[.63,.39],[.62,.61],[.39,.60]])
    row=Mark.parse({"kind":"polygon","role":"feature","points":geometry,"title":"Boundary"})
    min_depth=scalar(opts.get("edge_mm",2.7),"Boundary step (mm)",.4,28)
    maximum=scalar(opts.get("baseline",.4),"Display scale",.02,1)
    marker=_polygon_image((ctx.scale.width_px,ctx.scale.height_px),row.points)
    keep=ctx.demand(marker&ctx.measured(frame),floor=0,pixels=40)
    values,distance,near,far=_nearest_boundary_stats(ctx,frame,marker,keep,min_depth)
    res=frame.signal
    ribbon=mask_overlay(frame.rgb,[(near,(224,163,110),.70),(far,(128,188,173),.32)],frame.present)
    row_outline=[*row.points,row.points[0]]
    diag=mark_image(ribbon,[(row_outline,(250,221,149),"Boundary")])
    summary=dict(observed_share=ratio(keep,marker),edge_readings=values["near_luma"],
                 interior_readings=values["interior_luma"],measured_area_mm2=rounded(keep.sum()*ctx.scale.pixel_area))
    return Result("boundary","Marked-region boundary intensity",summary,
                  {**summary,"readout":values,"boundary":row.public(),"band_mm":min_depth},
                  (png("inner-and-outer",diag,ctx),
                   map_asset("distance-from-mark",distance,keep,ctx,low=0,high=max(min_depth*3,.1)),
                   map_asset("local-signal",res,keep,ctx,low=0,high=maximum)))


def _support_counts(frame,ctx,floor):
    clearance=distance_transform_edt(frame.measurable,sampling=(ctx.scale.step_y,ctx.scale.step_x))
    core=(clearance>=floor)&ctx.selection
    invalid=ctx.selection&~frame.measurable
    return core,invalid,clearance


def support_distance(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"buffer_mm","limit_mm")
    width=scalar(opts.get("buffer_mm",2.5),"Support buffer (mm)",.25,38)
    limit=scalar(opts.get("limit_mm",14),"Display limit (mm)",1,200)
    core,dead,clearance=_support_counts(frame,ctx,width)
    ring=ctx.selection&~dead&~core
    photo=mask_overlay(frame.rgb,[(dead,(221,112,111),.62),(ring,(235,173,103),.52)],frame.present)
    summary=dict(usable_inside_buffer=ratio(core,ctx.selection),margin_fraction=ratio(ring,ctx.selection),
                 withdrawn_fraction=ratio(dead,ctx.selection),minimum_required_clearance=width)
    return Result("support-buffer","Continuous support distance",summary,
                  {**summary,"distance_to_nearest_unavailable":percentiles(clearance,core),
                   "note":"Distances are calculated on the current registered plane. Camera saturation breaks continuous support."},
                  (map_asset("support-distance-mm",clearance,ctx.selection&frame.measurable,ctx,low=0,high=limit),
                   png("supported-interior",photo,ctx)))
