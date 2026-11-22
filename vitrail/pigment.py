from dataclasses import dataclass
from math import ceil
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, find_objects, gaussian_filter, label

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, object_data, option, rounded, scalar
from .leads import CONNECTION
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png
from .photometry import PHOTOMETRIC_FLOOR, rgb_to_lab


DARKNESS_FLOOR=.003
COLOR_COEFFICIENTS=np.array([.2126,.7152,.0722])


@dataclass(frozen=True)
class PaintBackground:
    level: tuple
    samples: int
    support_fraction: float
    irreg: float

    @classmethod
    def from_frame(cls,frame,context,reference):
        shape=frame.present.shape
        region=reference.mask(shape[1],shape[0])
        measurable=frame.measurable & region
        values=frame.rgb[measurable]
        if len(values)<45:
            raise Problem("Paint background patch needs at least 45 clear pixels.",422)
        channels=np.median(values,axis=0)
        if np.any(channels<DARKNESS_FLOOR):
            raise Problem("Choose a brighter glass-only patch for the paint reference.",422)
        q10,q90=np.percentile(values@COLOR_COEFFICIENTS,[10,90])
        raw=(-float(q10)+float(q90))/max(float(channels@COLOR_COEFFICIENTS),.001)
        return cls(tuple(channels),len(values),min(1,float(len(values)/region.sum())),raw)

    def public(self):
        return dict(reference_linear=[rounded(x) for x in self.level],samples=self.samples,
                    coverage=rounded(self.support_fraction),normalized_spread=rounded(self.irreg))


def _density_plane(rgb,reference):
    factor=np.maximum(rgb,DARKNESS_FLOOR)/np.asarray(reference.level)
    channels=-np.log10(factor)
    linear=np.sum(channels*COLOR_COEFFICIENTS,axis=-1)
    difference=channels.max(axis=-1)-channels.min(axis=-1)
    return linear,difference


def _clean_blobs(mask,min_area,max_area):
    regions,total=label(mask,structure=CONNECTION)
    sizes=np.bincount(regions.ravel(),minlength=total+1)
    sizes[0]=0
    accept=(sizes>=min_area)&(sizes<=max_area)&(sizes>0)
    return accept[regions],regions,sizes,accept


def paint_report(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"background","absorption","selectivity","min_patch_mm2","max_patch_mm2")
    original=Box.parse(opts.get("background",[.41,.14,.58,.24]),"Clear-glass reference")
    density_limit=scalar(opts.get("absorption",.25),"Absorption threshold",.02,2)
    stretch=scalar(opts.get("selectivity",.22),"Maximum channel disagreement",.01,1.8)
    low_area=scalar(opts.get("min_patch_mm2",1.8),"Minimum marked area (mm²)",.1,300)
    high_area=scalar(opts.get("max_patch_mm2",1300),"Maximum marked area (mm²)",low_area,50000)
    ref=PaintBackground.from_frame(frame,ctx,original)
    data, spectral=_density_plane(frame.rgb,ref)
    allow=ctx.demand(ctx.measured(frame) & (frame.signal>PHOTOMETRIC_FLOOR))
    paint=(data>=density_limit)&(spectral<=stretch)&allow
    min_area=max(2,ceil(low_area/ctx.scale.pixel_area))
    max_area=max(min_area,ceil(high_area/ctx.scale.pixel_area))
    supported,regions,sizes,accept=_clean_blobs(paint,min_area,max_area)
    rows=[]
    marks=[]
    bounds=find_objects(regions)
    total_visible=supported.sum()
    image=mask_overlay(frame.rgb,[(supported,(240,183,91),.47),(paint&~supported,(140,161,215),.43)],frame.present)
    for token,block in enumerate(bounds,1):
        if block is None or not accept[token]:
            continue
        local=regions[block]==token
        yy,xx=np.where(local)
        h,w=allow.shape
        center=[rounded((block[1].start+xx.mean()+.5)/w,7),
                rounded((block[0].start+yy.mean()+.5)/h,7)]
        spread=percentiles(data[block],local,(15,50,85),3)
        local_area=sizes[token]*ctx.scale.pixel_area
        row=dict(patch=len(rows)+1,center=center,area_mm2=rounded(local_area),absorption=spread,
                 channel_difference=percentiles(spectral[block],local,(50,90),3))
        rows.append(row)
        if len(rows)<=100:
            marks.append(([center],(248,229,136),str(len(rows))))
    x0,y0,x1,y1=original.public()
    marks.append(([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)],(159,224,202),"Clear glass"))
    summary=dict(reference=ref.public(),candidate_fragments=len(rows),
                 candidate_area_mm2=rounded(total_visible*ctx.scale.pixel_area),observed_fraction=ratio(allow,ctx.selection))
    return Result("paint","Paint absorption estimate",summary,
                  {**summary,"reference_box":original.public(),"candidates":rows[:100],
                   "absorption_limit":density_limit,"channel_agreement":stretch,"area_range_mm2":[low_area,high_area],
                   "note":"This experiment suggests approximately neutral absorbing marks. Lead and dark translucent glass may share that response."},
                  (png("pigment-candidates",mark_image(image,marks),ctx),
                   map_asset("estimated-absorption",data,allow,ctx,low=-.2,high=max(.7,2*density_limit),palette="ink"),
                   csv_table("pigment-patches",("patch","area_mm2","absorption_p15","median","p85","channel_difference_median"),
                             [(row["patch"],row["area_mm2"],*row["absorption"],row["channel_difference"][0]) for row in rows])))


def _sweep_stats(context,field,selection,probe_axis, slices, coverage):
    proj=field if probe_axis=="horizontal" else field.T
    chosen=selection if probe_axis=="horizontal" else selection.T
    selected_view=context.selection if probe_axis=="horizontal" else context.selection.T
    along=np.linspace(0,proj.shape[1],slices+1).round().astype(int)
    amplitude=context.scale.width_mm if probe_axis=="horizontal" else context.scale.height_mm
    rows=[]
    for i in range(slices):
        start,stop=along[i:i+2]
        required=selected_view[:,start:stop]
        seen=chosen[:,start:stop]&required
        measured_count=seen.sum()
        required_count=required.sum()
        fractional=measured_count/max(1,required_count)
        measured=percentiles(proj[:,start:stop],seen,(10,50,90),10) if fractional>=coverage else [None]*3
        center=(start+stop)/2
        rows.append(dict(at_mm=rounded(center*amplitude/proj.shape[1]),from_px=int(start),to_px=int(stop),
                         observations=int(measured_count),support=rounded(fractional),absorption=measured))
    return rows


def absorption_strip(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"reference","axis","bins","min_support")
    reference=Box.parse(opts.get("reference",[.36,.13,.58,.28]),"Clear reference")
    paint=PaintBackground.from_frame(frame,ctx,reference)
    axis=option(opts.get("axis","horizontal"),"Axis",("horizontal","vertical"))
    bins=integer(opts.get("bins",28),"Number of strips",5,160)
    coverage=scalar(opts.get("min_support",.38),"Minimum strip coverage",.05,.98)
    scene,_=_density_plane(frame.rgb,paint)
    valid=ctx.demand(ctx.measured(frame))
    rows=_sweep_stats(ctx,scene,valid,axis,bins,coverage)
    labels=[("P10",(98,145,171),0),("Median",(175,83,77),1),("P90",(111,125,86),2)]
    graphic=chart("absorption-sweep",[r["at_mm"] for r in rows],
            [(key,color,[r["absorption"][index] for r in rows]) for key,color,index in labels],
            "Relative optical absorption by strip","Density")
    out=csv_table("strip-readings",("position_mm","pixels","support","p10","median","p90"),
                  [(row["at_mm"],row["observations"],row["support"],*row["absorption"]) for row in rows])
    summary=dict(total_strips=bins,reported_strips=sum(item["absorption"][1] is not None for item in rows),
                 direction=axis,reference=paint.public())
    return Result("absorption-strip","Strip absorption profile",summary,
                  {**summary,"profile":rows,"minimum_strip_coverage":coverage},
                  (graphic,out,map_asset("absorbing-field",scene,valid,ctx,low=-.25,high=1.05,palette="ink")))


def _residual(rgb,weights,min_signal):
    gain=np.maximum(rgb.sum(axis=-1,keepdims=True),min_signal)
    proportion=rgb/gain
    difference=np.linalg.norm(proportion-weights,axis=-1)
    return difference


def _reference_contrast(channels,edge,positive):
    value=rgb_to_lab(np.maximum(0,np.minimum(1,channels)))
    median=np.median(value[edge],axis=0)
    delta=np.linalg.norm(value-median,axis=-1)
    luma=value[...,0]-median[0]
    critical=(luma>0) if positive else (luma<0)
    return delta,luma,critical


def stain_survey(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"color_reference","chroma_difference","radius_mm","minimum_luminance","dark_only")
    ref=Box.parse(opts.get("color_reference",[.43,.4,.54,.5]),"Reference glass")
    mm=scalar(opts.get("radius_mm",3.2),"Local stain radius (mm)",.3,35)
    limits=scalar(opts.get("chroma_difference",8),"Lab display threshold",1,50)
    floor=scalar(opts.get("minimum_luminance",.06),"Dark signal cutoff",.005,.4)
    active=ctx.measured(frame)&(frame.signal>floor)
    sample=frame.measurable&ref.mask(ctx.scale.width_px,ctx.scale.height_px)&(frame.signal>floor)
    if sample.sum()<45:
        raise Problem("Stain comparison needs at least 45 reference pixels above the dark threshold.",422)
    valid=ctx.demand(active)
    radius=max(.6,mm/ctx.scale.step_y),max(.6,mm/ctx.scale.step_x)
    smooth,local=masked_blur(frame.rgb,frame.measurable,radius,floor=.60)
    delta,nature,majority=_reference_contrast(smooth,sample,False)
    trustworthy=local&valid
    differs = trustworthy&(delta>limits)
    if opts.get("dark_only",False):
        differs &= majority
    amb= trustworthy & (delta>limits*.7)&~differs
    ref_vec=np.median(smooth[sample],axis=0)
    ref_vec=ref_vec/max(.001,ref_vec.sum())
    hue=_residual(smooth,ref_vec,floor)
    summary=dict(color_departure=percentiles(delta,trustworthy),darker_area_mm2=rounded((differs&majority).sum()*ctx.scale.pixel_area),
                 brighter_area_mm2=rounded((differs&~majority).sum()*ctx.scale.pixel_area),
                 usable_fraction=ratio(trustworthy,ctx.selection),selected_reference_pixels=int(sample.sum()))
    artwork=mask_overlay(frame.rgb,[(differs&majority,(186,125,231),.55),(differs&~majority,(244,201,99),.61),
                                     (amb,(103,156,200),.42)],frame.present)
    return Result("stains","Local tonal stain check",summary,
                  {**summary,"color_reference":ref.public(),"threshold":limits,"radius_mm":mm,
                   "normalized_color_difference":percentiles(hue,trustworthy),
                   "note":"Color shifts and dimming are separate clues, and do not distinguish damage from normal glazing pigments."},
                  (png("stain-candidates",artwork,ctx),
                   map_asset("lab-delta",delta,trustworthy,ctx,low=0,high=2*limits,palette="thermal"),
                   map_asset("luma-shift",nature,trustworthy,ctx,low=-18,high=18,palette="diverging")))
