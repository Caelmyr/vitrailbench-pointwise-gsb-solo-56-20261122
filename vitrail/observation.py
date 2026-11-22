from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, distance_transform_edt, find_objects, label

from .bench import Result, percentiles, ratio
from .camera import brightness, linear_to_srgb
from .contracts import Box, Problem, keys, label as clean_label, object_data, pair, rounded, scalar
from .leads import CONNECTION
from .photometry import rgb_to_lab
from .picture import csv_table, map_asset, mark_image, mask_overlay, png


def _pixel_hit(grid, point):
    x,y=grid.point_px(point)
    center=[(x+.5)/grid.width_px,(y+.5)/grid.height_px]
    return x,y,center


def _bounds_for_disk(scale,center,radius):
    lengths=np.asarray([scale.width_mm,scale.height_mm])
    return np.maximum(0,np.asarray(center)-radius/lengths),np.minimum(1,np.asarray(center)+radius/lengths)


def _disk(ctx,center,radius):
    w,h=ctx.scale.width_px,ctx.scale.height_px
    left,right=_bounds_for_disk(ctx.scale,center,radius)
    indices=Box(*left,*right).indices(w,h)
    x0,y0,x1,y1=indices
    xx=(np.arange(x0,x1)+.5)/w
    yy=(np.arange(y0,y1)+.5)/h
    x,y=np.meshgrid(ctx.scale.width_mm*(xx-center[0]),ctx.scale.height_mm*(yy-center[1]))
    selection=(x*x+y*y<=radius*radius)
    return indices,selection


def _source_site(ctx,frame,point,span):
    x,y,center=_pixel_hit(ctx.scale,point)
    observed=bool(frame.present[y,x])
    registration=frame.alignment
    mapped=registration.to_camera([center])[0]
    camera_width=frame.record["image"]["width"]
    camera_height=frame.record["image"]["height"]
    xcam=rounded(float(mapped[0])*camera_width-.5)
    ycam=rounded(float(mapped[1])*camera_height-.5)
    xm,ym=ctx.scale.point_mm(center)
    result=dict(capture=frame.record["id"],title=frame.record["data"]["title"],
                camera_location=[rounded(v,7) for v in mapped],
                camera_pixel=[xcam,ycam],photo_size=[camera_width,camera_height],panel_position=center,
                position_mm=[rounded(xm),rounded(ym)],has_pixel=observed,near_clip=bool(frame.clipped[y,x]),
                sensor_code=[rounded(v) for v in frame.code[y,x]] if observed else None,
                linear_rgb=[rounded(v) for v in frame.rgb[y,x]] if observed else None)
    if span>0:
        indices,mask=_disk(ctx,point,span)
        x0,y0,x1,y1=indices
        tile=frame.rgb[y0:y1,x0:x1]
        subset=frame.measurable[y0:y1,x0:x1]&mask
        measured=brightness(tile)
        result["neighborhood"]=dict(radius_mm=span,coverage=ratio(subset,mask),
                                    observations=int(subset.sum()),signal=percentiles(measured,subset,(10,50,90)))
    return result


def source_probe(ctx):
    settings=ctx.order.toolbox
    keys(settings,"point","radius_mm")
    point=pair(settings.get("point",[.5,.5]),"Panel point")
    radius=scalar(settings.get("radius_mm",2.5),"Local neighborhood (mm)",0,38)
    reads=[_source_site(ctx,frame,point,radius) for frame in ctx.frames]
    for site in reads:
        camera_site=site["camera_location"]
        site["last_alignment"]=next((part.alignment_ref["id"] if part.alignment_ref else None
                                     for part in ctx.frames if part.record["id"]==site["capture"]),None)
        site["inside_camera"]=all(value is not None and 0<=value<=1 for value in camera_site)
    summary=dict(selected_panel_point=list(point),captures=len(reads),observed_in=sum(item["has_pixel"] for item in reads),
                 clipped_in=sum(item["near_clip"] for item in reads))
    row=csv_table("sources",("photo","camera_x","camera_y","pixel_x","pixel_y","has_pixel","near_clip","red","green","blue"),
                  [(item["title"],*item["camera_location"],*item["camera_pixel"],item["has_pixel"],item["near_clip"],
                    *(item["linear_rgb"] or (None,None,None))) for item in reads])
    main=ctx.frames[0]
    graphic=mark_image(mask_overlay(main.rgb,[],main.present),[([point],(244,222,118),"Probe")])
    return Result("probe","Source pixel probe",summary,{**summary,"sources":reads},(png("probe",graphic,ctx,crop=False),row))


@dataclass(frozen=True)
class Swatch:
    name: str
    box: Box
    nominal: tuple | None

    @classmethod
    def parse(cls,value,i):
        data=object_data(value,"Color patch")
        keys(data,"name","box","nominal_srgb")
        ref=data.get("nominal_srgb")
        if ref is not None:
            if not isinstance(ref,list) or len(ref)!=3:
                raise Problem("Nominal RGB needs three sRGB decimal codes.")
            ref=tuple(scalar(code,"sRGB code",0,1) for code in ref)
        return cls(clean_label(data.get("name",f"Patch {i}"),"Name",50),
                   Box.parse(data.get("box"),"Color patch"),ref)

    def public(self):
        return dict(name=self.name,region=self.box.public(),nominal_srgb=list(self.nominal) if self.nominal else None)


def extract_swatches(value):
    raw=value
    if raw is None:
        raw=[{"name":"Top blue","box":[.18,.19,.29,.28]}, {"name":"Center amber","box":[.46,.36,.57,.46]},
             {"name":"Lower green","box":[.68,.67,.79,.77]}]
    if not isinstance(raw,list) or not 1<=len(raw)<=32:
        raise Problem("Use one to thirty-two named patches.")
    codes=[Swatch.parse(item,i) for i,item in enumerate(raw,1)]
    if len({item.name for item in codes})!=len(codes):
        raise Problem("Use different patch names.")
    return codes


def _patch_read(frame,patch,ctx):
    from .camera import srgb_to_linear
    bounds=patch.box.indices(ctx.scale.width_px,ctx.scale.height_px)
    x0,y0,x1,y1=bounds
    usable=frame.measurable[y0:y1,x0:x1]
    total=int(usable.size)
    data=frame.rgb[y0:y1,x0:x1][usable]
    n=data.shape[0]
    chroma=np.zeros(3,dtype=float)
    if n>=12:
        chroma=np.median(data,axis=0)
        spread=[rounded(x) for x in (np.percentile(data,90,axis=0)-np.percentile(data,10,axis=0))]
        rgb=linear_to_srgb(chroma)
        lab=rgb_to_lab(np.minimum(chroma,1))
        drift=rounded(np.linalg.norm(rgb_to_lab(srgb_to_linear(np.array(patch.nominal)))-lab)) if patch.nominal else None
        return dict(observed=n,total=total,coverage=rounded(n/total),median_rgb=[rounded(v) for v in chroma],
                    srgb=[int(v) for v in rgb],lab=[rounded(v) for v in lab],spread=spread,delta_to_nominal=drift,
                    clipped=int(np.sum(frame.clipped[y0:y1,x0:x1])))
    return dict(observed=n,total=total,coverage=rounded(n/total),median_rgb=None,srgb=None,lab=None,spread=None,
                delta_to_nominal=None,clipped=int(np.sum(frame.clipped[y0:y1,x0:x1])))


def _swatch_strip(reads):
    height=74
    width=550
    image=Image.new("RGB",(width,18+height*len(reads)),(246,241,228))
    draw=ImageDraw.Draw(image)
    for i,item in enumerate(reads):
        y=8+i*height
        draw.text((14,y+7),item["name"],fill=(41,53,54))
        results=item["captures"]
        room=min(90,(width-185)//max(1,len(results)))
        for j,record in enumerate(results):
            x=170+j*room
            color=record["srgb"]
            draw.rectangle((x,y+4,x+room-8,y+44),fill=tuple(color) if color else (91,94,103))
            draw.text((x,y+47),str(record["coverage"]),fill=(58,58,73))
    return png("swatches",image)


def patch_chart(ctx):
    opts=ctx.order.toolbox
    keys(opts,"patches")
    swatches=extract_swatches(opts.get("patches"))
    rows=[]
    tab=[]
    markups=[]
    for entry in swatches:
        measurement=[]
        first=None
        for frame in ctx.frames:
            result=_patch_read(frame,entry,ctx)
            if len(measurement)==0:
                first=result["lab"]
            diff=rounded(np.linalg.norm(np.array(result["lab"])-first)) if result["lab"] and first else None
            report={**result,"capture":frame.record["id"],"title":frame.record["data"]["title"],"delta_from_first":diff}
            measurement.append(report)
            tab.append((entry.name,report["title"],report["observed"],report["coverage"],
                        *(report["lab"] or (None,None,None)),report["delta_from_first"],report["delta_to_nominal"]))
        rows.append({**entry.public(),"captures":measurement})
        x0,y0,x1,y1=entry.box.public()
        markups.append(([(x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0)],(247,218,132),entry.name))
    main=ctx.frames[0]
    picture=mark_image(mask_overlay(main.rgb,[],main.present),markups)
    metrics=dict(patches=len(rows),captures=len(ctx.frames),patch_reads=sum(item["lab"] is not None
                        for row in rows for item in row["captures"]))
    table=csv_table("patches",("name","photo","observed_px","coverage","L","a","b","delta_to_first","delta_to_nominal"),tab)
    return Result("swatches","Named color patch readings",metrics,{**metrics,"patches":rows},
                  (png("patch-map",picture,ctx,crop=False),_swatch_strip(rows),table))


def _shape(label_map,region,token,grid):
    box_shape=label_map[region]
    local=box_shape==token
    ry,rx=np.where(local)
    y0=region[0].start
    x0=region[1].start
    area=int(local.sum())
    size=[region[1].stop-x0,region[0].stop-y0]
    centroid=[rounded((x0+rx.mean()+.5)/grid.width_px,7),rounded((y0+ry.mean()+.5)/grid.height_px,7)]
    return area,size,centroid


def photo_gaps(ctx):
    opts=ctx.order.toolbox
    keys(opts,"border_width_mm","min_area_mm2")
    frame=ctx.one()
    border=scalar(opts.get("border_width_mm",3.2),"Margin warning distance (mm)",.3,25)
    min_area=scalar(opts.get("min_area_mm2",4),"Show gaps larger than (mm²)",.05,1000)
    known=frame.present & ctx.selection
    gap=ctx.selection & ~known
    ids, total=label(gap,structure=CONNECTION)
    sets=find_objects(ids)
    distance=distance_transform_edt(known,sampling=(ctx.scale.step_y,ctx.scale.step_x))
    edge=known & (distance<=border)
    outline=binary_erosion(ctx.selection,structure=CONNECTION)
    details=[]
    marks=[]
    for index,region in enumerate(sets,1):
        if region is None:
            continue
        count,size,center=_shape(ids,region,index,ctx.scale)
        area=count*ctx.scale.pixel_area
        if area<min_area:
            continue
        connected=not bool(np.all(outline[region][ids[region]==index]))
        shape_region=dict(gap=len(details)+1,width_mm=rounded(size[0]*ctx.scale.step_x),
                          height_mm=rounded(size[1]*ctx.scale.step_y),area_mm2=rounded(area),
                          panel_centroid=center,open_at_region_boundary=connected)
        details.append(shape_region)
        if len(details)<=170:
            marks.append(([center],(246,194,103),str(len(details))))
    photo=mask_overlay(frame.rgb,[(gap,(222,109,116),.82),(edge,(236,177,92),.61)],frame.present)
    summary=dict(area_unseen_mm2=rounded(gap.sum()*ctx.scale.pixel_area),
                 fraction_unseen=ratio(gap,ctx.selection),area_of_risky_margin_mm2=rounded(edge.sum()*ctx.scale.pixel_area),
                 gap_components=total,reported_gaps=len(details))
    return Result("gaps","Source gaps and risky margins",summary,
                  {**summary,"margin_mm":border,"gaps":details[:170],
                   "note":"Distances are from the usable capture boundary in the registered panel plane."},
                  (png("gap-overview",mark_image(photo,marks),ctx),
                   map_asset("distance",distance,known,ctx,low=0,high=border*2,palette="field")))
