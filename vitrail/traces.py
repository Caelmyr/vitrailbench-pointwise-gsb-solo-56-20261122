from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .bench import Result, masked_blur, ratio
from .contracts import Problem, integer, keys, option, pair, rounded, scalar
from .picture import chart, csv_table, mark_image, mask_overlay, png


@dataclass(frozen=True)
class Traversal:
    center: np.ndarray
    along_mm: np.ndarray
    normals_unit: np.ndarray
    positions: np.ndarray
    angles: np.ndarray

    @classmethod
    def construct(cls,vertices,ctx,step):
        scale=np.array([ctx.scale.width_mm,ctx.scale.height_mm])
        line=np.asarray(vertices)*scale
        vectors=np.diff(line,axis=0)
        distances=np.linalg.norm(vectors,axis=1)
        if np.any(distances<.4):
            raise Problem("Each path segment needs at least 0.4 mm.")
        length=sum(distances)
        n=min(850,max(5,round(length/step)+1))
        marks=np.concatenate(([0],np.cumsum(distances)))
        edges=np.linspace(0,marks[-1],n)
        spans=np.searchsorted(marks[1:],edges,side="left")
        gaps=(edges-marks[spans])/distances[spans]
        locations=line[spans]+gaps[:,None]*vectors[spans]
        heading=vectors[spans]/distances[spans,None]
        normal=np.stack((-heading[:,1],heading[:,0]),axis=1)
        angles=np.degrees(np.arctan2(heading[:,1],heading[:,0]))%360
        return cls(locations,edges,normal,locations/scale,angles)

    def across(self,half,detector_step,scale):
        n=min(125,2*max(2,round(half/detector_step))+1)
        positions=np.linspace(-half,half,n)
        mm=self.center[:,None,:]+self.normals_unit[:,None,:]*positions[None,:,None]
        return positions,mm/np.array([scale.width_mm,scale.height_mm])


def _trace_field(intensity,valid,ctx,neighborhood,mode):
    sigma=(max(.8,neighborhood/ctx.scale.step_y),max(.8,neighborhood/ctx.scale.step_x))
    smoothed,enough=masked_blur(intensity,valid,sigma,floor=.57)
    projection=(smoothed-intensity) if mode=="dark" else (intensity-smoothed)
    return projection,valid&enough


def _sample_strip(field,valid,locations,scale):
    coordinates=np.stack((locations[...,1]*scale.height_px-.5,locations[...,0]*scale.width_px-.5))
    votes=map_coordinates(valid.astype(np.float32),coordinates,order=1,mode="constant",cval=0)
    data=map_coordinates(field,coordinates,order=1,mode="constant",cval=0)
    return data,votes>=.985


def _choices(signal,visible,position,min_strength):
    near=np.flatnonzero((signal[1:-1]>=signal[:-2])&(signal[1:-1]>signal[2:])&
                        (signal[1:-1]>=min_strength)&visible[:-2]&visible[1:-1]&visible[2:])+1
    rows=[]
    for i in near[np.argsort(-signal[near])[:5]]:
        strength=float(signal[i])
        p=float(position[i])
        uncertainty=float(position[i+1]-position[i-1])/2
        rows.append({"cross_mm":rounded(p),"response":rounded(strength),"sampling_step_mm":rounded(uncertainty)})
    return rows


def line_impression(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"path","response","half_width_mm","baseline_mm","spacing_mm","cutoff")
    raw=opts.get("path",[[.13,.48],[.66,.52],[.84,.46]])
    if not isinstance(raw,list) or not 2<=len(raw)<=72:
        raise Problem("A line impression requires two to seventy-two ordered vertices.")
    path=[pair(point,"Trace point") for point in raw]
    direction=option(opts.get("response","dark"),"Contrast direction",("dark","bright"))
    width=scalar(opts.get("half_width_mm",5),"Trace corridor half-width (mm)",.8,30)
    spacing=scalar(opts.get("spacing_mm",1.2),"Distance between cross sections (mm)",.25,18)
    baseline=scalar(opts.get("baseline_mm",1.4),"Local background radius (mm)",.4,15)
    threshold=scalar(opts.get("cutoff",.028),"Minimum response",.001,.55)
    scale=ctx.scale
    traversal=Traversal.construct(path,ctx,spacing)
    offsets,coords=traversal.across(width,min(scale.step_x,scale.step_y),scale)
    field,allowed=_trace_field(frame.signal,frame.measurable,ctx,baseline,direction)
    signals,good=_sample_strip(field,allowed&ctx.selection,coords,scale)
    marks=[(path,(155,223,193),"Drawn trace")]
    rows=[]
    for i,(points,visible) in enumerate(zip(signals,good,strict=True)):
        accepted=_choices(points,visible,offsets,threshold)
        axes=[rounded(v,7) for v in traversal.positions[i]]
        row=dict(index=i+1,distance_mm=rounded(traversal.along_mm[i]),panel=axes,camera_present_fraction=rounded(np.mean(visible)),
                 tangent_deg=rounded(traversal.angles[i]),proposals=accepted)
        if accepted:
            x=accepted[0]["cross_mm"]
            center=traversal.center[i]
            normal=traversal.normals_unit[i]
            new=(center+normal*x)/np.array([scale.width_mm,scale.height_mm])
            row["strongest_panel_position"]=[rounded(a,7) for a in new]
            color=(247,210,132) if len(accepted)==1 else (220,143,151)
            if i%3==0:
                marks.append(([row["panel"],row["strongest_panel_position"]],color,""))
        else:
            row["strongest_panel_position"]=None
        rows.append(row)
    art=mark_image(mask_overlay(frame.rgb,[],frame.present),marks)
    summary=dict(sections=len(rows),witnessed=sum(bool(row["proposals"]) for row in rows),
                 multiply_supported=sum(len(row["proposals"])>1 for row in rows),
                 span_mm=rounded(traversal.along_mm[-1]),half_width_mm=width)
    table=csv_table("local-trace-readings",("section","distance_mm","support","peak_offset_mm","peak_strength"),
                    [(row["index"],row["distance_mm"],row["camera_present_fraction"],
                      row["proposals"][0]["cross_mm"] if row["proposals"] else None,
                      row["proposals"][0]["response"] if row["proposals"] else None) for row in rows])
    overlay=png("marked-trace",art,ctx,crop=False)
    field_plot=chart("trace-offset",[row["distance_mm"] for row in rows],
                [("Strongest response",(128,106,175),[row["proposals"][0]["cross_mm"] if row["proposals"] else None for row in rows])],
                "Strongest local departure from drawn trace","Offset mm")
    return Result("trace","Drawn line impression",summary,{**summary,"path":path,"rows":rows,"pulse":direction,
                  "note":"Each reading is independent. Similar responses beside each other may belong to different marks."},
                  (overlay,field_plot,table))


def _profiles(first,second,valid_a,valid_b,depth,threshold):
    a=np.sum(first*valid_a,axis=1)/np.maximum(1,np.sum(valid_a,axis=1))
    b=np.sum(second*valid_b,axis=1)/np.maximum(1,np.sum(valid_b,axis=1))
    portion=(valid_a&valid_b).mean(axis=1)
    share=portion>=threshold
    return a,b,share,portion


def strip_pair(ctx):
    first,second=ctx.two()
    opts=ctx.order.toolbox
    keys(opts,"path","half_width_mm","spacing_mm","support")
    raw=opts.get("path",[[.20,.42],[.79,.56]])
    if not isinstance(raw,list) or not 2<=len(raw)<=60:
        raise Problem("Choose up to sixty vertices for a comparison trace.")
    points=[pair(point,"Trace point") for point in raw]
    span=scalar(opts.get("half_width_mm",3.5),"Comparison corridor half-width (mm)",.4,24)
    spacing=scalar(opts.get("spacing_mm",1.1),"Comparison sample spacing (mm)",.2,20)
    minimum=scalar(opts.get("support",.65),"Minimum shared section coverage",.23,.97)
    coords=Traversal.construct(points,ctx,spacing)
    offsets,locations=coords.across(span,min(ctx.scale.step_x,ctx.scale.step_y),ctx.scale)
    a,a_valid=_sample_strip(first.signal,first.measurable&ctx.selection,locations,ctx.scale)
    b,b_valid=_sample_strip(second.signal,second.measurable&ctx.selection,locations,ctx.scale)
    first_average,second_average,check,area=_profiles(a,b,a_valid,b_valid,span,minimum)
    if not np.any(check):
        raise Problem("The marked strip has no jointly photographed sections.",422)
    x=coords.along_mm
    rows=[]
    for i,distance in enumerate(x):
        part=dict(at_mm=rounded(distance),jointly_recorded=rounded(area[i]),measured=bool(check[i]),
                  first=rounded(first_average[i]) if check[i] else None,second=rounded(second_average[i]) if check[i] else None,
                  center=[rounded(v,7) for v in coords.positions[i]])
        part["gain_ev"]=rounded(np.log2(max(.001,second_average[i])/max(.001,first_average[i]))) if check[i] else None
        rows.append(part)
    data=chart("strip-means",x,[(name,ink,[part[key] for part in rows]) for name,key,ink in
                     (("First","first",(150,96,59)),("Second","second",(86,131,175)))],
               "Signal along paired window strip","Linear")
    artwork=mark_image(mask_overlay(first.rgb,[],first.present),[(points,(243,209,131),"Comparison")])
    summary=dict(compared_sections=int(check.sum()),total_sections=len(rows),half_width_mm=span,
                 typical_ev=rounded(np.median([row["gain_ev"] for row in rows if row["measured"]])))
    table=csv_table("strip-comparison",("distance_mm","support","first_signal","second_signal","shift_ev"),
                    [(row["at_mm"],row["jointly_recorded"],row["first"],row["second"],row["gain_ev"]) for row in rows])
    return Result("strip-pair","Registered strip comparison",summary,
                  {**summary,"readings":rows,"path":points,"support_floor":minimum},
                  (data,png("marked-strip",artwork,ctx,crop=False),table))
