from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, find_objects, label
from PIL import ImageColor

from .bench import Result, percentiles, ratio
from .contracts import Problem, integer, keys, rounded, scalar
from .leads import CONNECTION, CCROSS, LeadGraph, _graph_options
from .picture import csv_table, mark_image, mask_overlay, png


PALETTE = ("#df9d5f","#72afaa","#efcb74","#b98dc6","#85a8d1","#baae77","#d38e89","#91bb78")


def _outline(mask):
    near = binary_dilation(mask, structure=CCROSS)
    return near & ~mask


def _physical_box(bounds, scale):
    left,top,right,bottom=bounds
    return [rounded(left*scale.step_x),rounded(top*scale.step_y),
            rounded(right*scale.step_x),rounded(bottom*scale.step_y)]


def _edge_support(mask, lead_mask, visible, cut):
    outline = _outline(mask)
    outline &= cut
    if outline.any():
        near_lead = binary_dilation(lead_mask,structure=CONNECTION)
        measured = outline & visible
        unavailable = outline & ~visible
        separation = near_lead & measured
        return rounded(separation.sum()/outline.sum()),rounded(unavailable.sum()/outline.sum())
    return None,None


def _perimeter(mask, scale):
    padded = np.pad(mask,1,constant_values=False)
    count_h = np.count_nonzero(np.diff(padded,axis=0))
    count_v = np.count_nonzero(np.diff(padded,axis=1))
    return count_h*scale.step_x+count_v*scale.step_y


@dataclass(frozen=True)
class PaneMap:
    labels: np.ndarray
    observed: np.ndarray
    lead: np.ndarray
    entries: tuple

    @classmethod
    def find(cls, ctx, thickness, minimum, graph_options):
        lead=LeadGraph.detect(ctx,*graph_options)
        lateral=max(1,round(thickness/max(ctx.scale.step_y,ctx.scale.step_x)))
        if thickness:
            border=binary_closing(lead.mask,structure=CONNECTION,iterations=min(lateral,6))
            border |= lead.mask
        else:
            border=lead.mask
        available=lead.visible & ~border
        # Four connected interior pixels cannot cross a one-pixel diagonal lead seam.
        coded, total=label(available,structure=CCROSS)
        boxes=find_objects(coded)
        size=np.bincount(coded.ravel(),minlength=total+1)
        small = minimum/ctx.scale.pixel_area
        indices = [i for i in range(1,total+1) if size[i]>=small]
        if len(indices)>240:
            raise Problem("Too many pane candidates. Use a closer region or larger minimum area.",422)
        indices.sort(key=lambda i: (boxes[i-1][0].start,boxes[i-1][1].start,i))
        translated=np.zeros(total+1,dtype=np.int32)
        h,w=coded.shape
        frame=ctx.one()
        luma=frame.signal
        heading=[]
        x0,y0,x1,y1=ctx.order.region.indices(w,h)
        for index,raw in enumerate(indices,1):
            box=boxes[raw-1]
            start_y,end_y=box[0].start,box[0].stop
            start_x,end_x=box[1].start,box[1].stop
            narrow=round(min(end_x-start_x,end_y-start_y))
            if narrow < 4:
                continue
            translated[raw]=index
            region=coded==raw
            cut=ctx.selection
            line_cover,missing=_edge_support(region,border,lead.visible,cut)
            yy,xx=np.nonzero(region)
            center=[rounded((float(xx.mean())+.5)/w,7),rounded((float(yy.mean())+.5)/h,7)]
            touches=bool(start_x<=x0 or start_y<=y0 or end_x>=x1 or end_y>=y1)
            shell=binary_fill_holes(region)
            holes=shell & ~region
            bbox=(start_x,start_y,end_x,end_y)
            heading.append(dict(pane=index,center=center,bounds=bbox,physical_bounds_mm=_physical_box(bbox,ctx.scale),
                                area_mm2=rounded(size[raw]*ctx.scale.pixel_area),
                                boundary_length_mm=rounded(_perimeter(region,ctx.scale)),
                                interior_holes_mm2=rounded(holes.sum()*ctx.scale.pixel_area),
                                lead_outline_fraction=line_cover,unknown_border_fraction=missing,touches_window_edge=touches,
                                signal=percentiles(luma,region,(10,50,90)),
                                enclosed=not touches and (missing or 0)<.08 and (line_cover or 0)>.70))
        return cls(translated[coded],lead.visible,border,tuple(heading))


def pane_catalog(ctx):
    opts=ctx.order.toolbox
    parameters=_graph_options(opts,"close_lead_gap_mm","min_pane_mm2")
    gap=scalar(opts.get("close_lead_gap_mm",.8),"Lead-gap preview closing",0,3)
    minimum=scalar(opts.get("min_pane_mm2",120),"Minimum pane area (mm²)",4,80000)
    panes=PaneMap.find(ctx,gap,minimum,parameters)
    picture=ctx.one().rgb
    layers=[]
    shapes=[]
    for item in panes.entries:
        token=item["pane"]
        mask=panes.labels==token
        tint=ImageColor.getrgb(PALETTE[(token-1)%len(PALETTE)])
        layers.append((mask,tint,.27))
        layers.append((_outline(mask),tint,.63))
        shapes.append(([item["center"]],tint,str(token)))
    marked=mark_image(mask_overlay(picture,layers,panes.observed),shapes)
    tab=csv_table("pane-catalog",("pane","area_mm2","boundary_mm","internal_holes_mm2","lead_support","unobserved_edge","enclosed"),
                  [(row["pane"],row["area_mm2"],row["boundary_length_mm"],row["interior_holes_mm2"],
                    row["lead_outline_fraction"],row["unknown_border_fraction"],row["enclosed"]) for row in panes.entries])
    summary=dict(panes=len(panes.entries),enclosed=sum(item["enclosed"] for item in panes.entries),
                 mapped_fraction=ratio(panes.labels>0,ctx.selection))
    return Result("panes","Window-pane catalog",summary,
                  {**summary,"candidates":list(panes.entries),"closing_mm":gap,
                   "note":"Open regions and edges with missing support remain in review. Area covers observed interior only."},
                  (png("pane-labels",marked,ctx),tab))


def _strip_coords(a,b,scale,count):
    at=np.linspace(0,1,count)
    p=np.asarray(a)
    term=np.asarray(b)-p
    along=p[None,:]+at[:,None]*term[None,:]
    physical=term*np.array([scale.width_mm,scale.height_mm])
    distance=np.linalg.norm(physical)
    if distance<2:
        raise Problem("Place a line at least 2 mm long.")
    perp=np.array([-physical[1],physical[0]])/distance
    return along,perp,distance


def _run_width(signal,valid,threshold,step):
    center=len(signal)//2
    if not valid[center] or signal[center]>=threshold:
        return None,"center off lead"
    x0=x1=center
    while x0>0 and valid[x0-1] and signal[x0-1]<threshold:
        x0-=1
    while x1<len(signal)-1 and valid[x1+1] and signal[x1+1]<threshold:
        x1+=1
    if x0==0 or x1==len(signal)-1:
        return None,"strip too narrow"
    if not valid[x0-1] or not valid[x1+1]:
        return None,"one edge missing"
    return rounded((x1-x0+1)*step),"measured"


def lead_gauge(ctx):
    from .contracts import pair
    frame=ctx.one()
    p=ctx.order.toolbox
    keys(p,"from","to","half_width_mm","darkness","crossings")
    a=pair(p.get("from",[.45,.2]),"Start")
    b=pair(p.get("to",[.45,.8]),"End")
    half=scalar(p.get("half_width_mm",5),"Gauge half width",1,28)
    depth=scalar(p.get("darkness",.07),"Lead brightness threshold",.005,.5)
    count=integer(p.get("crossings",17),"Gauge cross sections",5,81)
    dims=ctx.scale
    path,perp,length=_strip_coords(a,b,dims,count)
    step=max(.08,min(dims.step_x,dims.step_y))
    n=min(251,2*max(3,int(half/step))+1)
    step=half/(n//2)
    offsets=np.linspace(-half,half,n)
    direction=np.array([perp[0]/dims.width_mm,perp[1]/dims.height_mm])
    width=[]
    marks=[]
    valid_map=frame.measurable
    signal=frame.signal
    for i, center in enumerate(path):
        locations=center[None,:]+offsets[:,None]*direction
        bounds=((locations>=0)&(locations<1)).all(axis=1)
        x=np.clip((locations[:,0]*dims.width_px).astype(int),0,dims.width_px-1)
        y=np.clip((locations[:,1]*dims.height_px).astype(int),0,dims.height_px-1)
        ok=valid_map[y,x]&bounds
        measure,desc=_run_width(signal[y,x],ok,depth,step)
        width.append(dict(at_mm=rounded(i*length/(count-1)),location=[rounded(z,7) for z in center],width_mm=measure,
                          status=desc,coverage=rounded(ok.mean())))
        marks.append(([locations[0],locations[-1]],(240,215,123) if measure else (234,112,108),""))
    successful=[row["width_mm"] for row in width if row["width_mm"] is not None]
    report=dict(measured=len(successful),crossings=count,track_mm=rounded(length),
                typical_width_mm=rounded(np.median(successful)) if successful else None)
    picture=mark_image(mask_overlay(frame.rgb,[],frame.present),[(list(path),(106,210,187),"Gauge"),*marks])
    table=csv_table("gauge",("distance_mm","x","y","width_mm","status","support"),
                    [(row["at_mm"],*row["location"],row["width_mm"],row["status"],row["coverage"]) for row in width])
    return Result("gauge","Lead profile gauge",report,{**report,"readings":width,"half_width_mm":half},
                  (png("gauge-path",picture,ctx),table))
