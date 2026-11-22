from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .bench import Result, percentiles
from .camera import apply_matrix
from .contracts import Problem, keys, label, object_data, pair, rounded, scalar
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


def _principal_axes(jacobian):
    tensor = np.asarray(jacobian)
    cross = np.linalg.det(tensor)
    singular = np.linalg.svd(tensor, compute_uv=False)
    longer = singular[..., 0]
    shorter = singular[..., 1]
    return longer, shorter, cross


def _jacobian(matrix, points):
    u = np.asarray(points)[...,0]
    v = np.asarray(points)[...,1]
    den = matrix[2,0]*u + matrix[2,1]*v + matrix[2,2]
    a = matrix[0,0]*u + matrix[0,1]*v + matrix[0,2]
    b = matrix[1,0]*u + matrix[1,1]*v + matrix[1,2]
    safe = np.maximum(den*den,1e-12)
    return np.stack((np.stack(((matrix[0,0]*den-a*matrix[2,0])/safe,(matrix[0,1]*den-a*matrix[2,1])/safe),axis=-1),
                     np.stack(((matrix[1,0]*den-b*matrix[2,0])/safe,(matrix[1,1]*den-b*matrix[2,1])/safe),axis=-1)),axis=-2)


def _mapped_tangent(frame, grid, pts):
    image = frame.record["image"]
    dx = np.array([image["width"],image["height"]])
    dy = np.array([1/grid.width_mm,1/grid.height_mm])
    return _jacobian(frame.alignment.matrix,pts) * dx[..., None]*dy


def _matrix_grid(frame,ctx):
    scale=ctx.scale
    points=scale.grid()
    w,h=scale.width_px,scale.height_px
    jac=_mapped_tangent(frame,scale,points)
    longer,shorter,cross=_principal_axes(jac)
    ratio=longer/np.maximum(shorter,1e-8)
    photo=frame.alignment.to_camera(points)
    center=photo[...,0]*(frame.record["image"]["width"]-1)
    return longer,shorter,ratio,cross,center


def _corners_rows(coords,frame,ctx):
    positions=frame.alignment.to_camera(coords)
    physical=ctx.scale
    lens_register=frame.record["image"]
    width=lens_register["width"]
    height=lens_register["height"]
    rows=[]
    for source,dest,label_text in zip(coords,positions,("upper left","upper right","lower right","lower left"),strict=True):
        rows.append({"corner":label_text,"panel_mm":[rounded(v) for v in physical.point_mm(source)],
                     "photo_px":[rounded(dest[0]*width-.5),rounded(dest[1]*height-.5)],
                     "inside_frame":bool(np.all((dest>=0)&(dest<=1)))})
    return rows


def calibration_grid(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"min_ppmm","max_distortion","spacing_mm")
    minimum=scalar(opts.get("min_ppmm",2.6),"Minimum capture sampling (px/mm)",.1,35)
    distortion=scalar(opts.get("max_distortion",1.35),"Maximum stretch ratio",1.02,8)
    step=scalar(opts.get("spacing_mm",18),"Calibration marks spacing (mm)",1,90)
    high,low,spread,cross,_=_matrix_grid(frame,ctx)
    safe=ctx.demand(ctx.measured(frame,allow_clipped=True) & (cross>0))
    dense=low>=minimum
    geometric=spread<=distortion
    unsupported=safe&(~dense|~geometric)
    both=safe&dense&geometric
    h,w=safe.shape
    grid=ctx.scale
    row_n=int(np.ceil(grid.height_mm/step))+1
    col_n=int(np.ceil(grid.width_mm/step))+1
    if row_n*col_n>2500:
        raise Problem("Use a wider interval for this physical scale grid.",413)
    # marks include all near-edge target lines even when the image has a missing boundary.
    lines=[]
    for i in range(col_n):
        x=min(1,(i*step)/grid.width_mm)
        lines.append(([(x,0),(x,1)],(241,217,140),str(round(x*grid.width_mm))))
    for j in range(row_n):
        y=min(1,(j*step)/grid.height_mm)
        lines.append(([(0,y),(1,y)],(196,226,205),str(round(y*grid.height_mm))))
    image=mask_overlay(frame.rgb,[(unsupported,(226,121,111),.53)],frame.present)
    rows=_corners_rows([[0,0],[1,0],[1,1],[0,1]],frame,ctx)
    summary=dict(supported_share=rounded(safe.sum()/max(1,ctx.selection.sum())),passes_both=rounded(both.sum()/max(1,safe.sum())),
                 lower_axis_sampling=percentiles(low,safe,(5,50,95)),upper_axis_sampling=percentiles(high,safe,(5,50,95)),
                 anisotropy=percentiles(spread,safe,(10,50,90)))
    return Result("calibration","Registered scale audit",summary,
                  {**summary,"corners":rows,"minimum_sampling":minimum,"maximum_stretch":distortion,
                   "note":"The lower principal-axis scale measures the weaker local camera sampling."},
                  (png("local-scale-grid",mark_image(image,lines),ctx,crop=False),
                   map_asset("sampling-px-mm",low,safe,ctx,low=0,high=max(2*minimum,10)),
                   map_asset("stretch-ratio",spread,safe,ctx,low=1,high=max(1.8,distortion*1.35),palette="thermal")))


@dataclass(frozen=True)
class RulerMark:
    title: str
    a: tuple
    b: tuple
    length: float

    @classmethod
    def parse(cls,value,i):
        row=object_data(value,"Ruler segment")
        keys(row,"name","from","to","measured_mm")
        name=label(row.get("name",f"Ruler mark {i}"),"Ruler label",65)
        a=pair(row.get("from"),"First mark")
        b=pair(row.get("to"),"Second mark")
        if np.linalg.norm(np.asarray(a)-b)<.01:
            raise Problem("A ruler segment must span at least one percent of the panel image.")
        return cls(name,a,b,scalar(row.get("measured_mm"),"Measured segment (mm)",.5,4000))

    def segment(self,scale):
        vector=(np.array(self.b)-self.a)*[scale.width_mm,scale.height_mm]
        return vector


def _reproject_lengths(frame,scale,mark):
    points=np.array([mark.a,mark.b])
    photo=frame.alignment.to_camera(points)
    valid=np.all((photo>=0)&(photo<=1))
    image=frame.record["image"]
    pixel=photo*np.array([image["width"],image["height"]])-.5
    return [[rounded(v) for v in pt] for pt in pixel],bool(valid)


def _scale_residuals(rows):
    mm_reported=np.array([item["known"] for item in rows])
    measured=np.array([item["observed_length"] for item in rows])
    compare=np.array([item["endpoints_present"] for item in rows])
    if compare.any():
        h=measured[compare]-mm_reported[compare]
        return rounded(np.sqrt(np.mean(h*h))),rounded(np.max(np.abs(h)))
    return None,None


def ruler_audit(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"rulers","tolerance_mm")
    tolerance=scalar(opts.get("tolerance_mm",1.5),"Allowed measurement difference (mm)",.05,80)
    raw=opts.get("rulers")
    if raw is None:
        raw=[{"name":"Horizontal","from":[.20,.26],"to":[.70,.26],"measured_mm":round(.50*ctx.scale.width_mm,4)},
             {"name":"Vertical","from":[.22,.20],"to":[.22,.75],"measured_mm":round(.55*ctx.scale.height_mm,4)}]
    if not isinstance(raw,list) or not 1<=len(raw)<=45:
        raise Problem("Provide one to forty-five measured ruler marks.")
    marks=[RulerMark.parse(raw_mark,i) for i,raw_mark in enumerate(raw,1)]
    measurements=[]
    shapes=[]
    for mark in marks:
        distance=np.linalg.norm(mark.segment(ctx.scale))
        cam,onsensor=_reproject_lengths(frame,ctx.scale,mark)
        ax,ay=ctx.scale.point_px(mark.a)
        bx,by=ctx.scale.point_px(mark.b)
        valid=onsensor and bool(frame.present[ay,ax] and frame.present[by,bx])
        delta=distance-mark.length
        summary=dict(name=mark.title,points=[mark.a,mark.b],photo_px=cam,known=mark.length,
                     observed_length=rounded(distance),difference_mm=rounded(delta),endpoints_present=valid,
                     status=("unseen" if not valid else ("within" if abs(delta)<=tolerance else "disagreement")))
        measurements.append(summary)
        ink=(127,216,173) if summary["status"]=="within" else (244,129,110)
        shapes.append(([mark.a,mark.b],ink,mark.title))
    rms,maximum=_scale_residuals(measurements)
    summary=dict(rms_mm=rms,largest_residual_mm=maximum,width_mm=ctx.scale.width_mm,height_mm=ctx.scale.height_mm,
                 check_count=len(measurements),observed=sum(row["endpoints_present"] for row in measurements))
    picture=mark_image(mask_overlay(frame.rgb,[],frame.present),shapes)
    table=csv_table("rulers",("name","physical_mm","registration_mm","difference_mm","status"),
                    [(row["name"],row["known"],row["observed_length"],row["difference_mm"],row["status"])
                     for row in measurements])
    return Result("rulers","Measured length cross-check",summary,{**summary,"rulers":measurements,"tolerance":tolerance},
                  (png("ruler-positions",picture,ctx,crop=False),table))


def _pixel_area_ratio(frame,ctx,points):
    matrix=frame.alignment.matrix
    jacobian=_jacobian(matrix,points)
    area=np.linalg.det(jacobian)
    width=frame.record["image"]["width"]/ctx.scale.width_mm
    height=frame.record["image"]["height"]/ctx.scale.height_mm
    return area*width*height


def camera_sampling_section(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"from","to","spacing_mm")
    start=np.asarray(pair(opts.get("from",[.15,.5]),"Start of section"))
    stop=np.asarray(pair(opts.get("to",[.85,.5]),"End of section"))
    spacing=scalar(opts.get("spacing_mm",1.2),"Profile spacing (mm)",.15,30)
    span=float(np.linalg.norm((stop-start)*[ctx.scale.width_mm,ctx.scale.height_mm]))
    if span<2*spacing:
        raise Problem("The camera sampling section is too short for the selected interval.")
    track=np.linspace(0,1,min(800,int(np.ceil(span/spacing))+1))
    coords=start[None,:]+(stop-start)[None,:]*track[:,None]
    pixels=np.array([ctx.scale.point_px(point) for point in coords])
    covered=frame.present[pixels[:,1],pixels[:,0]]
    jac=_mapped_tangent(frame,ctx.scale,coords)
    long_axis,short_axis,_=_principal_axes(jac)
    area=_pixel_area_ratio(frame,ctx,coords)
    observed=frame.alignment.to_camera(coords)
    rows=[]
    for i,fraction in enumerate(track):
        visible=covered[i]
        row=dict(distance_mm=rounded(fraction*span),observed=bool(visible),
                 major=rounded(long_axis[i]) if visible else None,minor=rounded(short_axis[i]) if visible else None,
                 camera_area_per_panel_mm2=rounded(area[i]) if visible else None,
                 panel_at=[rounded(v) for v in coords[i]],
                 camera_at=[rounded(v) for v in observed[i]] if visible else None)
        rows.append(row)
    legend=[("major",(162,81,46),[row["major"] for row in rows]),("minor",(55,111,146),[row["minor"] for row in rows])]
    graphic=chart("camera-sampling-section",[row["distance_mm"] for row in rows],legend,"Camera sampling along panel line","px/mm")
    read=csv_table("camera-sampling",("distance","observed","major_pxmm","minor_pxmm","area_px_mm2"),
                   [(row["distance_mm"],row["observed"],row["major"],row["minor"],row["camera_area_per_panel_mm2"]) for row in rows])
    summary=dict(length_mm=rounded(span),samples=len(track),usable=int(covered.sum()),
                 peak_stretch=rounded((long_axis[covered]/np.maximum(short_axis[covered],1e-7)).max()) if covered.any() else None)
    return Result("sampling-section","Camera sampling section",summary,{**summary,"samples":rows},(graphic,read))
