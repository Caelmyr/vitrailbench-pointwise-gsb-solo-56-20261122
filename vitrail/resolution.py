from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter

from .bench import Result, masked_blur, percentiles, ratio
from .contracts import Box, Problem, integer, keys, pair, rounded, scalar
from .picture import chart, csv_table, map_asset, mark_image, mask_overlay, png


@dataclass(frozen=True)
class Slant:
    origin: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    length: float

    @classmethod
    def from_marks(cls,a,b,scale):
        sizes=np.asarray([scale.width_mm,scale.height_mm])
        points=np.asarray([a,b])*sizes
        delta=points[1]-points[0]
        length=float(np.linalg.norm(delta))
        if length<5:
            raise Problem("The slanted edge must span at least 5 mm.")
        parallel=delta/length
        normal=np.array([-parallel[1],parallel[0]])
        return cls(points[0],parallel,normal,length)

    def project(self,positions):
        delta=positions-self.origin
        return delta@self.tangent,delta@self.normal

    def corners(self,extent,width,height):
        upper=self.origin+extent*self.normal
        lower=self.origin-extent*self.normal
        across=self.tangent*self.length
        loop=np.stack((upper,upper+across,lower+across,lower,upper))
        return (loop/np.array([width,height])).tolist()


def _bins(geometry,scale,values,valid,width,pitch,n):
    grid=scale.grid()*[scale.width_mm,scale.height_mm]
    along,outside=geometry.project(grid)
    bounded=(along>=0)&(along<=geometry.length)&(np.abs(outside)<width)&valid
    indexes=np.clip(np.floor((outside[bounded]+width)/pitch).astype(int),0,n-1)
    exposure=values[bounded]
    count=np.bincount(indexes,minlength=n)
    total=np.bincount(indexes,weights=exposure,minlength=n)
    samples=np.divide(total,count,out=np.zeros_like(total),where=count>0)
    confirmed=count>=4
    return samples,count,confirmed,bounded,outside


def _pulse(center,real,step):
    known=np.flatnonzero(real)
    if len(known)<7 or len(known)<round(.55*len(real)):
        raise Problem("Insufficient measured bins in the edge spread. Try a wider or longer edge.",422)
    use=np.arange(len(center))
    inferred=np.interp(use,use[real],center[real])
    k=max(2,round(len(real)/8))
    difference=float(np.median(inferred[-k:])-np.median(inferred[:k]))
    if abs(difference)<.025:
        raise Problem("The marked edge has too little bright-to-dark range.",422)
    blurred=gaussian_filter(inferred,sigma=1,mode="nearest")
    curve=(blurred-np.median(blurred[:k]))/difference
    spread=np.diff(curve)/step
    spectrum=np.abs(np.fft.rfft(spread*np.hanning(len(spread))))
    spectrum /= max(spectrum[0],.000001)
    frequencies=np.fft.rfftfreq(len(spread),step)
    first_zero=np.flatnonzero(spectrum<.5)
    mtf50=None
    if len(first_zero):
        index=int(first_zero[0])
        if index>0:
            x0,x1=frequencies[index-1:index+1]
            y0,y1=spectrum[index-1:index+1]
            if y0>y1:
                mtf50=x0+(y0-.5)/(y0-y1)*(x1-x0)
    return curve,spread,frequencies,spectrum,mtf50,difference


def edge_resolution(ctx):
    opts=ctx.order.toolbox
    frame=ctx.one()
    keys(opts,"from","to","strip_halfwidth_mm","subsamples")
    a=pair(opts.get("from",[.42,.30]),"Edge start")
    b=pair(opts.get("to",[.70,.33]),"Edge end")
    width=scalar(opts.get("strip_halfwidth_mm",3),"Sampling half-width (mm)",.6,18)
    sub=integer(opts.get("subsamples",2),"Bins per working pixel",1,4)
    base=max(ctx.scale.step_x,ctx.scale.step_y)
    count=min(160,int(2*width/base*sub))
    if count<12:
        raise Problem("The selected strip lacks enough native support for an edge curve.")
    geometry=Slant.from_marks(a,b,ctx.scale)
    pitch=2*width/count
    values,tallies,support,footprint,normal_mm=_bins(geometry,ctx.scale,frame.signal,ctx.measured(frame),width,pitch,count)
    curve,spread,freq,transfer,mid,difference=_pulse(values,support,pitch)
    positions=-width+pitch*(np.arange(count)+.5)
    readout=[dict(at_mm=rounded(at),measured_signal=rounded(raw) if good else None,
                  fitted_edge=rounded(fitted),observations=int(n))
             for at,raw,fitted,n,good in zip(positions,values,curve,tallies,support,strict=True)]
    frequencies=[dict(cycles_per_mm=rounded(wave),response=rounded(val)) for wave,val in zip(freq,transfer,strict=True)]
    image=mask_overlay(frame.rgb,[(footprint,(238,199,112),.32)],frame.present)
    lines=[(geometry.corners(width,ctx.scale.width_mm,ctx.scale.height_mm),(251,218,112),"Edge strip"),
           ([a,b],(114,237,214),"Edge")]
    details=dict(edge_length_mm=rounded(geometry.length),contrast=rounded(difference),
                 observed_bins=int(support.sum()),total_bins=len(support),
                 approximate_mtf50_cycles_mm=rounded(mid) if mid is not None else None)
    return Result("resolution","Slanted-edge capture resolution",details,
                  {**details,"edge_profile":readout,"response_curve":frequencies,
                   "note":"The width and orientation of the marked edge influence this optical estimate. Fitted bins are displayed separately."},
                  (png("edge-strip",mark_image(image,lines),ctx,crop=False),
                   chart("relative-response",freq,[("Response",(155,84,48),transfer)],
                         "Estimated modulation transfer","Response",x_caption="Cycles per mm"),
                   chart("edge-profile",positions,[("Observed",(94,160,129),[row["measured_signal"] for row in readout]),
                         ("Fitted",(177,108,69),curve)],"Edge spread through the marked strip","Level"),
                   csv_table("edge-data",("normal_distance_mm","observations","raw_signal","fitted_edge"),
                             [(row["at_mm"],row["observations"],row["measured_signal"],row["fitted_edge"]) for row in readout])))


def _surface_focus(frame,ctx,scale_mm):
    image=frame.signal
    available=frame.measurable
    widths=(max(.5,scale_mm/ctx.scale.step_y),max(.5,scale_mm/ctx.scale.step_x))
    low,local=masked_blur(image,available,widths,floor=.75)
    mid,adjacent=masked_blur(image,available,(widths[0]/3,widths[1]/3),floor=.75)
    sharpened=mid-low
    feed=local&adjacent&available
    filtered,dense=masked_blur(sharpened**2,feed,(widths[0]*2,widths[1]*2),floor=.65)
    return np.sqrt(filtered),feed&dense,sharpened


def _focus_windows(signal,mask,nx,ny,scale):
    h,w=mask.shape
    xs=np.rint(np.linspace(0,w,nx+1)).astype(int)
    ys=np.rint(np.linspace(0,h,ny+1)).astype(int)
    results=[]
    for row in range(ny):
        for col in range(nx):
            x0,x1=xs[col:col+2]
            y0,y1=ys[row:row+2]
            zone=mask[y0:y1,x0:x1]
            sample=signal[y0:y1,x0:x1]
            v=sample[zone]
            item=dict(id=f"{chr(65+col)}{row+1}",bounds=[int(x0),int(y0),int(x1),int(y1)],
                      median=rounded(np.median(v)) if len(v)>=20 else None,
                      high=rounded(np.percentile(v,85)) if len(v)>=20 else None,
                      measured=int(len(v)),coverage=rounded(len(v)/max(1,zone.size)),
                      area_mm2=rounded(zone.size*scale.pixel_area))
            results.append(item)
    return results


def focus_map(ctx):
    opts=ctx.order.toolbox
    frame=ctx.one()
    keys(opts,"detail_mm","columns","rows","reference")
    spacing=scalar(opts.get("detail_mm",1.2),"Detail scale in mm",.2,14)
    nx=integer(opts.get("columns",5),"Focus map columns",2,12)
    ny=integer(opts.get("rows",4),"Focus map rows",2,12)
    reference=Box.parse(opts.get("reference",[.25,.2,.75,.8]),"Focus reference")
    contrast,valid,detail=_surface_focus(frame,ctx,spacing)
    allowed=ctx.demand(valid&ctx.selection)
    comparison=ctx.demand(valid&reference.mask(ctx.scale.width_px,ctx.scale.height_px),floor=0,pixels=20)
    threshold=max(float(np.percentile(contrast[comparison],40)),.00001)
    values=10*np.log10(np.maximum(contrast,.00001)/threshold)
    metrics=_focus_windows(values,allowed,nx,ny,ctx.scale)
    rendered=[]
    for row in metrics:
        x0,y0,x1,y1=row["bounds"]
        w,h=ctx.scale.width_px,ctx.scale.height_px
        color=(239,198,112) if row["median"] is not None and row["median"]>=0 else (183,158,216)
        rendered.append(([(x0/w,y0/h),(x1/w,y0/h),(x1/w,y1/h),(x0/w,y1/h),(x0/w,y0/h)],color,row["id"]))
    info=dict(reference_detail_energy=rounded(threshold),measured_fraction=ratio(allowed,ctx.selection),
              responses_db=percentiles(values,allowed),windows=len(metrics))
    result=mask_overlay(frame.rgb,[(allowed&(values<-4),(150,116,213),.65)],frame.present)
    return Result("focus","Local detail consistency",info,
                  {**info,"cells":metrics,"scale_mm":spacing,
                   "note":"Detail differences can also come from the glass pattern itself. This map is not a lens calibration."},
                  (map_asset("relative-detail-db",values,allowed,ctx,low=-12,high=10,palette="diverging"),
                   png("camera-detail-windows",mark_image(result,rendered),ctx),
                   csv_table("detail-cells",("cell","pixels","coverage","median_db","upper_85_db"),
                             [(row["id"],row["measured"],row["coverage"],row["median"],row["high"]) for row in metrics])))
