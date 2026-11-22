from dataclasses import dataclass
from math import atan2, degrees, hypot
import secrets

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion

from .bench import Result, percentiles, ratio
from .contracts import Problem, integer, keys, label, object_data, option, pair, rounded
from .leads import CONNECTION
from .picture import csv_table, mark_image, mask_overlay, png


TYPES = ("point", "line", "polygon")
ROLES = ("feature", "repair", "missing", "datum", "note")
COLORS = {"feature": (247, 218, 104), "repair": (120, 206, 233), "missing": (231, 111, 108),
          "datum": (173, 224, 162), "note": (208, 166, 231)}


def _orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])


def _on(a, b, p):
    return abs(_orient(a, b, p)) < 1e-10 and all(min(a[i], b[i])-1e-9 <= p[i] <= max(a[i], b[i])+1e-9
                                               for i in (0, 1))


def _cross(a, b, c, d):
    z = _orient(a,b,c),_orient(a,b,d),_orient(c,d,a),_orient(c,d,b)
    if z[0]*z[1] < -1e-16 and z[2]*z[3] < -1e-16:
        return True
    return _on(a,b,c) or _on(a,b,d) or _on(c,d,a) or _on(c,d,b)


def _rings(points):
    return list(zip(points, [*points[1:], points[0]], strict=True))


def _area(points):
    return .5*sum(a[0]*b[1]-b[0]*a[1] for a,b in _rings(points))


def _check_polygon(points):
    edges = _rings(points)
    count = len(edges)
    for i, (a,b) in enumerate(edges):
        for j, (c,d) in enumerate(edges[i+1:], start=i+1):
            if j==i+1 or (i==0 and j==count-1):
                continue
            if _cross(a,b,c,d):
                raise Problem("A polygon may not cross itself or touch at a nonadjacent edge.")
    if abs(_area(points)) < .000003:
        raise Problem("The marked polygon is too narrow to measure.")


def _test_points(points, kind):
    if len(set(points)) != len(points):
        raise Problem("Each mark must use different vertices.")
    if kind == "polygon":
        _check_polygon(points)
    if kind != "point" and any(hypot(b[0]-a[0], b[1]-a[1]) < .001 for a,b in zip(points,points[1:])):
        raise Problem("Two vertices are too close to distinguish.")


@dataclass(frozen=True)
class Mark:
    key: str
    kind: str
    role: str
    title: str
    points: tuple

    @classmethod
    def parse(cls, raw):
        data = object_data(raw, "Survey mark")
        keys(data, "key", "kind", "role", "title", "points")
        kind = option(data.get("kind"), "Mark kind", TYPES)
        role = option(data.get("role", "feature"), "Mark role", ROLES)
        vertices = data.get("points")
        minimum, maximum = {"point": (1,1), "line": (2,90), "polygon": (3,120)}[kind]
        if not isinstance(vertices, list) or not minimum <= len(vertices) <= maximum:
            raise Problem(f"Mark {kind} requires {minimum}–{maximum} vertices.")
        positions = tuple(pair(item,"Panel position") for item in vertices)
        _test_points(positions,kind)
        token = label(data.get("key",secrets.token_hex(6)),"Mark key",30)
        if not token.isalnum():
            raise Problem("Mark keys use letters and digits.")
        return cls(token,kind,role,label(data.get("title","Mark"),"Mark title",80),positions)

    def public(self):
        return dict(key=self.key,kind=self.kind,role=self.role,title=self.title,
                    points=[list(item) for item in self.points])

    def physical(self, width_mm, height_mm):
        spatial = [(x*width_mm,y*height_mm) for x,y in self.points]
        length = sum(hypot(b[0]-a[0],b[1]-a[1]) for a,b in
                     (_rings(spatial) if self.kind=="polygon" else zip(spatial,spatial[1:])))
        area = abs(_area(spatial)) if self.kind == "polygon" else None
        extent = [max(p[axis] for p in spatial)-min(p[axis] for p in spatial) for axis in (0,1)]
        straight = hypot(spatial[-1][0]-spatial[0][0],spatial[-1][1]-spatial[0][1])
        return dict(points_mm=[[rounded(x),rounded(y)] for x,y in spatial], lengths_mm=rounded(length),
                    width_mm=rounded(extent[0]),height_mm=rounded(extent[1]),area_mm2=rounded(area) if area else None,
                    end_separation_mm=rounded(straight))


def _polygon_image(shape, points):
    width, height = shape
    image = Image.new("1",shape,0)
    canvas = ImageDraw.Draw(image)
    clipped = [(.5+width*x,.5+height*y) for x,y in points]
    canvas.polygon(clipped,fill=1)
    return np.asarray(image,dtype=bool)


def _line_sample(scale, points):
    xs = []
    ys = []
    segment_distance = []
    for a,b in zip(points,points[1:]):
        ay,ax = a[1]*scale.height_px,a[0]*scale.width_px
        by,bx = b[1]*scale.height_px,b[0]*scale.width_px
        n = max(3,min(300,round(hypot(by-ay,bx-ax)*1.3)))
        values = np.linspace(0,1,n,endpoint=False)
        local_x = np.clip(np.floor(ax+values*(bx-ax)).astype(int),0,scale.width_px-1)
        local_y = np.clip(np.floor(ay+values*(by-ay)).astype(int),0,scale.height_px-1)
        xs.extend(local_x)
        ys.extend(local_y)
        physical_length = hypot((b[0]-a[0])*scale.width_mm,(b[1]-a[1])*scale.height_mm)
        segment_distance.extend([physical_length/n]*len(local_x))
    return np.asarray(xs),np.asarray(ys),np.asarray(segment_distance)


def _point_data(frame, ctx, point):
    col,row = ctx.scale.point_px(point)
    measured = bool(frame.measurable[row,col])
    camera = frame.alignment.to_camera([point])[0]
    return {"observed": bool(frame.present[row,col]),"measured": measured,
            "camera": [rounded(v,7) for v in camera],
            "transmitted_linear": rounded(frame.signal[row,col]) if measured else None,
            "clipped": bool(frame.clipped[row,col])}


def parse_survey(value):
    doc = object_data(value,"Survey")
    keys(doc,"name","marks","id","revision")
    name = label(doc.get("name"),"Survey title")
    items = doc.get("marks")
    if not isinstance(items,list) or not 1 <= len(items) <= 100:
        raise Problem("Save one to one hundred survey marks.")
    marks = [Mark.parse(item) for item in items]
    if len({mark.key for mark in marks}) != len(marks):
        raise Problem("Survey marks must have different identities.")
    return name,marks


def survey_measure(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts,"marks","tint_regions")
    _, marks = parse_survey({"name":"Preview","marks":opts.get("marks")})
    rows, overlays, areas = [],[],[]
    dims = ctx.scale
    full = frame.measurable
    for mark in marks:
        enriched = {**mark.public(),**mark.physical(dims.width_mm,dims.height_mm)}
        accent = COLORS[mark.role]
        if mark.kind == "polygon":
            mask = _polygon_image((dims.width_px,dims.height_px),mark.points)
            allowed = full & mask
            enriched["observed_fraction"] = ratio(allowed,mask)
            enriched["observed_area_mm2"] = rounded(allowed.sum()*dims.pixel_area)
            enriched["signal_percentiles"] = percentiles(frame.signal,allowed,[10,50,90])
            areas.append((mask,accent,.23))
            overlays.append(([*mark.points,mark.points[0]],accent,mark.title))
        elif mark.kind == "line":
            xs,ys,weights = _line_sample(dims,mark.points)
            measured = full[ys,xs]
            enriched["observed_fraction"] = rounded(weights[measured].sum()/max(weights.sum(),1e-8))
            enriched["observed_length_mm"] = rounded(weights[measured].sum())
            overlays.append((mark.points,accent,mark.title))
        else:
            enriched["reading"] = _point_data(frame,ctx,mark.points[0])
            overlays.append((mark.points,accent,mark.title))
        rows.append(enriched)
    canvas = mask_overlay(frame.rgb,areas if opts.get("tint_regions",True) else [],frame.present)
    overlay = png("survey-map",mark_image(canvas,overlays),ctx,crop=False)
    table = csv_table("marks",("name","type","role","length_mm","area_mm2","observed_length_mm","observed_area_mm2",
                               "observed_fraction"),
                      [(r["title"],r["kind"],r["role"],r["lengths_mm"],r["area_mm2"],r.get("observed_length_mm"),
                        r.get("observed_area_mm2"),r.get("observed_fraction")) for r in rows])
    summary = dict(marks=len(rows),polygons=sum(item["kind"]=="polygon" for item in rows),
                   physical_panel_mm=[dims.width_mm,dims.height_mm])
    return Result("survey","Physical condition survey",summary,{**summary,"marks":rows},(overlay,table))


def grid_loss(ctx):
    frame = ctx.one()
    opts = ctx.order.toolbox
    keys(opts,"columns","rows","min_cell")
    nx = integer(opts.get("columns",8),"Cell columns",2,24)
    ny = integer(opts.get("rows",6),"Cell rows",2,24)
    floor = integer(opts.get("min_cell",12),"Minimum support pixels",4,300)
    scale = ctx.scale
    w,h = scale.width_px,scale.height_px
    x0,y0,x1,y1 = ctx.order.region.indices(w,h)
    if (x1-x0)<nx or (y1-y0)<ny:
        raise Problem("The selected image window cannot hold that many non-empty cells.")
    xs = np.linspace(x0,x1,nx+1).round().astype(int)
    ys = np.linspace(y0,y1,ny+1).round().astype(int)
    rows = []
    lines = []
    colors = [(116,216,162),(239,207,119),(241,118,102)]
    for i in range(ny):
        for j in range(nx):
            bounds=[int(xs[j]),int(ys[i]),int(xs[j+1]),int(ys[i+1])]
            lx,ly,rx,ry=bounds
            block=frame.present[ly:ry,lx:rx]
            presented=int(block.sum())
            count=int(block.size)
            fallback=int(np.sum(frame.clipped[ly:ry,lx:rx] & block))
            category=2 if presented<floor or presented/count<.5 else (1 if presented<count or fallback else 0)
            cell=f"{chr(65+j)}{i+1}"
            rows.append(dict(cell=cell,bounds=bounds,observed=presented,clear=count-fallback-(count-presented),
                             absent=count-presented,clipped=fallback,
                             absent_area_mm2=rounded((count-presented)*scale.pixel_area),
                             observed_fraction=rounded(presented/count), category=["complete","partial","thin"][category]))
            lines.append(([(lx/w,ly/h),(rx/w,ly/h),(rx/w,ry/h),(lx/w,ry/h),(lx/w,ly/h)],colors[category],cell))
    check=int(ctx.selection.sum())
    full=int(np.sum(frame.present&ctx.selection))
    clipped=int(np.sum(frame.clipped&ctx.selection))
    area = check*scale.pixel_area
    photo=mask_overlay(frame.rgb,[(~frame.present&ctx.selection,(216,95,106),.68),
                                  (frame.clipped&ctx.selection,(236,196,92),.65)])
    atlas=png("physical-grid",mark_image(photo,lines),ctx)
    table=csv_table("cells",("cell","category","observed_px","clipped_px","missing_px","missing_mm2","observed_fraction"),
                    [(item["cell"],item["category"],item["observed"],item["clipped"],item["absent"],
                      item["absent_area_mm2"],item["observed_fraction"]) for item in rows])
    summary=dict(sampled_area_mm2=rounded(area),uncaptured_area_mm2=rounded((check-full)*scale.pixel_area),
                 clipped_area_mm2=rounded(clipped*scale.pixel_area),cells=len(rows),
                 thin_cells=sum(item["category"]=="thin" for item in rows))
    return Result("coverage","Capture coverage grid",summary,
                  {**summary,"cells":rows,"dimensions":[nx,ny],"note":"Missing pixels and saturation are recorded separately."},
                  (atlas,table))
