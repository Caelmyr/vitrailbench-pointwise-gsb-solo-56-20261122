from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from .bench import Result, percentiles, ratio
from .camera import linear_to_srgb
from .contracts import Box, Problem, integer, keys, label, option, rounded, scalar
from .picture import csv_table, mask_overlay, png
from .printroom import PDF, PDFPage, _page_dimensions, _photo_object, _scalebar


def _gray_bytes(frame,ctx,box,pad):
    rgb=mask_overlay(frame.rgb,[],frame.present)
    sides=(ctx.scale.step_x,ctx.scale.step_y)
    shape=rgb.shape[:2]
    bounds=box.indices(shape[1],shape[0])
    x0,y0,x1,y1=bounds
    view=rgb[y0:y1,x0:x1]
    image=Image.fromarray(view)
    if pad>0:
        border=max(1,round(pad/max(sides)))
        if max(image.size)+2*border>3200:
            raise Problem("Padded image is too large.",413)
        area=Image.new("RGB",(image.width+2*border,image.height+2*border),(242,235,218))
        area.paste(image,(border,border))
        image=area
    return image,bounds


def _cell_order(cells,columns):
    return [(i%columns,i//columns) for i in range(len(cells))]


@dataclass(frozen=True)
class Card:
    width: int
    height: int
    gap: int
    gutter: int
    photos: tuple
    captions: bool

    def page(self,cols,title):
        rows=math.ceil(len(self.photos)/cols)
        width=2*self.gutter+cols*self.width+(cols-1)*self.gap
        height=2*self.gutter+rows*(self.height+24)+(rows-1)*self.gap+28
        background=Image.new("RGB",(width,height),(239,234,219))
        draw=ImageDraw.Draw(background)
        draw.text((self.gutter,12),title,fill=(29,38,41))
        for slot,(col,row) in enumerate(_cell_order(self.photos,cols)):
            x=self.gutter+col*(self.width+self.gap)
            y=self.gutter+25+row*(self.height+self.gap+24)
            panel, caption, measure = self.photos[slot]
            if self.captions:
                draw.text((x,y+self.height+4),caption[:55],fill=(43,46,47))
            preview=panel.copy()
            preview.thumbnail((self.width,self.height),Image.Resampling.LANCZOS)
            draw.rectangle((x,y,x+self.width,y+self.height),fill=(36,42,47))
            background.paste(preview,(x+(self.width-preview.width)//2,y+(self.height-preview.height)//2))
        return background


def comparison_card(ctx):
    opts=ctx.order.toolbox
    keys(opts,"columns","tile_long_edge","border_mm","legend","title")
    columns=integer(opts.get("columns",2),"Images per row",1,4)
    long_edge=integer(opts.get("tile_long_edge",385),"Longest card tile edge",120,820)
    border=scalar(opts.get("border_mm",0),"Edge padding (mm)",0,25)
    legend=option(opts.get("legend","captions"),"Capture captions",("captions","plain"))
    title=label(opts.get("title","Registered comparisons"),"Sheet title",72)
    box=ctx.order.region
    format_rows=[]
    cells=[]
    width=ctx.scale.width_mm*(box.right-box.left)
    height=ctx.scale.height_mm*(box.bottom-box.top)
    aspect=(width+2*border)/(height+2*border)
    prime_w,prime_h=(long_edge,round(long_edge/aspect)) if aspect>=1 else (round(long_edge*aspect),long_edge)
    for index,frame in enumerate(ctx.frames,1):
        image,bounds=_gray_bytes(frame,ctx,box,border)
        source=frame.record
        region=ctx.selection
        share=ratio(frame.present&region,region)
        label_text=f"{source['data']['title']} · {share:.0%} observed"
        cells.append((image,label_text,share))
        format_rows.append(dict(slot=index,capture=source["id"],title=source["data"]["title"],bounds=list(bounds),
                                observed_fraction=share,original_sha=source["original"]))
    card=Card(prime_w,prime_h,16,18,tuple(cells),legend=="captions")
    page=card.page(min(columns,len(cells)),title)
    tableau=csv_table("tile-sources",("slot","source","capture","observed","original_sha"),
                     [(a["slot"],a["title"],a["capture"],a["observed_fraction"],a["original_sha"]) for a in format_rows])
    summary=dict(captures=len(cells),image_px=list(page.size),panel_window_mm=[rounded(width),rounded(height)],
                 capture_area_hint="Same panel window across all tiles")
    return Result("comparison-card","Registered photo contact card",summary,{**summary,"tiles":format_rows,"caption_mode":legend,
                  "note":"Contact tiles are digital previews. Their lengths are not intended as a scale reference."},
                  (png("contact-card",page),tableau))


def _transparent_export(frame,allowed):
    rgb=linear_to_srgb(frame.rgb)
    rgba=np.concatenate((rgb,np.uint8(255*allowed[...,None])),axis=2)
    rgba[~allowed]=0
    return rgba


def aligned_asset(ctx):
    frame=ctx.one()
    opts=ctx.order.toolbox
    keys(opts,"highlights","boundary","bounds")
    highlights=option(opts.get("highlights","include"),"Saturated pixels",("include","exclude"))
    boundary=option(opts.get("boundary","transparent"),"Outside support",("transparent","checker"))
    full=option(opts.get("bounds","region"),"Output window",("full","region"))
    selected=ctx.measured(frame,allow_clipped=highlights=="include")
    empty=ctx.selection&~selected
    ctx.demand(selected,floor=.008,pixels=20)
    if full=="full":
        selected=frame.present.copy() if highlights=="include" else frame.measurable.copy()
    rgba=_transparent_export(frame,selected)
    clear=png("registered-transparency",rgba,ctx,crop=full=="region")
    assets=[clear]
    if boundary=="checker":
        y,x=np.mgrid[:selected.shape[0],:selected.shape[1]]
        parity=(x//13+y//13)%2
        pad=np.where(parity[...,None],np.array([109,112,117]),np.array([78,82,88])).astype(np.uint8)
        padded=np.where(selected[...,None],rgba[...,:3],pad)
        assets.append(png("registered-over-checker",padded,ctx,crop=full=="region"))
    summary=dict(source=frame.record["id"],visible_fraction=ratio(selected,ctx.selection if full=="region" else np.ones_like(selected)),
                 qualified_clear_pixels=int(selected.sum()),physical_grid=ctx.scale.public()["raster"])
    return Result("aligned-asset","Registered inspection image",summary,
                  {**summary,"output_box":[0,0,1,1] if full=="full" else ctx.order.region.public(),
                   "region_excluded_pixels":int(empty.sum()),"highlight_policy":highlights,
                   "note":"Derived registered preview. Transparent pixels denote unsupported observation. Source bytes remain unchanged."},
                  tuple(assets))


def _microchips(width,col,gap,label_rows):
    inner=width-26
    tile=(inner-(col-1)*gap)/col
    height=4.2*label_rows+tile*.56
    return tile,height


def _ticks_on_page(pdf,page,font,image_id,pages,slots):
    commands=pdf.stream("\n".join(page.content))
    resources=f"/Font << /F1 {font} 0 R >>"
    if image_id:
        resources+=" /XObject << "+" ".join(f"/Im{i} {item} 0 R" for i,item in enumerate(image_id))+" >>"
    pid=pdf.add(f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {page.x(page.width)} {page.x(page.height)}] "
                f"/Resources << {resources} >> /Contents {commands} 0 R >>")
    slots.append(pid)


def _place_image(page,x,y,w,h,key):
    page.content.append(f"q {page.x(w)} 0 0 {page.x(h)} {page.x(x)} {page.y(y+h)} cm /Im{key} Do Q")


def _record_stamp(record):
    title=str(record["name"]).replace("\n"," ")
    return title[:43]+"…" if len(title)>44 else title


def archive_index_pdf(library, panel, reports, paper="a4", orientation="portrait"):
    width,height=_page_dimensions(paper,orientation)
    if not 1<=len(reports)<=25:
        raise Problem("Choose one to twenty-five inspections for the document index.")
    cols=2 if orientation=="portrait" else 3
    gap=3
    tile,row_h=_microchips(width,cols,gap,4)
    rows=max(1,int((height-58)/(row_h+8)))
    page_slots=cols*rows
    pdf=PDF()
    catalog=pdf.reserve()
    chapters=pdf.reserve()
    font=pdf.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages=[]
    for offset in range(0,len(reports),page_slots):
        number=len(pages)+1
        page=PDFPage(width,height,[])
        page.caption(13,18,"Vitrail Bench  /  inspection index",15)
        page.caption(13,25,f"{panel['title'][:60]} · {panel['accession'][:35]}",10)
        page.line(13,28,width-13,28)
        images=[]
        for slot,report in enumerate(reports[offset:offset+page_slots]):
            col=slot%cols
            row=slot//cols
            x=13+col*(tile+gap)
            y=33+row*(row_h+8)
            page.rectangle(x,y,tile,row_h)
            candidate=next((a for a in report["assets"] if a["mime"]=="image/png"),None)
            if candidate:
                jpeg=library.blobs.read(candidate["sha256"])
                pointer,img_w,img_h=_photo_object(pdf,jpeg)
                images.append(pointer)
                top=tile*.55
                factor=min((tile-4)/img_w,top/img_h)
                tw,th=img_w*factor,img_h*factor
                _place_image(page,x+(tile-tw)/2,y+(top-th)/2+1,tw,th,len(images)-1)
            base=y+tile*.55+5
            page.caption(x+2,base,_record_stamp(report),9)
            page.caption(x+2,base+4.5,f"{report['kind']} [{report['review']['state']}]",8)
            page.caption(x+2,base+8.5,report["id"][:17],7)
        page.caption(width-31,height-12,f"Page {number}",8)
        page.caption(13,height-12,"Thumbnail sizes are illustrative",8)
        _ticks_on_page(pdf,page,font,images,chapters,pages)
    pdf.set(chapters,f"<< /Type /Pages /Count {len(pages)} /Kids ["+" ".join(f"{item} 0 R" for item in pages)+"] >>")
    pdf.set(catalog,f"<< /Type /Catalog /Pages {chapters} 0 R >>")
    return pdf.assemble(catalog)
