from dataclasses import dataclass
from html import escape
from io import BytesIO
from xml.etree.ElementTree import Element, SubElement, tostring
import math

from PIL import Image

from .contracts import Problem, label, option, rounded, scalar


MM_TO_POINT = 72 / 25.4
PAPER = {"a4":(210,297),"letter":(215.9,279.4),"a3":(297,420)}


def _text(s):
    return str(s).encode("latin-1","replace").decode("latin-1").replace("\\","\\\\").replace("(","\\(").replace(")","\\)")


def _units(value):
    return f"{float(value):.4f}".rstrip("0").rstrip(".") or "0"


def _page_dimensions(paper,orientation):
    paper=option(paper,"Paper size",tuple(PAPER))
    orientation=option(orientation,"Orientation",("portrait","landscape"))
    w,h=PAPER[paper]
    return (h,w) if orientation=="landscape" else (w,h)


class PDF:
    def __init__(self):
        self.objects=[]

    def reserve(self):
        self.objects.append(b"")
        return len(self.objects)

    def set(self,number,payload):
        self.objects[number-1]=payload.encode("latin-1") if isinstance(payload,str) else payload

    def add(self,payload):
        n=self.reserve()
        self.set(n,payload)
        return n

    def stream(self,payload,props=""):
        if isinstance(payload,str):
            payload=payload.encode("latin-1")
        return self.add(f"<< /Length {len(payload)} {props} >>\nstream\n".encode()+payload+b"\nendstream")

    def assemble(self,catalog):
        target=BytesIO()
        target.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        indices=[]
        for i,part in enumerate(self.objects,1):
            indices.append(target.tell())
            target.write(f"{i} 0 obj\n".encode()+part+b"\nendobj\n")
        offset=target.tell()
        target.write(f"xref\n0 {len(indices)+1}\n0000000000 65535 f \n".encode())
        for value in indices:
            target.write(f"{value:010d} 00000 n \n".encode())
        target.write(f"trailer\n<< /Size {len(indices)+1} /Root {catalog} 0 R >>\nstartxref\n{offset}\n%%EOF\n".encode())
        return target.getvalue()


@dataclass
class PDFPage:
    width: float
    height: float
    content: list

    def x(self,val):
        return _units(val*MM_TO_POINT)

    def y(self,val):
        return _units((self.height-val)*MM_TO_POINT)

    def line(self,x1,y1,x2,y2,width=.22,ink="0.25 0.28 0.3"):
        self.content.append(f"{ink} RG {self.x(width)} w {self.x(x1)} {self.y(y1)} m {self.x(x2)} {self.y(y2)} l S")

    def rectangle(self,x,y,w,h,gray=None):
        if gray is None:
            op="S"
            self.content.append("0.3 G 0.7 w")
        else:
            op="f"
            self.content.append(f"{gray} g")
        self.content.append(f"{self.x(x)} {self.y(y+h)} {self.x(w)} {self.x(h)} re {op}")

    def caption(self,x,y,text,size=9):
        self.content.append(f"0.15 0.2 0.23 rg BT /F1 {size:g} Tf {self.x(x)} {self.y(y)} Td ({_text(text)}) Tj ET")

    def image(self,x,y,w,h):
        self.content.append(f"q {self.x(w)} 0 0 {self.x(h)} {self.x(x)} {self.y(y+h)} cm /Im0 Do Q")


def _photo_object(pdf,image_bytes):
    with Image.open(BytesIO(image_bytes)) as image:
        reduced=image.convert("RGB")
        reduced.thumbnail((1800,1800),Image.Resampling.LANCZOS)
        out=BytesIO()
        reduced.save(out,"JPEG",quality=88,optimize=True)
        w,h=reduced.size
        obj=pdf.stream(out.getvalue(),
                       f"/Type /XObject /Subtype /Image /ColorSpace /DeviceRGB /Filter /DCTDecode /Width {w} /Height {h} /BitsPerComponent 8")
        return obj,w,h


def _scalebar(page,x,y,length):
    page.line(x,y,x+length,y,width=.43)
    for step in (0,length/2,length):
        page.line(x+step,y-2,x+step,y+2)
    page.caption(x,y+6,f"{length:g} mm on paper",8)


def _short(text,count=83):
    value=str(text).replace("\n"," ")
    return value if len(value)<=count else value[:count-3]+"..."


def _sheet_summary(record):
    data=record["result"]
    summary=[]
    for name,value in data.items():
        if isinstance(value,(int,float,str,bool)) and name!="note":
            word=" ".join(name.split("_")).capitalize()
            summary.append((word,_short(value,36)))
        if len(summary)>=12:
            break
    return summary


def print_inspection(library,record,paper,orientation,asset_key=None):
    width,height=_page_dimensions(paper,orientation)
    available=[asset for asset in record["assets"] if asset["mime"]=="image/png"]
    selected=next((asset for asset in available if asset["key"]==asset_key),None) if asset_key else (available[0] if available else None)
    if asset_key and selected is None:
        raise Problem("Select an image attached to this report.",404)
    page=PDFPage(width,height,[])
    pdf=PDF()
    catalog=pdf.reserve()
    pages=pdf.reserve()
    unit=pdf.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    img_obj=None
    inner=width-30
    page.caption(15,20,"Vitrail Bench  /  inspection sheet",16)
    page.caption(15,29,_short(record["name"]),13)
    page.caption(15,37,f"Kind: {record['kind']}  |  Retained: {_short(record['created'], 25)}  |  Status: {record['review']['state']}",9)
    case=record['facts']['panel']
    page.caption(15,43,f"{_short(case['title'],50)}   {case['width_mm']:g} x {case['height_mm']:g} mm",9)
    page.line(15,47,width-15,47)
    y=55
    if selected:
        img_obj,img_w,img_h=_photo_object(pdf,library.blobs.read(selected["sha256"]))
        ratio=min(inner/img_w,(height-127)/img_h)
        w,h=img_w*ratio,img_h*ratio
        page.image(15+(inner-w)/2,y,w,h)
        y+=h+7
        page.caption(15,y,_short(selected["filename"])+" (image fitted to sheet)",8)
        y+=7
    for index,(key,value) in enumerate(_sheet_summary(record)):
        col=index%2
        row=index//2
        page.caption(15+col*inner/2,y+row*4.4,f"{_short(key,31)}: {_short(value,27)}",8)
    source_ids=record["facts"].get("sources",[])
    page.caption(15,height-27,f"Record {record['id']}  |  {len(source_ids)} source capture(s)",8)
    if case.get("synthetic"):
        page.caption(15,height-21,"Synthetic source specimen",8)
    _scalebar(page,15,height-13,min(40,inner/3))
    commands=pdf.stream("\n".join(page.content))
    resources=f"/Font << /F1 {unit} 0 R >>"
    if img_obj is not None:
        resources+=f" /XObject << /Im0 {img_obj} 0 R >>"
    p=pdf.add(f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {page.x(width)} {page.x(height)}] "
              f"/Resources << {resources} >> /Contents {commands} 0 R >>")
    pdf.set(pages,f"<< /Type /Pages /Count 1 /Kids [{p} 0 R] >>")
    pdf.set(catalog,f"<< /Type /Catalog /Pages {pages} 0 R >>")
    return pdf.assemble(catalog)


def _svg_line(svg,a,b,color="#232b39",width=.25):
    SubElement(svg,"line",{"x1":_units(a[0]),"y1":_units(a[1]),"x2":_units(b[0]),"y2":_units(b[1]),
                          "stroke":color,"stroke-width":_units(width)})


def _svg_text(svg,x,y,value,font_size=4):
    node=SubElement(svg,"text",{"x":_units(x),"y":_units(y),"fill":"#242c34",
                                "font-family":"sans-serif","font-size":_units(font_size)})
    node.text=str(value)


def scale_target(width,height,ticks,accent):
    width=scalar(width,"Frame width (mm)",24,600)
    height=scalar(height,"Frame height (mm)",24,600)
    gap=scalar(ticks,"Target interval (mm)",2,min(width,height)/2)
    name=label(accent,"Target label",68)
    margin=11
    sw,sh=width+2*margin,height+2*margin+13
    svg=Element("svg",{"xmlns":"http://www.w3.org/2000/svg","width":f"{sw:g}mm","height":f"{sh:g}mm",
                       "viewBox":f"0 0 {sw:g} {sh:g}"})
    SubElement(svg,"rect",{"x":"0","y":"0","width":str(sw),"height":str(sh),"fill":"white"})
    SubElement(svg,"rect",{"x":str(margin),"y":str(margin),"width":_units(width),"height":_units(height),
                          "fill":"none","stroke":"#394451","stroke-width":".25"})
    for side_length,axis in ((width,0),(height,1)):
        last=int(math.floor(side_length/gap))
        for i in range(last+1):
            at=margin+i*gap
            heavy=i%5==0
            marker=3.2 if heavy else 1.8
            if i>0 and i<last and heavy:
                if axis==0:
                    _svg_text(svg,at+1,margin-4,_units(i*gap),2.5)
                else:
                    _svg_text(svg,margin+1,at-1,_units(i*gap),2.5)
            if axis==0:
                _svg_line(svg,(at,margin-marker),(at,margin))
                _svg_line(svg,(at,margin+height),(at,margin+height+marker))
            else:
                _svg_line(svg,(margin-marker,at),(margin,at))
                _svg_line(svg,(margin+width,at),(margin+width+marker,at))
    _svg_text(svg,margin,sh-14,name)
    _svg_text(svg,margin,sh-8,f"{width:g} × {height:g} mm  ⎮  {gap:g} mm ticks  ⎮  print at 100%",3.1)
    return tostring(svg,encoding="utf-8",xml_declaration=True)
