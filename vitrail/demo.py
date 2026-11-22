from dataclasses import dataclass
from io import BytesIO
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from .camera import linear_to_srgb
from .contracts import PanelSpec, PhotoSpec, Problem


W,H=880,660


@dataclass(frozen=True)
class Specimen:
    key: str
    title: str
    accession: str
    width_mm: float
    height_mm: float
    theme: str
    caption: str

    def public(self):
        return dict(key=self.key,title=self.title,accession=self.accession,width_mm=self.width_mm,height_mm=self.height_mm,
                    caption=self.caption,thumbnail=f"/api/samples/{self.key}/thumbnail")


SAMPLES=(
    Specimen("ivy","Ivy Stairlight","SYN-ST11",176,132,"ivy",
             "Three glazing bays with soft rolled texture, painted details and a sloping bright hairline."),
    Specimen("lantern","Copper Lantern","SYN-LA28",154,115.5,"lantern",
             "Honey and cobalt color fields with crossing leadwork, a bright highlight and a missing camera margin."),
)


IVY_COLORS=np.array([[.14,.37,.52],[.62,.42,.10],[.39,.21,.19],[.19,.40,.26],[.30,.49,.31],
                     [.51,.24,.31],[.46,.35,.19],[.17,.29,.48],[.51,.39,.28]],dtype=np.float32)
LANTERN_COLORS=np.array([[.12,.23,.53],[.54,.28,.12],[.68,.47,.17],[.18,.33,.47],[.58,.39,.15],
                         [.41,.23,.33],[.29,.42,.56],[.45,.26,.11],[.58,.31,.14]],dtype=np.float32)


def _seam_limits(u,v,theme):
    if theme=="ivy":
        a=.31+.05*np.sin(11*v)
        b=.67-.03*np.sin(7*v+1)
        c=.34+.021*np.sin(8*u)
        d=.66-.024*np.sin(6*u)
    else:
        a=.29+.050*v
        b=.68-.040*v
        c=.29+.045*np.sin(6*u)
        d=.68+.033*np.cos(11*u)
    return a,b,c,d


def _insert_lines(layer, u, v, theme):
    stripe_fn=(
        lambda x: .77-.15*x+.014*np.sin(58*x),
        lambda x: .18+.23*x+.009*np.sin(83*x),
    )
    center1=stripe_fn[0](u)
    center2=stripe_fn[1](u)
    crack=(np.abs(v-center1)<.0018)&(u>.40)&(u<.84)
    faint=(np.abs(v-center2)<.0013)&(u>.45)&(u<.70)
    if theme=="lantern":
        crack=(np.abs(v-.74+.46*u)<.002)&(u>.47)&(u<.77)
    layer[crack]=np.minimum(layer[crack]+.25,1.1)
    layer[faint]=np.maximum(layer[faint]-.073,.005)
    return layer


def _paint(u,v,glass,theme):
    waves=(v-.53)+.036*np.sin(19*u)+.018*np.sin(61*u)
    edge=(u>.14)&(u<.86)&(np.abs(waves)<.010)
    mark=(u>.11)&(u<.88)&(np.abs(waves-.034*np.sin(17*u))<.008)
    drops=np.absolute(np.mod(u*48,1)-.5)<.08
    detail=mark&(drops|edge)
    depth=np.where(detail,.48,1.0)
    if theme=="lantern":
        ribbon=np.abs(v-.62+.045*np.cos(17*u))<.016
        depth=np.where(ribbon&(u>.2)&(u<.82),.61,depth)
    return glass*depth[...,None]


def _glass(u,v,theme,flavor):
    a,b,c,d=_seam_limits(u,v,theme)
    pane_x=(u>a).astype(np.uint8)+(u>b).astype(np.uint8)
    pane_y=(v>c).astype(np.uint8)+(v>d).astype(np.uint8)
    position=3*pane_y+pane_x
    colors=IVY_COLORS if theme=="ivy" else LANTERN_COLORS
    glass=colors[position].copy()
    grain=.94+.08*np.sin(36*u+89*v)+.035*np.sin(199*u-19*v)+.012*np.sin(260*u+530*v)
    noise=np.random.default_rng(6209).normal(size=u.shape)*.006
    texture=grain+noise+.06*np.sin(22*u+np.sin(14*v)*2)
    glass *= texture[...,None]
    glass=_paint(u,v,glass,theme)
    lead=(np.minimum(np.abs(u-a),np.abs(u-b))<.008) | (np.minimum(np.abs(v-c),np.abs(v-d))<.009)
    outer=(u<.032)|(v<.035)|(u>.968)|(v>.965)
    tiny_cross=(u>.04)&(u<.30)&(np.abs(v-0.19-.4*u)<.008)
    solder=np.exp(-((u-a)**2+(v-c)**2)/.0013)
    dark=.008+.004*np.sin(u*220+v*380)+.018*solder
    if theme=="ivy":
        lead |= tiny_cross
    glass[lead|outer]=dark[lead|outer,None]*np.array([1.05,1,.95])
    glass=_insert_lines(glass,u,v,theme)
    power=.93+.11*u-.15*v+.1*np.cos(3*u-2*v)
    if flavor=="comparison":
        power=.79+.24*u-.020*v
        streak=np.exp(-((u-.70)**2/.0076+(v-.30)**2/.0038))*.28
        glass+=streak[...,None]*np.array([1,1.03,1.10])
    if flavor=="reflected":
        glass=.49*glass+.09
        glint=np.exp(-((u-.37)**2/.0032+(v-.21)**2/.0014))*1.10
        glass+=glint[...,None]
    if flavor=="flat":
        levels=.60*power+.02*np.sin(18*u+13*v)
        glass=levels[...,None]*np.array([1.03,1.00,.95])
        return glass,np.ones(u.shape,dtype=bool)
    active=~((u<.032)&(v>.61)) if flavor!="reflected" else ~((u>.95)&(v>.65))
    if theme=="lantern" and flavor!="transmitted":
        active = active & ~((u>.63)&(u<.72)&(v>.90))
    if flavor=="comparison":
        active &= ~((u>.82)&(v>.71))
    return glass*power[...,None],active


@lru_cache(maxsize=8)
def capture(theme,flavor):
    yy,xx=np.mgrid[:H,:W]
    u,v=(xx+.5)/W,(yy+.5)/H
    light,present=_glass(u,v,theme,flavor)
    image=np.concatenate((linear_to_srgb(light),np.uint8(present[...,None]*255)),axis=2)
    out=BytesIO()
    Image.fromarray(image).save(out,format="PNG",compress_level=4)
    return out.getvalue()


def load_demo(library,key):
    sample=next((item for item in SAMPLES if item.key==key),None)
    if sample is None:
        raise Problem("Demonstration specimen not found.",404)
    meta=PanelSpec(sample.title,sample.accession,sample.width_mm,sample.height_mm,
                   f"Synthetic example. {sample.caption}",True)
    roles=(("Transmitted under even LED","transmitted"),
           ("Second transmitted reading","comparison"),
           ("Reflected lamp angle","reflected"),
           ("Open backlight capture","flat"))
    images=[]
    for title,role in roles:
        photo=PhotoSpec(title,"Generated demonstration pixels.",role,0,0,"Synthetic lamp")
        images.append((f"{sample.key}-{role}.png",capture(sample.theme,role),photo))
    survey={"marks":[
        {"key":"demoa","kind":"line","role":"feature","title":"Visible sloping line","points":[[.43,.71],[.65,.67],[.82,.648]]},
        {"key":"demob","kind":"polygon","role":"repair","title":"Patch for overview",
         "points":[[.15,.23],[.28,.24],[.28,.33],[.16,.31]]},
        {"key":"democ","kind":"point","role":"datum","title":"Panel datum","points":[[.24,.50]]},
    ]}
    return library.load_sample(meta,key,images,[("Demonstration marks",survey)])


@lru_cache(maxsize=2)
def thumbnail(sample):
    image=Image.open(BytesIO(capture(sample.theme,"transmitted"))).convert("RGB")
    image.thumbnail((550,390),Image.Resampling.LANCZOS)
    overlay=ImageDraw.Draw(image)
    overlay.rectangle((0,image.height-24,138,image.height),fill=(46,41,42))
    overlay.text((8,image.height-17),"Synthetic specimen",fill=(249,239,211))
    stream=BytesIO()
    image.save(stream,format="WEBP",quality=83)
    return stream.getvalue()
