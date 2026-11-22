from dataclasses import dataclass

from .bubbles import paired_bubbles
from .calibration import calibration_grid, camera_sampling_section, ruler_audit
from .correspondence import pair_control, pattern_spacing, registered_edges
from .inscription import frosted_haze, inscription_survey
from .leads import lead_network, solder_joints
from .observation import patch_chart, photo_gaps, source_probe
from .panes import lead_gauge, pane_catalog
from .photometry import color_field, density, flatness, optical_section, registered_compare, transmission
from .pigment import absorption_strip, paint_report, stain_survey
from .quality import border_shadow, channel_clip, exposure_check, repeatability
from .resolution import edge_resolution, focus_map
from .shape_checks import boundary_intensity, seed_inclusions, support_distance
from .sheets import aligned_asset, comparison_card
from .traces import line_impression, strip_pair
from .surface import crack_candidates, edge_roughness, glare, texture
from .survey import grid_loss, survey_measure
from .contracts import Problem


@dataclass(frozen=True)
class Field:
    tag: str
    label: str
    form: str
    default: object
    extra: object = None

    def public(self):
        return {"key":self.tag,"label":self.label,"input":self.form,"default":self.default,"options":self.extra}


@dataclass(frozen=True)
class Tool:
    key: str
    title: str
    area: str
    input_count: int
    fields: tuple
    run: object
    explanation: str

    def public(self):
        return dict(key=self.key,title=self.title,area=self.area,input_count=self.input_count,
                    fields=[field.public() for field in self.fields],explanation=self.explanation)


def number(name,title,value, increments=.1):
    return Field(name,title,"number",value,increments)


def selection(name,title,value,choices):
    return Field(name,title,"select",value,choices)


def drawn(name,title,value):
    return Field(name,title,"position",value)


LEAD=(
    number("absolute_dark","Absolute signal ceiling",.10,.005),
    number("local_radius_mm","Background radius (mm)",5),
    number("min_component","Minimum component pixels",12,1),
    number("close_mm","Closing length (mm)",0),
    number("min_run_mm","Minimum run (mm)",2.2),
)

TOOLS=(
    Tool("transmission","Transmitted signal","Light & color",1,(
        number("bins","Histogram bins",20,1),number("display_ceiling","Display ceiling",1),
        number("tiles","Tiles along each side",4,1)),transmission,
         "Linear luminance, clipping and tile readings for the active region."),
    Tool("density","Reference optical density","Light & color",1,(
        drawn("reference","Bright reference box",[.44,.38,.59,.49]),number("max_density","Display limit",2.5),
        selection("allow_brighter","Brighter than reference","signed",("signed","clamp"))),density,
         "Read relative density from an observed reference patch."),
    Tool("flatness","Backlight level survey","Light & color",1,(
        drawn("reference","Reference box",[.4,.35,.6,.65]),number("radius_mm","Neighborhood (mm)",8),
        number("allowed_delta_ev","Allowed deviation (EV)",.2,.01)),flatness,
         "Compare slowly varying brightness with a near-center reading."),
    Tool("compare","Registered capture comparison","Light & color",2,(
        number("tolerance_ev","Difference limit (EV)",.25,.01),
        drawn("reference","Shared reference area",[0,0,1,1]),number("mix","Wipe boundary",.5,.05)),registered_compare,
         "Compare two captures in the physical panel plane."),
    Tool("color","Transmission color map","Light & color",1,(
        drawn("reference","Patch box",[.44,.38,.59,.49]),number("threshold","Color distance display limit",18)),
         color_field,"Map color distance from a patch under the declared lighting."),
    Tool("swatches","Named color patch readings","Light & color",0,(),patch_chart,
         "Read the fixed set of neutral panel-coordinate patches across selected captures."),
    Tool("section","Optical section","Light & color",1,(
        drawn("path","Section points",[[.15,.5],[.85,.5]]),number("spacing_mm","Reading spacing (mm)",.7)),optical_section,
         "Trace brightness along a line or polyline while preserving missing intervals."),
    Tool("leads","Lead came network","Construction",1,LEAD+(
        selection("include_ends","Endpoints","show",("show","hide")),),lead_network,
         "Extract a dark lead-like network, with path lengths and width estimates."),
    Tool("joints","Solder joint review","Construction",1,LEAD+(
        number("arm_radius_mm","Arm check radius (mm)",5),),solder_joints,
         "Inspect arm separation and visible support at possible junctions."),
    Tool("panes","Window-pane catalog","Construction",1,LEAD+(
        number("close_lead_gap_mm","Preview gap closing (mm)",.8),
        number("min_pane_mm2","Minimum pane (mm²)",120,1)),pane_catalog,
         "Label connected interiors and distinguish open or under-observed edges."),
    Tool("gauge","Lead profile gauge","Construction",1,(
        drawn("from","Start",[.45,.2]),drawn("to","End",[.45,.8]),
        number("half_width_mm","Gauge half width (mm)",5),number("darkness","Lead signal threshold",.07,.005),
        number("crossings","Cross sections",17,1)),lead_gauge,
         "Measure the dark run across a drawn centerline when both edges are visible."),
    Tool("texture","Impressed glass texture","Surface",1,(
        number("wavelength_mm","Feature spacing (mm)",2.4),number("neighborhood_mm","Neighborhood (mm)",7.5),
        number("min_coherence","Minimum direction coherence",.36,.03)),texture,
         "Inspect texture contrast and direction in local neighborhoods."),
    Tool("cracks","Crack candidate tracing","Surface",1,(
        number("radius_mm","Background radius (mm)",1.4),number("amplitude","Signal change",.052,.004),
        number("min_length_mm","Minimum run (mm)",5),
        selection("polarity","Candidate polarity","bright",("bright","dark","both"))),crack_candidates,
         "Locate narrow ridges that may be cracks, glare or painted details."),
    Tool("glare","Saturated highlights","Surface",1,(
        number("clip_code","Near-clip code",.985,.005),number("halo_radius_mm","Halo width (mm)",1.6),
        number("min_patch_mm2","Minimum patch (mm²)",1.15)),glare,
         "Map camera overload and neighboring zones."),
    Tool("edges","Edge and paint-detail survey","Surface",1,(
        number("smoothing_mm","Smoothing radius (mm)",1.4),number("gradient_floor","Strong gradient",.12,.01)),edge_roughness,
         "Compare physical-scale gradients and local departures."),
    Tool("probe","Source pixel probe","Coverage & records",0,(
        drawn("point","Panel point",[.5,.5]),number("radius_mm","Local radius (mm)",2.5)),source_probe,
         "Read the matching registered locations in selected camera originals."),
    Tool("gaps","Source gaps and risky margins","Coverage & records",1,(
        number("border_width_mm","Margin warning distance (mm)",3.2),number("min_area_mm2","Minimum gap (mm²)",4)),photo_gaps,
         "Identify missing registered camera pixels and narrow supported margins."),
    Tool("coverage","Capture coverage grid","Coverage & records",1,(
        number("columns","Columns",8,1),number("rows","Rows",6,1),number("min_cell","Minimum pixels",12,1)),grid_loss,
         "Account for observed, clipped and unavailable pixels in panel cells."),
    Tool("survey","Physical condition survey","Coverage & records",1,(),survey_measure,
         "Measure a saved survey on the active capture."),
    Tool("calibration","Registered scale audit","Camera calibration",1,(
        number("min_ppmm","Minimum detail (px/mm)",2.6),
        number("max_distortion","Maximum stretch",1.35),
        number("spacing_mm","Calibration marks (mm)",18)),calibration_grid,
        "Check where perspective gives the weakest sampling."),
    Tool("rulers","Measured length cross-check","Camera calibration",1,(
        number("tolerance_mm","Length tolerance (mm)",1.5),),ruler_audit,
        "Read direction and measured physical span in the aligned panel plane."),
    Tool("sampling-section","Camera sampling section","Camera calibration",1,(
        drawn("from","Start",[.15,.5]),
        drawn("to","End",[.85,.5]),
        number("spacing_mm","Read interval (mm)",1.2)),camera_sampling_section,
        "Check the spread of camera scale along a user-marked line."),
    Tool("matching-pairs","Paired capture mark residuals","Camera calibration",2,(
        number("tolerance_mm","Misalignment tolerance (mm)",2),),pair_control,
        "Measure residual separation between points on aligned camera sources."),
    Tool("paired-edges","Registered edge comparison","Camera calibration",2,(
        number("analysis_scale_mm","Edge scale (mm)",1.3),
        number("threshold","Minimum edge energy",.007,.001),
        number("only_one_amplitude","Single image contrast",.55,.01)),registered_edges,
        "Compare aligned edge strengths and identify evidence unique to either camera."),
    Tool("periodic","Pattern spacing survey","Surface",1,(
        number("step_budget_px","Longest pattern step (pixels)",65,1),
        number("display_limit","Readout level",.70,.02),
        number("min_overlap","Minimum sampled overlap",.66,.01)),pattern_spacing,
        "Find recurring spatial offsets within a selected patch."),
    Tool("inscription","Painted mark legibility","Surface",1,(
        selection("polarity","Mark polarity","dark",("dark","bright")),
        number("radius_mm","Local character scale (mm)",1.2),
        number("relative_contrast","Minimum local contrast",.15,.01),
        number("min_area_mm2","Minimum part area (mm²)",.8)),inscription_survey,
        "Review which parts of a painted mark remain supported by the image."),
    Tool("haze","Clouded glass detail survey","Surface",1,(
        drawn("reference","Clean patch",[.38,.13,.58,.30]),
        number("broad_mm","Haze neighborhood (mm)",7.5),
        number("fine_mm","Fine texture diameter (mm)",1.3),
        number("strips","Summary strips",18,1)),frosted_haze,
        "Compare relative loss of fine features against a marked clean patch."),
    Tool("paint","Paint absorption estimate","Light & color",1,(
        drawn("background","Glass reference",[.41,.14,.58,.24]),
        number("absorption","Absorption threshold",.25,.01),
        number("selectivity","Channel agreement",.22,.01),
        number("min_patch_mm2","Smallest patch (mm²)",1.8)),paint_report,
        "Flag neutral absorbing fragments against a clean glass reference."),
    Tool("absorption-strip","Strip absorption profile","Light & color",1,(
        drawn("reference","Glass reference",[.36,.13,.58,.28]),
        selection("axis","Direction","horizontal",("horizontal","vertical")),
        number("bins","Strips",28,1),
        number("min_support","Minimum support",.38,.01)),absorption_strip,
        "Track relative absorption across a selected panel area."),
    Tool("stains","Local tonal stain check","Light & color",1,(
        drawn("color_reference","Glass reference",[.43,.4,.54,.5]),
        number("chroma_difference","Color difference",8,1),
        number("radius_mm","Local radius (mm)",3.2)),stain_survey,
        "Compare local color changes after smoothing within clear source support."),
    Tool("repeat","Capture repeatability","Camera calibration",0,(
        drawn("reference","Repeated reference",[.4,.3,.6,.7]),
        number("threshold","Absolute variation",.028,.001)),repeatability,
        "Compare signal changes in the jointly measured areas of repeated photos."),
    Tool("exposure","Exposure consistency check","Camera calibration",2,(
        number("minimum_linear","Dark threshold",.020,.002),
        number("interval_count","Intervals",16,1),
        number("max_channel_drift","Channel deviation (EV)",.28,.02)),exposure_check,
        "Compare camera response on common observations."),
    Tool("channels","Camera channel clipping","Camera calibration",1,(
        number("clip_code","Clip code",.99,.005),
        number("bins","Histogram size",40,1),
        number("shadow_code","Dark code",.04,.005)),channel_clip,
        "Map which camera channel reaches its code limits."),
    Tool("border-shadow","Panel edge shading","Light & color",1,(
        number("zones","Inset steps",24,1),
        number("darkness_limit","Dark relative to middle",.65,.01),
        drawn("middle","Middle reference",[.32,.31,.68,.70])),border_shadow,
        "Review edge falloff in physical inset bands."),
    Tool("resolution","Slanted-edge capture resolution","Camera calibration",1,(
        drawn("from","Edge start",[.42,.30]),drawn("to","Edge end",[.70,.33]),
        number("strip_halfwidth_mm","Sampling strip half-width (mm)",3),
        number("subsamples","Working-pixel divisions",2,1)),edge_resolution,
        "Estimate edge transfer from a well-defined slanted image boundary."),
    Tool("focus","Local detail consistency","Surface",1,(
        number("detail_mm","Detail scale (mm)",1.2),
        number("columns","Columns",5,1),
        number("rows","Rows",4,1),
        drawn("reference","Focus reference",[.25,.2,.75,.8])),focus_map,
        "Find areas whose fine detail differs from the marked reference."),
    Tool("inclusions","Bubble and inclusion candidates","Surface",1,(
        number("radius_mm","Neighborhood (mm)",1.5),
        number("cutoff","Local signal difference",.042,.001),
        number("min_area","Minimum area (mm²)",.27,.05),
        number("max_area","Maximum area (mm²)",10),
        selection("polarity","Candidate polarity","dark",("dark","bright"))),seed_inclusions,
        "Inspect compact variations and their physical shape."),
    Tool("bubble-census","Paired bubble census","Surface",2,(
        number("radius_mm","Neighborhood (mm)",1.5),
        number("cutoff","Local signal difference",.042,.001),
        number("min_area","Minimum spot area (mm²)",.27,.05),
        number("max_area","Maximum spot area (mm²)",30,1),
        selection("polarity","Candidate polarity","dark",("dark","bright")),
        number("match_mm","Centroid match tolerance (mm)",1,.05)),paired_bubbles,
        "Census one region against two captures: count a matching bubble once, keep the possible range where a spot may hide two, and list spots seen in only one capture."),
    Tool("boundary","Marked-region boundary intensity","Coverage & records",1,(
        drawn("boundary","Mark boundary",[[.38,.38],[.63,.39],[.62,.61],[.39,.60]]),
        number("edge_mm","Boundary band (mm)",2.7),
        number("baseline","Display scale",.4,.02)),boundary_intensity,
        "Read boundary bands and the center of a marked shape."),
    Tool("support-buffer","Continuous support distance","Coverage & records",1,(
        number("buffer_mm","Support buffer (mm)",2.5),
        number("limit_mm","Display limit (mm)",14,1)),support_distance,
        "Find distances to missing or saturated observations."),
    Tool("comparison-card","Registered photo contact card","Coverage & records",0,(
        number("columns","Images per row",2,1),
        number("tile_long_edge","Tile long edge (px)",385,1),
        number("border_mm","Edge padding (mm)",0),
        selection("legend","Captions","captions",("captions","plain"))),comparison_card,
        "Arrange the same panel coordinates from multiple captures."),
    Tool("aligned-asset","Registered inspection image","Coverage & records",1,(
        selection("highlights","Saturated pixels","include",("include","exclude")),
        selection("boundary","Outside support","transparent",("transparent","checker")),
        selection("bounds","Export window","region",("region","full"))),aligned_asset,
        "Export a derived panel view with visibly supported pixels."),
    Tool("trace","Drawn line impression","Construction",1,(
        drawn("path","Trace points",[[.13,.48],[.66,.52],[.84,.46]]),
        selection("response","Contrast direction","dark",("dark","bright")),
        number("half_width_mm","Trace half width (mm)",5),
        number("baseline_mm","Local background (mm)",1.4),
        number("spacing_mm","Section spacing (mm)",1.2),
        number("cutoff","Minimum local response",.028,.002)),line_impression,
        "Compare a hand drawn line with nearby observed features."),
    Tool("strip-pair","Registered strip comparison","Coverage & records",2,(
        drawn("path","Strip points",[[.20,.42],[.79,.56]]),
        number("half_width_mm","Strip half width (mm)",3.5),
        number("spacing_mm","Read spacing (mm)",1.1),
        number("support","Minimum shared support",.65,.05)),strip_pair,
        "Compare jointly captured registered strip sections.")

)
INDEX={tool.key:tool for tool in TOOLS}


def tool_by_key(key):
    tool=INDEX.get(key)
    if tool is None:
        raise Problem("Unknown image inspection.",404)
    return tool
