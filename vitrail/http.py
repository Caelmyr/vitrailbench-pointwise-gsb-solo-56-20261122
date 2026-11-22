from io import BytesIO
from pathlib import Path
import json
import os
import re
import sys
from urllib.parse import quote

from flask import Flask, Response, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from .bench import Bench, WorkBudget
from .camera import MAX_PHOTO_BYTES, Alignment, open_camera
from .contracts import (Problem, PanelSpec, PhotoSpec, WorkOrder, bool_value, digest, ident,
                        integer, keys, label, object_data, option, safe_filename)
from .demo import SAMPLES, load_demo, thumbnail
from .library import Library
from .packets import MAX_ARCHIVE, archive_inspection, verify_packet
from .picture import mask_overlay, png
from .provenance import inspection_sources
from .printroom import print_inspection, scale_target
from .sheets import archive_index_pdf
from .survey import parse_survey
from .tools import TOOLS, tool_by_key


STATIC=Path(__file__).resolve().parents[1]/"static"
IDENTITY="[a-zA-Z0-9_-]{1,90}"


def _json():
    if not request.is_json:
        raise Problem("Send data as JSON.")
    try:
        payload=request.get_json()
    except BadRequest as error:
        raise Problem("JSON body could not be decoded.") from error
    return object_data(payload)


def _file(key,limit):
    file=request.files.get(key)
    if file is None:
        raise Problem(f"Choose a {key} file.")
    filename=safe_filename(file.filename or "")
    data=file.stream.read(limit+1)
    if len(data)>limit:
        raise Problem(f"The file exceeds {round(limit/1_000_000)} MB.",413)
    if not data:
        raise Problem("The uploaded file is empty.")
    return filename,data


def _form_object(key,value):
    raw=request.form.get(key)
    if raw is None:
        return value
    try:
        result=json.loads(raw)
    except (ValueError,RecursionError) as error:
        raise Problem(f"The {key} field is not valid JSON.") from error
    return object_data(result,key)


def _deliver(contents,mime,name,download=False):
    result=send_file(BytesIO(contents),mimetype=mime,as_attachment=download,
                     download_name=name,etag=digest(contents),max_age=0)
    result.headers["Cache-Control"]="private, no-cache"
    result.headers["X-Content-Type-Options"]="nosniff"
    return result


def _expect_key(value,name="File"):
    if not re.fullmatch(IDENTITY,value):
        raise Problem(f"{name} not found.",404)
    return value


def _build_record(report):
    rid=report["id"]
    return {**report,"assets":[{**asset,"url":f"/api/inspections/{rid}/files/{quote(asset['key'])}"}
                              for asset in report["assets"]]}


def register_library(app,library,executor):
    @app.get("/api/library")
    def list_panels():
        return {"panels":library.panel_list()}

    @app.get("/api/samples")
    def list_samples():
        return {"samples":[item.public() for item in SAMPLES]}

    @app.get("/api/samples/<key>/thumbnail")
    def sample_cover(key):
        specimen=next((item for item in SAMPLES if item.key==key),None)
        if specimen is None:
            raise Problem("Demonstration specimen not found.",404)
        return _deliver(thumbnail(specimen),"image/webp",key+".webp")

    @app.post("/api/samples/<key>")
    def sample_select(key):
        with executor:
            panel,created=load_demo(library,key)
        return {"panel":panel,"created":created},201 if created else 200

    @app.post("/api/panels")
    def add_panel():
        spec=PanelSpec.parse(_json())
        panel,_=library.create_panel(spec)
        return {"panel":panel},201

    @app.get("/api/panels/<panel_id>")
    def panel_details(panel_id):
        return library.panel_summary(ident(panel_id))

    @app.put("/api/panels/<panel_id>")
    def change_panel(panel_id):
        pid=ident(panel_id)
        doc=_json()
        keys(doc,"revision","panel")
        previous=library.panel_summary(pid)["panel"]
        revision=integer(doc.get("revision"),"Panel revision",1,2_000_000)
        spec=PanelSpec.parse(doc.get("panel"),existing=previous)
        return {"panel":library.update_panel(pid,revision,spec)}

    @app.post("/api/panels/<panel_id>/photos")
    def put_photo(panel_id):
        filename,data=_file("photo",MAX_PHOTO_BYTES)
        fields=_form_object("capture",{})
        spec=PhotoSpec.parse(fields,filename=filename)
        with executor:
            photo=library.add_photo(ident(panel_id),filename,data,spec)
        return {"photo":photo},201

    @app.get("/api/photos/<photo_id>")
    def get_photo(photo_id):
        return {"photo":library.photo(ident(photo_id))}

    @app.put("/api/photos/<photo_id>")
    def change_photo(photo_id):
        cap=ident(photo_id)
        doc=_json()
        keys(doc,"revision","capture")
        revision=integer(doc.get("revision"),"Capture revision",1,2_000_000)
        spec=PhotoSpec.parse(doc.get("capture"),existing=library.photo(cap)["data"])
        return {"photo":library.update_photo(cap,revision,spec)}

    @app.get("/api/photos/<photo_id>/view/<kind>")
    def photo_media(photo_id,kind):
        if kind not in ("original","preview"):
            raise Problem("Image representation not found.",404)
        photo,contents=library.photo_bytes(ident(photo_id))
        if kind=="original":
            return _deliver(contents,photo["image"]["mime"],photo["filename"],True)
        with executor:
            camera=open_camera(contents)
            preview=camera.preview()
        return _deliver(preview,"image/png",safe_filename(photo["data"]["title"])+".png")

    @app.get("/api/photos/<photo_id>/alignments")
    def list_alignments(photo_id):
        return library.alignments(ident(photo_id))

    @app.post("/api/photos/<photo_id>/alignments")
    def fit_alignment(photo_id):
        cap=ident(photo_id)
        doc=_json()
        keys(doc,"revision","alignment","preview")
        proposed=Alignment.parse(doc.get("alignment"))
        revision=integer(doc.get("revision"),"Capture revision",1,2_000_000)
        if bool_value(doc.get("preview",True),"Preview"):
            photo=library.photo(cap)
            if photo["revision"]!=revision:
                raise Problem("The capture was changed on another screen.",409)
            audit=_try_alignment(library,photo,proposed,executor)
            return {"alignment":proposed.public(),**audit}
        return {"alignment":library.add_alignment(cap,revision,proposed)},201

    @app.put("/api/photos/<photo_id>/active-alignment")
    def restore_alignment(photo_id):
        body=_json()
        keys(body,"revision","alignment_id")
        revision=integer(body.get("revision"),"Capture revision",1,2_000_000)
        register=body.get("alignment_id")
        if register is not None:
            register=ident(register,"Alignment")
        return {"photo":library.use_alignment(ident(photo_id),revision,register)}


def _try_alignment(library,photo,proposed,executor):
    from .camera import PhysicalGrid, normalized_grid, sample_rgb, srgb_to_linear
    from .contracts import rounded
    panel=library.panel_summary(photo["panel_id"])["panel"]
    with executor:
        _,contents=library.photo_bytes(photo["id"])
        camera=open_camera(contents)
        dims=PhysicalGrid.from_panel(panel,[camera],640)
        plan=normalized_grid(dims.width_px,dims.height_px)
        points=proposed.to_camera(plan)
        code,usable=sample_rgb(camera,points)
        asset=png("aligned-camera",mask_overlay(srgb_to_linear(code),[],usable))
    return {"image":asset.presentation(),"coverage":rounded(usable.mean()),"grid":dims.public()}


def register_surveys(app,library):
    @app.post("/api/panels/<panel_id>/surveys")
    def store_survey(panel_id):
        doc=_json()
        name,structure=parse_survey(doc)
        uid=doc.get("id")
        revision=None
        if uid is not None:
            uid=ident(uid,"Survey")
            revision=integer(doc.get("revision"),"Survey revision",1,2_000_000)
        result=library.save_survey(ident(panel_id),name,{"marks":[mark.public() for mark in structure]},uid,revision)
        return {"survey":result},201 if uid is None else 200


def _validate_selection(tool,order):
    if tool.input_count and len(order.capture_ids)!=tool.input_count:
        count=tool.input_count
        raise Problem(f"Select exactly {count} capture{'s' if count!=1 else ''}.")


def register_tools(app,library,executor):
    @app.get("/api/tools")
    def list_tools():
        return {"tools":[tool.public() for tool in TOOLS]}

    @app.post("/api/panels/<panel_id>/tools/<kind>")
    def inspect_panel(panel_id,kind):
        pid=ident(panel_id)
        tool=tool_by_key(kind)
        payload=_json()
        order=WorkOrder.parse(payload)
        _validate_selection(tool,order)
        hold=bool_value(payload.get("save",False),"Save inspection")
        with executor:
            context=Bench.load(library,pid,order)
            outcome=tool.run(context)
            lineage=context.facts()
            checksum=outcome.fingerprint(lineage)
            if not hold:
                return outcome.presentation(lineage,checksum)
            if payload.get("preview")!=checksum:
                raise Problem("The observed result has changed. Run the preview again before retaining it.",409)
            filename=label(payload.get("name",tool.title),"Inspection name")
            receipt=library.save_report(pid,filename,outcome,lineage,checksum)
        return {"record":_build_record(receipt)},201

    @app.get("/api/inspections/<report_id>")
    def read_inspection(report_id):
        return {"record":_build_record(library.report(ident(report_id)))}

    @app.get("/api/inspections/<report_id>/files/<key>")
    def evidence_file(report_id,key):
        asset,raw=library.report_asset(ident(report_id),_expect_key(key))
        download=asset["mime"]!="image/png" or request.args.get("download")=="1"
        return _deliver(raw,asset["mime"],asset["filename"],download)

    @app.get("/api/inspections/<report_id>/source-audit")
    def audit_sources(report_id):
        return inspection_sources(library,ident(report_id))

    @app.put("/api/inspections/<report_id>/review")
    def review_report(report_id):
        body=_json()
        keys(body,"revision","state","name","note")
        revision=integer(body.get("revision"),"Record revision",1,2_000_000)
        evaluation={"state":option(body.get("state"),"Review status",("observed","confirmed","return")),
                    "by":label(body.get("name"),"Reviewer",80),
                    "note":label(body.get("note",""),"Reviewer note",560,True)}
        result=library.report_review(ident(report_id),revision,evaluation)
        return {"record":_build_record(result)}

    @app.get("/api/inspections/<report_id>/packet")
    def download_packet(report_id):
        rid=ident(report_id)
        original=request.args.get("originals","1")
        if original not in ("0","1"):
            raise Problem("Original captures must be included or omitted.")
        with executor:
            contents=archive_inspection(library,rid,original=="1")
        return _deliver(contents,"application/zip",f"vitrail-{rid[:9]}.zip",True)

    @app.post("/api/packets/verify")
    def check_packet():
        _,raw=_file("archive",MAX_ARCHIVE)
        with executor:
            result=verify_packet(raw)
        return result

    @app.get("/api/inspections/<report_id>/sheet")
    def download_sheet(report_id):
        record=library.report(ident(report_id))
        paper=request.args.get("paper","a4")
        orientation=request.args.get("orientation","portrait")
        asset=request.args.get("asset")
        with executor:
            raw=print_inspection(library,record,paper,orientation,_expect_key(asset) if asset else None)
        return _deliver(raw,"application/pdf",f"inspection-{record['id'][:8]}.pdf",True)

    @app.get("/api/panels/<panel_id>/inspection-index")
    def download_index(panel_id):
        pid=ident(panel_id)
        panel_data=library.panel_summary(pid)
        recent=panel_data["reports"]
        requested=request.args.getlist("record")
        if len(requested)>25 or len(requested)!=len(set(requested)):
            raise Problem("Select up to twenty-five different inspections.")
        if requested:
            selected=[ident(token,"Inspection") for token in requested]
            belong={row["id"] for row in recent}
            if set(selected)-belong:
                raise Problem("Index can only include this panel's inspections.",409)
        else:
            selected=[item["id"] for item in recent[:18]]
        paper=request.args.get("paper","a4")
        orientation=request.args.get("orientation","portrait")
        with executor:
            reports=[library.report(rid) for rid in selected]
            raw=archive_index_pdf(library,panel_data["panel"],reports,paper,orientation)
        return _deliver(raw,"application/pdf",f"inspection-index-{pid[:8]}.pdf",True)

    @app.post("/api/scale-target")
    def prepare_target():
        body=_json()
        keys(body,"width_mm","height_mm","ticks_mm","name")
        raw=scale_target(body.get("width_mm"),body.get("height_mm"),body.get("ticks_mm",5),body.get("name","Panel reference"))
        return _deliver(raw,"image/svg+xml","vitrail-scale-target.svg",True)


def create_app(data_dir=None):
    app=Flask(__name__,static_folder=str(STATIC),static_url_path="/assets")
    app.config["MAX_CONTENT_LENGTH"]=54_000_000
    app.config["MAX_FORM_PARTS"]=16
    app.config["MAX_FORM_MEMORY_SIZE"]=550_000
    storage=Library(data_dir or os.environ.get("VITRAIL_DATA","./vitrail-data"))
    executor=WorkBudget()
    app.extensions["vitrail_library"]=storage
    register_library(app,storage,executor)
    register_surveys(app,storage)
    register_tools(app,storage,executor)

    @app.errorhandler(Problem)
    def handle_problem(error):
        response={"error":str(error)}
        if error.field:
            response["field"]=error.field
        return response,error.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return {"error":"Upload exceeds the local file limit."},413

    @app.get("/")
    def application():
        response=send_from_directory(STATIC,"index.html",max_age=0)
        response.headers["Content-Security-Policy"]="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.get("/api/health")
    def healthy():
        with storage.connection() as conn:
            conn.execute("SELECT 1 FROM panels LIMIT 1").fetchall()
        return {"status":"ok","product":"Vitrail Bench","python":sys.version.split()[0]}

    return app
