from io import BytesIO
from pathlib import PurePosixPath
from tempfile import SpooledTemporaryFile
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo
import json
import re

from .contracts import Problem, digest, frozen_json


SCHEMA = "vitrail-evidence/1"
MAX_ARCHIVE = 50_000_000
MAX_UNPACKED = 78_000_000
MAX_ITEMS = 90
MAX_PART = 30_000_000


def _safe_entry(value):
    if not isinstance(value,str) or not value.isascii() or len(value)>150 or "\\" in value:
        raise Problem("Malformed archive paths.")
    p=PurePosixPath(value)
    if p.is_absolute() or p.as_posix()!=value or ".." in p.parts or value.endswith("/"):
        raise Problem("Archive paths use relative files only.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9./_-]*",value):
        raise Problem("Archive path contains unsupported characters.")
    return value


def _file_alias(record):
    ext={"image/png":"png","text/csv":"csv","image/svg+xml":"svg","application/pdf":"pdf"}[record["mime"]]
    if not re.fullmatch(r"[a-z0-9-]{1,80}",record["key"]):
        raise Problem("An evidence image has an invalid key.",500)
    return f"results/{record['key']}.{ext}"


def _camera_alias(metadata):
    suffix={"PNG":"png","JPEG":"jpg","TIFF":"tif","WEBP":"webp"}[metadata["image"]["format"]]
    return f"captures/{metadata['id']}.{suffix}"


def _line(path,contents,role,link):
    return {"path":path,"sha256":digest(contents),"bytes":len(contents),"role":role,"ref":link}


def _writer(z, path, contents):
    path=_safe_entry(path)
    template=ZipInfo(path)
    template.date_time=(1980,1,1,0,0,0)
    template.external_attr=0o100644<<16
    template.compress_type=ZIP_STORED if path.endswith((".png",".jpg",".webp",".zip")) else ZIP_DEFLATED
    z.writestr(template,contents,compresslevel=5)


def _manifest(report,entries,originals):
    commitment_fields=("id","panel_id","kind","name","created","fingerprint")
    review={key:report["review"].get(key) for key in ("state","by","note","at")}
    return {"schema":SCHEMA,"record":{key:report[key] for key in commitment_fields},
            "review":review,"source_capture_bytes_included":originals,
            "observations":report["facts"],"files":entries}


def archive_inspection(library,report_id,include_sources=True):
    report=library.report(report_id)
    rows=[]
    total=0
    with SpooledTemporaryFile(max_size=5_000_000) as spooled:
        with ZipFile(spooled,"w",allowZip64=False) as z:
            details=frozen_json({"kind":report["kind"],"result":report["result"],"fingerprint":report["fingerprint"]})
            rows.append(_line("results/readout.json",details,"report",report_id))
            _writer(z,"results/readout.json",details)
            total+=len(details)
            for meta in report["assets"]:
                contents=library.blobs.read(meta["sha256"])
                path=_file_alias(meta)
                total+=len(contents)
                if total>MAX_UNPACKED:
                    raise Problem("This packet exceeds the export budget.",413)
                rows.append(_line(path,contents,"analysis",meta["key"]))
                _writer(z,path,contents)
            if include_sources:
                for photo in report["facts"]["sources"]:
                    contents=library.blobs.read(photo["original"])
                    path=_camera_alias(photo)
                    total+=len(contents)
                    if total>MAX_UNPACKED:
                        raise Problem("Source photographs exceed the packet budget. Export an evidence-only packet.",413)
                    rows.append(_line(path,contents,"source",photo["id"]))
                    _writer(z,path,contents)
            manifest=frozen_json(_manifest(report,rows,include_sources))
            _writer(z,"manifest.json",manifest)
        length=spooled.tell()
        if length>MAX_ARCHIVE:
            raise Problem("Packet is larger than 50 MB. Export without original captures.",413)
        spooled.seek(0)
        return spooled.read()


def _content(z,entry):
    if entry.file_size>MAX_PART:
        raise Problem("One archive item is too large.",413)
    with z.open(entry) as stream:
        data=stream.read(MAX_PART+1)
    if len(data)!=entry.file_size:
        raise Problem("An archive part could not be read in full.")
    return data


def _doc(payload, where):
    try:
        value=json.loads(payload)
    except (UnicodeError,ValueError,RecursionError) as error:
        raise Problem(f"{where} is not valid JSON.") from error
    if not isinstance(value,dict):
        raise Problem(f"{where} must be an object.")
    return value


def _index_archive(z):
    items=z.infolist()
    if not 3<=len(items)<=MAX_ITEMS:
        raise Problem(f"An evidence archive requires 3–{MAX_ITEMS} files.")
    if sum(item.file_size for item in items)>MAX_UNPACKED:
        raise Problem("Unpacked packet exceeds the local read budget.",413)
    paths={}
    for item in items:
        name=_safe_entry(item.filename)
        if name in paths or item.flag_bits&1 or item.is_dir():
            raise Problem("The archive contains duplicate paths, directories or encryption.")
        if item.compress_type not in (ZIP_STORED,ZIP_DEFLATED):
            raise Problem("Archive compression is not supported.")
        paths[name]=item
    if "manifest.json" not in paths:
        raise Problem("Evidence archive has no manifest.")
    return paths


def _manifest_rows(manifest):
    if manifest.get("schema")!=SCHEMA:
        raise Problem("Evidence packet schema is not supported.")
    rows=manifest.get("files")
    if not isinstance(rows,list) or not 2<=len(rows)<MAX_ITEMS:
        raise Problem("Invalid manifest file listing.")
    files={}
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"path","sha256","bytes","role","ref"}:
            raise Problem("Invalid manifest row.")
        path=_safe_entry(row["path"])
        if path=="manifest.json" or path in files:
            raise Problem("Invalid manifest filename.")
        sha=row["sha256"]
        if not isinstance(sha,str) or not re.fullmatch(r"[a-f0-9]{64}",sha):
            raise Problem("A manifest SHA-256 is malformed.")
        if not isinstance(row["bytes"],int) or isinstance(row["bytes"],bool) or not 0<row["bytes"]<=MAX_PART:
            raise Problem("A manifest length is malformed.")
        role=row["role"]
        if not isinstance(role,str) or role not in ("report","analysis","source"):
            raise Problem("An archive role is invalid.")
        if not isinstance(row["ref"],str) or not row["ref"]:
            raise Problem("An archive part is missing its reference.")
        files[path]=row
    return files


def _capture_refs(manifest,rows):
    observations=manifest.get("observations")
    if not isinstance(observations,dict) or not isinstance(observations.get("sources"),list):
        raise Problem("Manifest is missing the selected capture references.")
    sources={}
    for capture in observations["sources"]:
        if not isinstance(capture,dict) or not re.fullmatch(r"[a-f0-9]{32}",str(capture.get("id"))) or not re.fullmatch(r"[a-f0-9]{64}",str(capture.get("original"))):
            raise Problem("A capture reference has invalid identity or checksum.")
        if capture["id"] in sources:
            raise Problem("Repeated source capture reference.")
        sources[capture["id"]]=capture["original"]
    included=manifest.get("source_capture_bytes_included")
    if not isinstance(included,bool):
        raise Problem("Manifest does not say whether camera bytes were included.")
    tally={}
    for name,part in rows.items():
        if part["role"]!="source":
            continue
        token=part["ref"]
        if not name.startswith("captures/") or token in tally or sources.get(token)!=part["sha256"]:
            raise Problem("Camera capture receipt does not match its archive entry.")
        tally[token]=name
    if (included and set(tally)!=set(sources)) or (not included and tally):
        raise Problem("Source camera listing does not match the packet scope.")
    return len(sources),len(tally)


def verify_packet(data):
    if not data or len(data)>MAX_ARCHIVE:
        raise Problem("Upload an evidence archive with at most 50 MB.",413)
    try:
        with ZipFile(BytesIO(data)) as z:
            listing=_index_archive(z)
            manifest=_doc(_content(z,listing["manifest.json"]),"Manifest")
            rows=_manifest_rows(manifest)
            if set(rows)!=set(listing)-{"manifest.json"}:
                raise Problem("Archive inventory differs from the manifest.")
            source_total,included=_capture_refs(manifest,rows)
            if "results/readout.json" not in rows or rows["results/readout.json"]["role"]!="report":
                raise Problem("The evidence archive lacks its report data.")
            if any(not name.startswith("results/") for name,part in rows.items() if part["role"]=="analysis"):
                raise Problem("Result artwork appears outside the designated archive folder.")
            record=manifest.get("record")
            if not isinstance(record,dict) or not {"id","kind","name","fingerprint"}<=set(record):
                raise Problem("Manifest report identity is missing.")
            if rows["results/readout.json"]["ref"]!=record["id"]:
                raise Problem("Report data points to a different inspection.")
            counts={"report":0,"analysis":0,"source":0}
            for path,promised in rows.items():
                payload=_content(z,listing[path])
                if len(payload)!=promised["bytes"] or digest(payload)!=promised["sha256"]:
                    raise Problem(f"Packet hash or length does not match: {path}.",409)
                counts[promised["role"]]+=1
                if path=="results/readout.json":
                    readout=_doc(payload,"Inspection")
                    if readout.get("kind")!=record["kind"] or readout.get("fingerprint")!=record["fingerprint"]:
                        raise Problem("The manifest and report identity disagree.",409)
            if counts["report"]!=1 or counts["analysis"]<1:
                raise Problem("The packet should have one report and at least one evidence file.")
    except (RuntimeError,BadZipFile,OSError,NotImplementedError) as error:
        raise Problem("The archive could not be opened as a supported evidence packet.") from error
    return {"valid":True,"schema":SCHEMA,"report":record,"files":len(rows),"counts":counts,
            "cited_captures":source_total,"source_photographs":included,
            "scope":"Checks archive contents against its own manifest. It does not authenticate the author."}
