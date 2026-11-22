from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4
import json
import sqlite3

from .camera import Alignment, open_camera
from .contracts import Problem, digest, frozen_json, ident, label, safe_filename


SCHEMA = """
CREATE TABLE IF NOT EXISTS panels(
    id TEXT PRIMARY KEY, data TEXT NOT NULL, created TEXT NOT NULL, changed TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1, sample_key TEXT UNIQUE
);
CREATE TABLE IF NOT EXISTS photos(
    id TEXT PRIMARY KEY, panel_id TEXT NOT NULL REFERENCES panels(id), filename TEXT NOT NULL,
    original TEXT NOT NULL, image TEXT NOT NULL, data TEXT NOT NULL, created TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1, active_alignment TEXT
);
CREATE INDEX IF NOT EXISTS photo_panel ON photos(panel_id, created, id);
CREATE TABLE IF NOT EXISTS alignments(
    id TEXT PRIMARY KEY, photo_id TEXT NOT NULL REFERENCES photos(id),
    photo_revision INTEGER NOT NULL, data TEXT NOT NULL, created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alignment_photo ON alignments(photo_id, created, id);
CREATE TABLE IF NOT EXISTS reports(
    id TEXT PRIMARY KEY, panel_id TEXT NOT NULL REFERENCES panels(id), kind TEXT NOT NULL,
    name TEXT NOT NULL, result TEXT NOT NULL, facts TEXT NOT NULL, assets TEXT NOT NULL,
    created TEXT NOT NULL, review TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
    fingerprint TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS report_panel ON reports(panel_id, created, id);
CREATE TABLE IF NOT EXISTS surveys(
    id TEXT PRIMARY KEY, panel_id TEXT NOT NULL REFERENCES panels(id),
    name TEXT NOT NULL, geometry TEXT NOT NULL, created TEXT NOT NULL,
    edited TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS survey_panel ON surveys(panel_id, created, id);
CREATE TABLE IF NOT EXISTS actions(
    number INTEGER PRIMARY KEY, panel_id TEXT NOT NULL REFERENCES panels(id),
    at TEXT NOT NULL, care TEXT NOT NULL, target TEXT NOT NULL, detail TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS action_panel ON actions(panel_id, number);
"""

PHOTO_LIMIT = 30
REPORT_LIMIT = 60
ALIGNMENT_LIMIT = 20
SURVEY_LIMIT = 30


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_id():
    return uuid4().hex


def encode(value):
    return frozen_json(value).decode()


def unpack(row, names):
    if row is None:
        return None
    decoded = dict(row)
    for name in names:
        decoded[name] = json.loads(decoded[name])
    return decoded


def record_action(conn, panel_id, kind, target, detail):
    conn.execute("INSERT INTO actions(panel_id, at, care, target, detail) VALUES(?, ?, ?, ?, ?)",
                 (panel_id, utc(), kind, target, encode(detail)))


class BlobTree:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def location(self, checksum):
        if len(checksum) != 64 or any(ch not in "1234567890abcdef" for ch in checksum):
            raise Problem("Stored asset reference is malformed.", 500)
        return self.root / checksum[:2] / checksum[2:]

    def write(self, data):
        checksum = digest(data)
        target = self.location(checksum)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=target.parent, prefix=".incoming-", delete=False) as output:
                staging = Path(output.name)
                output.write(data)
            try:
                staging.replace(target)
            finally:
                staging.unlink(missing_ok=True)
        return checksum

    def read(self, checksum):
        try:
            data = self.location(checksum).read_bytes()
        except FileNotFoundError as error:
            raise Problem("A stored asset has gone missing.", 410) from error
        if digest(data) != checksum:
            raise Problem("Stored asset has changed on disk.", 409)
        return data


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "vitrail.sqlite"
        self.blobs = BlobTree(self.root / "blobs")
        with self.connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    @contextmanager
    def connection(self, write=False):
        conn = sqlite3.connect(self.db, timeout=12)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _panel(conn, panel_id):
        row = conn.execute("SELECT * FROM panels WHERE id=?", (panel_id,)).fetchone()
        if row is None:
            raise Problem("Panel not found.", 404)
        data = unpack(row, ("data",))
        return {**data.pop("data"), **data}

    @staticmethod
    def _photo(conn, photo_id):
        row = conn.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()
        if row is None:
            raise Problem("Capture not found.", 404)
        return unpack(row, ("data", "image"))

    @staticmethod
    def _alignment(conn, photo_id, alignment_id):
        if alignment_id is None:
            return None
        row = conn.execute("SELECT * FROM alignments WHERE photo_id=? AND id=?", (photo_id, alignment_id)).fetchone()
        if row is None:
            raise Problem("Alignment does not belong to this capture.", 404)
        return unpack(row, ("data",))

    @staticmethod
    def _report(conn, report_id):
        row = conn.execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
        if row is None:
            raise Problem("Inspection record not found.", 404)
        return unpack(row, ("result", "facts", "assets", "review"))

    @staticmethod
    def _survey(conn, panel_id, survey_id):
        row = conn.execute("SELECT * FROM surveys WHERE panel_id=? AND id=?", (panel_id, survey_id)).fetchone()
        if row is None:
            raise Problem("Survey not found for this panel.", 404)
        return unpack(row, ("geometry",))

    def _panel_count(self, conn):
        return conn.execute("SELECT count(*) FROM panels").fetchone()[0]

    def panel_list(self):
        with self.connection() as conn:
            rows = conn.execute("""
                SELECT p.*, (SELECT COUNT(*) FROM photos WHERE panel_id=p.id) photos,
                    (SELECT COUNT(*) FROM reports WHERE panel_id=p.id) reports
                FROM panels p ORDER BY changed DESC, p.id LIMIT 250
            """).fetchall()
            return [{**json.loads(row["data"]), "id": row["id"], "changed": row["changed"],
                     "photos": row["photos"], "reports": row["reports"]} for row in rows]

    def create_panel(self, spec, sample_key=None):
        created = utc()
        with self.connection(True) as conn:
            if sample_key is not None:
                previous = conn.execute("SELECT id FROM panels WHERE sample_key=?", (sample_key,)).fetchone()
                if previous:
                    return self._panel(conn, previous["id"]), False
            if self._panel_count(conn) >= 250:
                raise Problem("The local collection has reached 250 panels.", 413)
            pid = new_id()
            conn.execute("INSERT INTO panels VALUES(?, ?, ?, ?, 1, ?)",
                         (pid, encode(spec.public()), created, created, sample_key))
            record_action(conn, pid, "panel-added", pid, {"synthetic": spec.synthetic})
            return self._panel(conn, pid), True

    def load_sample(self, spec, sample_key, photos, surveys):
        # Decode before entering the write lock; panel and all its references commit together.
        with self.connection() as conn:
            row = conn.execute("SELECT id FROM panels WHERE sample_key=?", (sample_key,)).fetchone()
            if row:
                return self._panel(conn, row["id"]), False
        prepared = [(safe_filename(filename), self.blobs.write(contents), open_camera(contents).info(), spec)
                    for filename, contents, spec in photos]
        created = utc()
        with self.connection(True) as conn:
            row = conn.execute("SELECT id FROM panels WHERE sample_key=?", (sample_key,)).fetchone()
            if row:
                return self._panel(conn, row["id"]), False
            if self._panel_count(conn) >= 250:
                raise Problem("The local collection has reached 250 panels.", 413)
            pid = new_id()
            conn.execute("INSERT INTO panels VALUES(?, ?, ?, ?, 1, ?)",
                         (pid, encode(spec.public()), created, created, sample_key))
            for filename, checksum, image, photo_spec in prepared:
                cap = new_id()
                conn.execute("INSERT INTO photos VALUES(?, ?, ?, ?, ?, ?, ?, 1, NULL)",
                             (cap, pid, filename, checksum, encode(image), encode(photo_spec.public()), created))
            for name, geometry in surveys:
                conn.execute("INSERT INTO surveys VALUES(?, ?, ?, ?, ?, ?, 1)",
                             (new_id(), pid, name, encode(geometry), created, created))
            record_action(conn, pid, "sample-loaded", pid, {"sample": sample_key, "captures": len(prepared)})
            return self._panel(conn, pid), True

    def panel_summary(self, panel_id):
        with self.connection() as conn:
            panel = self._panel(conn, panel_id)
            photo_rows = conn.execute("SELECT * FROM photos WHERE panel_id=? ORDER BY rowid", (panel_id,)).fetchall()
            report_rows = conn.execute("""
                SELECT id,name,kind,created,review,revision FROM reports WHERE panel_id=?
                ORDER BY created DESC,id
            """, (panel_id,)).fetchall()
            survey_rows = conn.execute("""
                SELECT * FROM surveys WHERE panel_id=? ORDER BY edited DESC,id
            """, (panel_id,)).fetchall()
            actions = conn.execute("""
                SELECT at,care,target,detail FROM actions WHERE panel_id=? ORDER BY number DESC LIMIT 80
            """, (panel_id,)).fetchall()
            return dict(panel=panel, photos=[unpack(row, ("data", "image")) for row in photo_rows],
                        reports=[unpack(row, ("review",)) for row in report_rows],
                        surveys=[unpack(row, ("geometry",)) for row in survey_rows],
                        activity=[unpack(row, ("detail",)) for row in actions])

    def update_panel(self, panel_id, revision, spec):
        with self.connection(True) as conn:
            old = self._panel(conn, panel_id)
            if old["revision"] != revision:
                raise Problem("Panel was changed on another screen. Reload the library.", 409)
            changed = utc()
            conn.execute("UPDATE panels SET data=?, changed=?,revision=revision+1 WHERE id=?",
                         (encode(spec.public()), changed, panel_id))
            record_action(conn, panel_id, "panel-edited", panel_id, {"previous_revision": revision})
            return self._panel(conn, panel_id)

    def add_photo(self, panel_id, filename, payload, spec):
        filename = safe_filename(filename)
        camera = open_camera(payload)
        checksum = self.blobs.write(payload)
        photo_id = new_id()
        created = utc()
        with self.connection(True) as conn:
            self._panel(conn, panel_id)
            current = conn.execute("SELECT count(*) FROM photos WHERE panel_id=?", (panel_id,)).fetchone()[0]
            if current >= PHOTO_LIMIT:
                raise Problem(f"A panel may hold up to {PHOTO_LIMIT} captures.", 413)
            conn.execute("INSERT INTO photos VALUES(?, ?, ?, ?, ?, ?, ?, 1, NULL)",
                         (photo_id, panel_id, filename, checksum, encode(camera.info()), encode(spec.public()), created))
            conn.execute("UPDATE panels SET changed=? WHERE id=?", (created, panel_id))
            record_action(conn, panel_id, "photo-added", photo_id, {"checksum": checksum, "role": spec.role})
            return self._photo(conn, photo_id)

    def update_photo(self, photo_id, revision, spec):
        with self.connection(True) as conn:
            photo = self._photo(conn, photo_id)
            if photo["revision"] != revision:
                raise Problem("Capture settings were changed on another screen.", 409)
            conn.execute("UPDATE photos SET data=?,revision=revision+1 WHERE id=?", (encode(spec.public()), photo_id))
            record_action(conn, photo["panel_id"], "photo-edited", photo_id, {"previous_revision": revision})
            return self._photo(conn, photo_id)

    def photo(self, photo_id):
        with self.connection() as conn:
            return self._photo(conn, photo_id)

    def photo_bytes(self, photo_id):
        photo = self.photo(photo_id)
        return photo, self.blobs.read(photo["original"])

    def alignments(self, photo_id):
        with self.connection() as conn:
            photo = self._photo(conn, photo_id)
            rows = conn.execute("SELECT * FROM alignments WHERE photo_id=? ORDER BY created DESC,id",
                                (photo_id,)).fetchall()
            return dict(active=photo["active_alignment"], revisions=[unpack(row, ("data",)) for row in rows])

    def add_alignment(self, photo_id, revision, alignment):
        with self.connection(True) as conn:
            photo = self._photo(conn, photo_id)
            if photo["revision"] != revision:
                raise Problem("Capture changed while the alignment was being measured.", 409)
            count = conn.execute("SELECT count(*) FROM alignments WHERE photo_id=?", (photo_id,)).fetchone()[0]
            if count >= ALIGNMENT_LIMIT:
                raise Problem(f"A capture can keep up to {ALIGNMENT_LIMIT} alignments.", 413)
            aid = new_id()
            conn.execute("INSERT INTO alignments VALUES(?, ?, ?, ?, ?)",
                         (aid, photo_id, revision, encode(alignment.public()), utc()))
            conn.execute("UPDATE photos SET active_alignment=?, revision=revision+1 WHERE id=?", (aid, photo_id))
            record_action(conn, photo["panel_id"], "alignment-saved", photo_id, {"alignment": aid})
            return self._alignment(conn, photo_id, aid)

    def use_alignment(self, photo_id, revision, alignment_id):
        with self.connection(True) as conn:
            photo = self._photo(conn, photo_id)
            if photo["revision"] != revision:
                raise Problem("Capture settings changed. Reload before restoring an alignment.", 409)
            self._alignment(conn, photo_id, alignment_id)
            if photo["active_alignment"] != alignment_id:
                conn.execute("UPDATE photos SET active_alignment=?, revision=revision+1 WHERE id=?", (alignment_id, photo_id))
                record_action(conn, photo["panel_id"], "alignment-activated", photo_id, {"alignment": alignment_id})
            return self._photo(conn, photo_id)

    def snapshot(self, panel_id, order):
        with self.connection() as conn:
            conn.execute("BEGIN")
            panel = self._panel(conn, panel_id)
            paired = []
            for photo_id in order.capture_ids:
                photo = self._photo(conn, photo_id)
                if photo["panel_id"] != panel_id:
                    raise Problem("All selected photos must belong to the active panel.", 409)
                alignment = self._alignment(conn, photo_id, order.corrections.get(photo_id, photo["active_alignment"]))
                paired.append((photo, alignment))
        return panel, [(photo, alignment, self.blobs.read(photo["original"])) for photo, alignment in paired]

    def save_report(self, panel_id, name, result, facts, signature):
        created = utc()
        choices = [part.persist(self.blobs) for part in result.assets]
        with self.connection(True) as conn:
            panel = self._panel(conn, panel_id)
            if panel["revision"] != facts["panel"]["revision"]:
                raise Problem("The panel dimensions or settings changed. Refresh the inspection.", 409)
            for source in facts["sources"]:
                photo = self._photo(conn, source["id"])
                if photo["revision"] != source["revision"]:
                    raise Problem("A source capture changed. Refresh the inspection before retaining it.", 409)
            count = conn.execute("SELECT count(*) FROM reports WHERE panel_id=?", (panel_id,)).fetchone()[0]
            if count >= REPORT_LIMIT:
                raise Problem(f"A panel may hold up to {REPORT_LIMIT} inspections.", 413)
            rid = new_id()
            review = {"state": "observed", "by": "", "note": "", "at": None}
            conn.execute("INSERT INTO reports VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
                         (rid, panel_id, result.kind, name, encode(result.details), encode(facts), encode(choices),
                          created, encode(review), signature))
            record_action(conn, panel_id, "inspection-retained", rid, {"kind": result.kind})
            return self._report(conn, rid)

    def report(self, report_id):
        with self.connection() as conn:
            return self._report(conn, report_id)

    def report_asset(self, report_id, asset_key):
        record = self.report(report_id)
        for part in record["assets"]:
            if part["key"] == asset_key:
                return part, self.blobs.read(part["sha256"])
        raise Problem("Inspection does not contain that file.", 404)

    def report_review(self, report_id, revision, decision):
        with self.connection(True) as conn:
            record = self._report(conn, report_id)
            if record["revision"] != revision:
                raise Problem("This inspection review changed on another screen.", 409)
            decision = {**decision, "at": utc()}
            conn.execute("UPDATE reports SET review=?,revision=revision+1 WHERE id=?",
                         (encode(decision), report_id))
            record_action(conn, record["panel_id"], "inspection-reviewed", report_id, decision)
            return self._report(conn, report_id)

    def save_survey(self, panel_id, name, geometry, survey_id=None, revision=None):
        with self.connection(True) as conn:
            self._panel(conn, panel_id)
            if survey_id is None:
                count = conn.execute("SELECT count(*) FROM surveys WHERE panel_id=?", (panel_id,)).fetchone()[0]
                if count >= SURVEY_LIMIT:
                    raise Problem(f"A panel may hold up to {SURVEY_LIMIT} surveys.", 413)
                survey_id = new_id()
                at = utc()
                conn.execute("INSERT INTO surveys VALUES(?, ?, ?, ?, ?, ?, 1)",
                             (survey_id, panel_id, name, encode(geometry), at, at))
                action = "survey-created"
            else:
                old = self._survey(conn, panel_id, survey_id)
                if old["revision"] != revision:
                    raise Problem("This survey was edited elsewhere. Reopen its latest revision.", 409)
                conn.execute("UPDATE surveys SET name=?,geometry=?,edited=?,revision=revision+1 WHERE id=?",
                             (name, encode(geometry), utc(), survey_id))
                action = "survey-edited"
            record_action(conn, panel_id, action, survey_id, {"mark_count": len(geometry["marks"])})
            return self._survey(conn, panel_id, survey_id)
