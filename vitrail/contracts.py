from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
import json
import math
import re

import numpy as np


class Problem(Exception):
    def __init__(self, message, status=400, field=None):
        super().__init__(message)
        self.status = status
        self.field = field


def object_data(value, name="Request"):
    if not isinstance(value, dict):
        raise Problem(f"{name} must be an object.")
    return value


def keys(value, *allowed):
    extras = sorted(set(value) - set(allowed))
    if extras:
        raise Problem(f"Unknown field: {extras[0]}.", field=extras[0])


def label(value, name, limit=120, optional=False):
    if not isinstance(value, str):
        raise Problem(f"{name} must be text.")
    value = value.strip()
    if (not optional and not value) or len(value) > limit:
        raise Problem(f"{name} must contain {'up to' if optional else '1–'} {limit} characters.")
    if any(ord(char) < 32 and char not in "\r\n\t" for char in value):
        raise Problem(f"{name} has a control character.")
    return value


def scalar(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Problem(f"{name} must be a real number.")
    if not math.isfinite(value) or not lower <= value <= upper:
        raise Problem(f"{name} must be between {lower} and {upper}.")
    return float(value)


def integer(value, name, lower, upper):
    if not isinstance(value, int) or isinstance(value, bool):
        raise Problem(f"{name} must be an integer.")
    if not lower <= value <= upper:
        raise Problem(f"{name} must be between {lower} and {upper}.")
    return value


def bool_value(value, name):
    if not isinstance(value, bool):
        raise Problem(f"{name} must be true or false.")
    return value


def option(value, name, choices):
    if value not in choices:
        raise Problem(f"{name} must be one of {', '.join(choices)}.")
    return value


def ident(value, name="ID"):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise Problem(f"{name} is invalid.", 404)
    return value


def digest(contents):
    return sha256(contents).hexdigest()


def frozen_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def safe_filename(value):
    if not isinstance(value, str):
        raise Problem("Filename must be text.")
    candidate = PurePosixPath(value.replace("\\", "/")).name
    if not candidate or candidate in (".", "..") or len(candidate.encode()) > 180:
        raise Problem("Invalid image filename.")
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        raise Problem("Invalid character in filename.")
    return candidate


def pair(value, name="Position", lower=0, upper=1):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise Problem(f"{name} must contain x and y.")
    return tuple(scalar(item, name, lower, upper) for item in value)


def rounded(value, decimals=5):
    value = float(value)
    return round(value, decimals) if math.isfinite(value) else None


@dataclass(frozen=True)
class Box:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def parse(cls, value=None, name="Region"):
        if value is None:
            return cls(0, 0, 1, 1)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise Problem(f"{name} needs left, top, right and bottom.")
        box = cls(*(scalar(item, name, 0, 1) for item in value))
        if box.right <= box.left or box.bottom <= box.top:
            raise Problem(f"{name} must have positive area.")
        return box

    def public(self):
        return [self.left, self.top, self.right, self.bottom]

    def indices(self, width, height):
        x0 = min(width - 1, math.floor(width * self.left))
        y0 = min(height - 1, math.floor(height * self.top))
        x1 = max(x0 + 1, min(width, math.ceil(width * self.right)))
        y1 = max(y0 + 1, min(height, math.ceil(height * self.bottom)))
        return x0, y0, x1, y1

    def slice(self, array):
        height, width = array.shape[:2]
        x0, y0, x1, y1 = self.indices(width, height)
        return array[y0:y1, x0:x1]

    def mask(self, width, height):
        mask = np.zeros((height, width), dtype=bool)
        x0, y0, x1, y1 = self.indices(width, height)
        mask[y0:y1, x0:x1] = True
        return mask


@dataclass(frozen=True)
class PanelSpec:
    title: str
    accession: str
    width_mm: float
    height_mm: float
    note: str
    synthetic: bool

    @classmethod
    def parse(cls, raw, existing=None):
        value = object_data(raw, "Window panel")
        keys(value, "title", "accession", "width_mm", "height_mm", "note", "synthetic")
        previous = existing or {}
        return cls(
            label(value.get("title", previous.get("title")), "Panel title"),
            label(value.get("accession", previous.get("accession", "")), "Accession", 72, True),
            scalar(value.get("width_mm", previous.get("width_mm")), "Width in mm", 8, 3000),
            scalar(value.get("height_mm", previous.get("height_mm")), "Height in mm", 8, 3000),
            label(value.get("note", previous.get("note", "")), "Notes", 900, True),
            bool_value(value.get("synthetic", previous.get("synthetic", False)), "Synthetic"),
        )

    def public(self):
        return dict(vars(self))


CAPTURE_ROLES = ("transmitted", "flat", "reflected", "comparison")


@dataclass(frozen=True)
class PhotoSpec:
    title: str
    note: str
    role: str
    exposure_ev: float
    black_code: float
    illuminant: str

    @classmethod
    def parse(cls, raw, existing=None, filename=None):
        value = object_data(raw, "Capture")
        keys(value, "title", "note", "role", "exposure_ev", "black_code", "illuminant")
        previous = existing or {}
        return cls(
            label(value.get("title", previous.get("title", filename)), "Capture name"),
            label(value.get("note", previous.get("note", "")), "Capture notes", 700, True),
            option(value.get("role", previous.get("role", "transmitted")), "Capture kind", CAPTURE_ROLES),
            scalar(value.get("exposure_ev", previous.get("exposure_ev", 0)), "Exposure EV", -6, 6),
            scalar(value.get("black_code", previous.get("black_code", 0)), "Black code", 0, 0.35),
            label(value.get("illuminant", previous.get("illuminant", "")), "Lighting label", 72, True),
        )

    def public(self):
        return dict(vars(self))


@dataclass(frozen=True)
class WorkOrder:
    capture_ids: tuple
    region: Box
    toolbox: dict
    corrections: dict
    resolution: int

    @classmethod
    def parse(cls, raw):
        doc = object_data(raw, "Inspection")
        keys(doc, "captures", "region", "options", "registrations", "resolution",
             "save", "name", "preview")
        ids = doc.get("captures")
        if not isinstance(ids, list) or not 1 <= len(ids) <= 5:
            raise Problem("Choose one to five captures.")
        ids = tuple(ident(rec, "Capture") for rec in ids)
        if len(set(ids)) != len(ids):
            raise Problem("A capture may occur only once.")
        revisions = object_data(doc.get("registrations", {}), "Registrations")
        if set(revisions) - set(ids):
            raise Problem("Select registrations only for the requested photographs.")
        registry = {key: ident(value, "Registration") for key, value in revisions.items()}
        return cls(ids, Box.parse(doc.get("region")), object_data(doc.get("options", {}), "Options"),
                   registry, integer(doc.get("resolution", 700), "Preview long edge", 160, 1120))

    def public(self):
        return dict(captures=list(self.capture_ids), region=self.region.public(),
                    options=self.toolbox, registrations=self.corrections, resolution=self.resolution)
