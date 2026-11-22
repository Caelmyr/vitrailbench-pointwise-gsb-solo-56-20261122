from dataclasses import dataclass
from threading import BoundedSemaphore
import base64
import numpy as np
from scipy.ndimage import gaussian_filter

from .camera import Alignment, PhysicalGrid, brightness, open_camera, sample_rgb, srgb_to_linear
from .contracts import Problem, digest, frozen_json, rounded


@dataclass(frozen=True)
class Frame:
    record: dict
    alignment: Alignment
    alignment_ref: dict | None
    rgb: np.ndarray
    code: np.ndarray
    present: np.ndarray
    clipped: np.ndarray

    @property
    def signal(self):
        return brightness(self.rgb)

    @property
    def measurable(self):
        return self.present & ~self.clipped

    def describe(self, context):
        info = self.record
        region = context.selection
        alpha = self.present & region
        clipped = self.clipped & region
        n = max(1, int(region.sum()))
        return dict(id=info["id"], title=info["data"]["title"], original=info["original"],
                    revision=info["revision"], image=info["image"], metadata=info["data"],
                    observed_fraction=rounded(alpha.sum()/n), clipped_fraction=rounded(clipped.sum()/n),
                    alignment=self.alignment.public(),
                    alignment_id=self.alignment_ref["id"] if self.alignment_ref else None)


def load_frame(photo, reg, camera, grid):
    if reg:
        basic_alignment = {k: reg["data"][k] for k in ("title", "corners", "model")}
        alignment = Alignment.parse(basic_alignment)
    else:
        alignment = Alignment.identity()
    code, present = sample_rgb(camera, alignment.to_camera(grid))
    clipped = present & np.any(code >= 0.993, axis=2)
    black = srgb_to_linear(photo["data"]["black_code"])
    rgb = np.maximum(srgb_to_linear(code) - black, 0) / 2 ** photo["data"]["exposure_ev"]
    return Frame(photo, alignment, reg, rgb.astype(np.float32), code, present, clipped)


@dataclass(frozen=True)
class Bench:
    panel: dict
    order: object
    scale: PhysicalGrid
    selection: np.ndarray
    frames: tuple

    @classmethod
    def load(cls, library, panel_id, order):
        panel, records = library.snapshot(panel_id, order)
        original_pixels = sum(row[0]["image"]["width"]*row[0]["image"]["height"] for row in records)
        if original_pixels > 34_000_000:
            raise Problem("Select fewer captures. Active inputs exceed 34 million camera pixels.", 413)
        cameras = [open_camera(contents) for _, _, contents in records]
        scale = PhysicalGrid.from_panel(panel, cameras, order.resolution)
        grid = scale.grid()
        frames = tuple(load_frame(record[0], record[1], camera, grid)
                       for record, camera in zip(records, cameras, strict=True))
        return cls(panel, order, scale, order.region.mask(scale.width_px, scale.height_px), frames)

    def exactly(self, count):
        if len(self.frames) != count:
            raise Problem(f"This inspection requires exactly {count} capture{'s' if count != 1 else ''}.")
        return self.frames

    def one(self):
        return self.exactly(1)[0]

    def two(self):
        return self.exactly(2)

    def measured(self, frame, *, allow_clipped=False):
        support = frame.present if allow_clipped else frame.measurable
        return self.selection & support

    def shared(self, allow_clipped=False):
        common = self.selection.copy()
        for frame in self.frames:
            common &= frame.present if allow_clipped else frame.measurable
        return common

    def demand(self, usable, floor=0.03, pixels=36):
        threshold = max(pixels, np.ceil(floor * self.selection.sum()))
        if usable.sum() < threshold:
            raise Problem("The selected window has insufficient measured pixels for this operation.", 422)
        return usable

    def facts(self):
        names = ("id", "title", "accession", "width_mm", "height_mm", "revision", "synthetic")
        return dict(panel={key: self.panel[key] for key in names}, parameters=self.order.fingerprint_inputs(),
                    scale=self.scale.public(), sources=[frame.describe(self) for frame in self.frames])


def masked_blur(field, valid, sigma, floor=0.4):
    weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="constant")
    if field.ndim == 2:
        signal = gaussian_filter(np.where(valid, field, 0), sigma=sigma, mode="constant")
        return signal / np.maximum(weights, 0.0001), weights >= floor
    planes = [gaussian_filter(np.where(valid, plane, 0), sigma=sigma, mode="constant")
              / np.maximum(weights, 0.0001) for plane in np.moveaxis(field, -1, 0)]
    return np.stack(planes, axis=-1), weights >= floor


def percentiles(values, allowed, requested=(5, 25, 50, 75, 95), floor=1):
    check = np.asarray(values[allowed])
    check = check[np.isfinite(check)]
    if check.size < floor:
        return [None] * len(requested)
    return [rounded(value) for value in np.percentile(check, requested)]


def ratio(mask, reference):
    return rounded(mask.sum() / max(1, reference.sum()), 6)


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    mime: str
    payload: bytes
    source_box: tuple | None = None

    def meta(self):
        data = dict(key=self.key, filename=self.name, mime=self.mime, sha256=digest(self.payload), bytes=len(self.payload))
        if self.source_box is not None:
            data["panel_box"] = list(self.source_box)
        return data

    def persist(self, storage):
        attrs = self.meta()
        attrs["sha256"] = storage.write(self.payload)
        return attrs

    def presentation(self):
        attrs = self.meta()
        attrs["data"] = base64.b64encode(self.payload).decode()
        return attrs


@dataclass(frozen=True)
class Result:
    kind: str
    title: str
    summary: dict
    details: dict
    assets: tuple

    def evidence_digest(self, facts, exempt=frozenset()):
        # Determination annotations are excluded: they record a human decision about unchanged
        # evidence, so retaining them must still match the original preview fingerprint.
        strip = lambda value: {key: item for key, item in value.items() if key not in exempt}
        return digest(frozen_json({"kind": self.kind, "summary": strip(self.summary),
                                   "details": strip(self.details), "inputs": facts,
                                   "assets": [part.meta() for part in self.assets]}))

    def determination_digest(self, evidence, exempt=frozenset()):
        decision = {key: self.details.get(key) for key in sorted(exempt) if key in self.details}
        return digest(evidence.encode() + frozen_json({"determination": decision}))

    def presentation(self, facts, fingerprint):
        return dict(kind=self.kind, title=self.title, summary=self.summary, result=self.details,
                    facts=facts, fingerprint=fingerprint,
                    assets=[part.presentation() for part in self.assets])


class WorkBudget:
    def __init__(self, slots=2):
        self.limiter = BoundedSemaphore(slots)

    def __enter__(self):
        if not self.limiter.acquire(blocking=False):
            raise Problem("Two local image operations are in progress. Try again after one finishes.", 429)
        return self

    def __exit__(self, typ, value, trace):
        self.limiter.release()
