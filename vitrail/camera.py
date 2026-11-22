from dataclasses import dataclass
from io import BytesIO
from warnings import catch_warnings, simplefilter
import math

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy.ndimage import map_coordinates

from .contracts import Problem, keys, label, object_data, pair, rounded


FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff", "WEBP": "image/webp"}
MAX_IMAGE_PIXELS = 16_000_000
MAX_PHOTO_BYTES = 28_000_000
CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))


def srgb_to_linear(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    encoded = np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1 / 2.4) - 0.055)
    return np.rint(255 * encoded).astype(np.uint8)


def brightness(value):
    return np.tensordot(value, np.array([0.2126, 0.7152, 0.0722], np.float32), axes=(-1, 0))


def preview_bytes(rgb, linear=True):
    image = Image.fromarray(linear_to_srgb(rgb) if linear else rgb)
    output = BytesIO()
    image.save(output, format="PNG", compress_level=5)
    return output.getvalue()


def _png_header_depth(payload, kind):
    if kind == "PNG" and len(payload) >= 25 and payload[12:16] == b"IHDR":
        return payload[24]
    return None


@dataclass(frozen=True)
class CameraImage:
    image: Image.Image
    format: str
    mode: str
    native_size: tuple
    bits: int
    profile: str
    orientation: int

    @property
    def size(self):
        return self.image.size

    @property
    def has_alpha(self):
        return "A" in self.image.getbands()

    def info(self):
        return dict(width=self.size[0], height=self.size[1], native_size=list(self.native_size),
                    format=self.format, mode=self.mode, bit_depth=self.bits, profile=self.profile,
                    orientation=self.orientation, has_alpha=self.has_alpha, mime=FORMATS[self.format])

    def samples(self):
        if self.image.mode.startswith("I;16"):
            gray = np.asarray(self.image, dtype=np.float32) / 65535
            rgb = np.repeat(gray[..., None], 3, axis=2)
            alpha = np.ones(gray.shape, dtype=np.float32)
        else:
            rgba = np.asarray(self.image.convert("RGBA"), dtype=np.float32) / 255
            rgb = rgba[..., :3]
            alpha = rgba[..., 3]
        return rgb, alpha

    def preview(self, limit=1000):
        if self.image.mode.startswith("I;16"):
            data = np.asarray(self.image, dtype=np.float32) / 65535
            preview = Image.fromarray(np.rint(data * 255).astype(np.uint8), "L")
        else:
            preview = self.image.copy()
        preview.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        output = BytesIO()
        preview.save(output, format="PNG", compress_level=5)
        return output.getvalue()


def open_camera(payload):
    if not payload or len(payload) > MAX_PHOTO_BYTES:
        raise Problem("Use an image with at most 28 MB.", 413)
    try:
        with catch_warnings():
            simplefilter("error", Image.DecompressionBombWarning)
            source = Image.open(BytesIO(payload))
            kind = source.format
            if kind not in FORMATS:
                raise Problem("Supported captures are PNG, JPEG, TIFF and WebP.")
            if getattr(source, "n_frames", 1) > 1:
                raise Problem("Choose a single frame or page.")
            w, h = source.size
            if min(w, h) < 32 or w * h > MAX_IMAGE_PIXELS:
                raise Problem("Captures require at least 32 px per side and at most 16 megapixels.", 413)
            mode = source.mode
            if mode not in ("RGB", "RGBA", "CMYK", "L", "LA", "P", "1", "I;16", "I;16L", "I;16B"):
                raise Problem("Unsupported camera channel format.")
            orientation = source.getexif().get(274, 1)
            original_bits = _png_header_depth(payload, kind) or (16 if mode.startswith("I;16") else 8)
            if original_bits > 8 and not mode.startswith("I;16"):
                raise Problem("High-bit captures must be grayscale to preserve the pixel codes.")
            icc = source.info.get("icc_profile")
            source = ImageOps.exif_transpose(source)
            source.load()
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise Problem("The camera image could not be decoded.") from error
    if source.mode == "P":
        source = source.convert("RGBA" if "transparency" in source.info else "RGB")
    profile_name = "Untagged sRGB"
    if icc and not source.mode.startswith("I;16"):
        try:
            profile = ImageCms.ImageCmsProfile(BytesIO(icc))
            alpha = source.getchannel("A") if "A" in source.getbands() else None
            component = source if source.mode in ("L", "CMYK") else source.convert("RGB")
            converted = ImageCms.profileToProfile(component, profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
            if alpha is not None:
                converted.putalpha(alpha)
            source = converted
            profile_name = label(ImageCms.getProfileDescription(profile).strip(), "Profile", 300, True) or "Embedded ICC"
        except (ImageCms.PyCMSError, ValueError, OSError) as error:
            raise Problem("The embedded color profile could not be converted.") from error
    elif source.mode == "CMYK":
        raise Problem("A CMYK proof needs its embedded profile.")
    return CameraImage(source, kind, mode, (w, h), original_bits, profile_name, int(orientation) if isinstance(orientation, int) else 1)


def normalized_grid(width, height):
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    return np.stack(np.meshgrid(xs, ys), axis=-1)


def _transformation(points, model):
    a = []
    b = []
    primary = np.array(CORNERS)
    target = np.asarray(points)
    if model == "projective":
        for (u, v), (x, y) in zip(primary, target, strict=True):
            a.extend(((u, v, 1, 0, 0, 0, -u*x, -v*x), (0, 0, 0, u, v, 1, -u*y, -v*y)))
            b.extend((x, y))
        try:
            coeff = np.linalg.solve(a, b)
        except np.linalg.LinAlgError as error:
            raise Problem("The four corners do not determine an alignment.") from error
        return np.array([*coeff, 1.0]).reshape(3, 3)
    design = np.column_stack((primary, np.ones(4)))
    coeff, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < 3:
        raise Problem("Corners do not determine an affine plane.")
    return np.vstack((coeff.T, (0, 0, 1)))


def apply_matrix(matrix, points):
    flat = np.asarray(points, dtype=np.float64)
    aff = flat @ matrix[:, :2].T + matrix[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return aff[..., :2] / aff[..., 2:]


@dataclass
class Alignment:
    title: str
    corners: tuple
    model: str
    matrix: np.ndarray
    local: object = None
    warp: object = None
    inverse_warp: object = None

    @classmethod
    def identity(cls):
        return cls("Whole image", CORNERS, "projective", np.eye(3))

    @classmethod
    def parse(cls, raw):
        body = object_data(raw, "Alignment")
        keys(body, "title", "corners", "model", "local", "error_rms", "matrix")
        title = label(body.get("title", "Alignment"), "Alignment name")
        entries = body.get("corners")
        if not isinstance(entries, (tuple, list)) or len(entries) != 4:
            raise Problem("Enter four corners from upper left, clockwise.")
        corners = tuple(pair(value, "Corner", -0.15, 1.15) for value in entries)
        model = body.get("model", "projective")
        if model not in ("projective", "affine", "local"):
            raise Problem("Alignment model must be projective, affine or local.")
        if model == "local":
            local = object_data(body.get("local"), "Local measured-line fit")
            keys(local, "segments", "tolerance_mm", "smoothing", "panel_mm", "note")
            if not isinstance(local.get("segments"), list) or not local["segments"]:
                raise Problem("A local fit needs at least one measured line.")
            matrix = _transformation(corners, "projective")
            return cls(title, corners, "local", matrix, local, None)
        xy = np.array(corners)
        edges = np.roll(xy, -1, axis=0) - xy
        next_edges = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
        if (cross < 0.003).any():
            raise Problem("The sheet outline must be a clockwise convex quadrilateral.")
        matrix = _transformation(corners, model)
        corners_z = np.array(CORNERS) @ matrix[2, :2] + matrix[2, 2]
        if min(corners_z) < 0.1 or np.linalg.cond(matrix) > 120:
            raise Problem("The proposed alignment is too compressed or unstable.")
        return cls(title, corners, model, matrix, None, None)

    def public(self):
        predicted = apply_matrix(self.matrix, CORNERS)
        data = dict(title=self.title, corners=[list(p) for p in self.corners], model=self.model,
                    error_rms=rounded(np.sqrt(np.mean((predicted - self.corners) ** 2))),
                    matrix=np.round(self.matrix, 8).tolist())
        if self.local is not None:
            data["local"] = self.local
        return data

    def to_camera(self, panel_positions):
        if self.warp is None:
            return apply_matrix(self.matrix, panel_positions)
        return _local_camera(self, panel_positions)

    def to_panel(self, camera_positions):
        base_panel = apply_matrix(np.linalg.inv(self.matrix), camera_positions)
        if self.warp is None or self.inverse_warp is None:
            return base_panel
        return self.inverse_warp(np.clip(base_panel, 0, 1))


def _local_camera(alignment, panel_positions):
    points = np.asarray(panel_positions, dtype=np.float64)
    corrected = alignment.warp(np.clip(points, 0, 1))
    return apply_matrix(alignment.matrix, corrected)


def sample_rgb(camera, grid, alpha_floor=0.985):
    encoded, alpha = camera.samples()
    w, h = camera.size
    u, v = np.moveaxis(grid, -1, 0)
    x, y = u*w - 0.5, v*h - 0.5
    safe = np.isfinite(x) & np.isfinite(y)
    observed = safe & (x >= -1e-5) & (x <= w-1+1e-5) & (y >= -1e-5) & (y <= h-1+1e-5)
    coords = np.stack((np.where(safe, np.clip(y, 0, h-1), 0), np.where(safe, np.clip(x, 0, w-1), 0)))
    opacity = map_coordinates(alpha, coords, order=1, mode="nearest", prefilter=False)
    premultiplied = encoded * alpha[..., None]
    values = np.stack([map_coordinates(premultiplied[..., ch], coords, order=1, mode="nearest", prefilter=False)
                       for ch in range(3)], axis=-1)
    valid = observed & (opacity >= alpha_floor)
    rgb = values / np.maximum(opacity[..., None], 0.001)
    rgb[~valid] = 0
    return rgb.astype(np.float32), valid


@dataclass(frozen=True)
class PhysicalGrid:
    width_px: int
    height_px: int
    width_mm: float
    height_mm: float

    @classmethod
    def from_panel(cls, panel, photos, limit):
        ratio = panel["width_mm"] / panel["height_mm"]
        # Budget is bounded both by the workstation request and native camera dimensions.
        longest = min(limit, max(max(photo.size) for photo in photos))
        w, h = (longest, round(longest / ratio)) if ratio >= 1 else (round(longest * ratio), longest)
        if min(w, h) < 10:
            raise Problem("The working grid needs at least ten pixels on its short edge.")
        return cls(w, h, panel["width_mm"], panel["height_mm"])

    @property
    def step_x(self):
        return self.width_mm / self.width_px

    @property
    def step_y(self):
        return self.height_mm / self.height_px

    @property
    def pixel_area(self):
        return self.step_x * self.step_y

    def grid(self):
        return normalized_grid(self.width_px, self.height_px)

    def public(self):
        return dict(raster=[self.width_px, self.height_px], physical_mm=[self.width_mm, self.height_mm],
                    step_mm=[rounded(self.step_x, 7), rounded(self.step_y, 7)])

    def point_px(self, point):
        return (min(self.width_px-1, max(0, math.floor(point[0] * self.width_px))),
                min(self.height_px-1, max(0, math.floor(point[1] * self.height_px))))

    def point_mm(self, point):
        return (point[0] * self.width_mm, point[1] * self.height_mm)
