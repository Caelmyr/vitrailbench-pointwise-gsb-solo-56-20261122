from io import BytesIO, StringIO
import csv

import numpy as np
from PIL import Image, ImageDraw

from .bench import Asset
from .camera import linear_to_srgb


MISSING = (42, 46, 56)
BRIGHT = (249, 238, 211)
PALETTES = {
    "thermal": ((20, 23, 44), (77, 51, 137), (200, 66, 105), (248, 143, 65), (252, 244, 165)),
    "field": ((29, 45, 89), (45, 108, 150), (100, 178, 149), (244, 224, 135), (249, 249, 230)),
    "diverging": ((50, 91, 163), (123, 168, 230), (237, 233, 220), (223, 146, 119), (161, 48, 42)),
    "ink": ((252, 230, 151), (206, 141, 91), (136, 71, 92), (63, 49, 95), (22, 25, 53)),
}


def colorize(values, lower, upper, palette, valid):
    colors = PALETTES[palette]
    norm = np.clip((values - lower) / max(upper-lower, 1e-7), 0, 1)
    coordinates = np.linspace(0, 1, len(colors))
    rgb = np.stack([np.interp(norm, coordinates, np.asarray(colors)[:, band])
                    for band in range(3)], axis=-1).astype(np.uint8)
    rgb[~valid] = MISSING
    return rgb


def coded(scene, valid=None):
    layer = linear_to_srgb(scene).copy()
    if valid is not None:
        layer[~valid] = MISSING
    return layer


def png(key, image, context=None, *, crop=True):
    array = np.asarray(image)
    region = None
    if context is not None:
        region = tuple(context.order.region.public()) if crop else (0, 0, 1, 1)
        if crop:
            array = context.order.region.slice(array)
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG", compress_level=5)
    return Asset(key, f"{key}.png", "image/png", output.getvalue(), region)


def csv_table(key, labels, rows):
    fp = StringIO(newline="")
    writer = csv.writer(fp)
    writer.writerow(labels)
    writer.writerows(rows)
    return Asset(key, f"{key}.csv", "text/csv", fp.getvalue().encode("utf-8"))


def mask_overlay(scene, layers, support=None):
    rgb = coded(scene, support).astype(np.float32)
    for mask, color, blend in layers:
        rgb[mask] = rgb[mask] * (1-blend) + np.asarray(color) * blend
    return np.rint(rgb).astype(np.uint8)


def map_asset(key, values, allowed, context, *, low, high, palette="field"):
    return png(key, colorize(values, low, high, palette, allowed), context)


def chart(key, positions, series, title, ylabel, x_caption="Distance (mm)", *, width=790, height=365):
    picture = Image.new("RGB", (width, height), (250, 248, 241))
    draw = ImageDraw.Draw(picture)
    left, top, right, bottom = 67, 42, width-28, height-47
    draw.text((left, 14), title, fill=(36, 45, 56))
    draw.text((16, 26), ylabel, fill=(36, 45, 56))
    data = [value for _, _, values in series for value in values if value is not None and np.isfinite(value)]
    lo, hi = (min(data), max(data)) if data else (0, 1)
    if hi - lo < 0.00001:
        lo, hi = lo - 0.05, hi + 0.05
    x_min = float(min(positions))
    x_max = float(max(positions))
    for offset in range(5):
        y = top + (bottom-top)*offset/4
        draw.line((left, y, right, y), fill=(216, 212, 202))
        draw.text((14, y-4), f"{hi-(hi-lo)*offset/4:.3g}", fill=(66, 63, 57))
    draw.line(((left, top), (left, bottom), (right, bottom)), fill=(45, 50, 53), width=2)
    draw.text(((left+right)/2-36, height-19), x_caption, fill=(54, 58, 62))
    for offset in range(5):
        x = left + (right-left)*offset/4
        draw.text((x-16, bottom+8), f"{x_min+(x_max-x_min)*offset/4:.4g}", fill=(66, 63, 57))
    for tag_i, (name, color, values) in enumerate(series):
        previous = None
        for position, value in zip(positions, values, strict=True):
            if value is None or not np.isfinite(value):
                previous = None
                continue
            x = left + (float(position)-x_min) / max(x_max-x_min, 0.001) * (right-left)
            y = bottom - (float(value)-lo) / (hi-lo) * (bottom-top)
            if previous:
                draw.line((previous, (x, y)), fill=color, width=2)
            else:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            previous = (x, y)
        draw.text((right-122, 14+tag_i*13), name, fill=color)
    return png(key, picture)


def mark_image(image, shapes):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    h, w = image.shape[:2]
    scale = max(1, int(round(max(w, h)/320)))
    for points, color, text in shapes:
        pixels = [(round(x*(w-1)), round(y*(h-1))) for x, y in points]
        if len(pixels) == 1:
            x, y = pixels[0]
            draw.line([(x-7, y), (x+7, y)], fill=color, width=scale)
            draw.line([(x, y-7), (x, y+7)], fill=color, width=scale)
        else:
            draw.line(pixels, fill=color, width=scale, joint="curve")
            for x, y in pixels:
                draw.ellipse((x-2*scale, y-2*scale, x+2*scale, y+2*scale), outline=color, width=scale)
        if text and pixels:
            x, y = pixels[0]
            draw.text((min(x+7, w-70), min(y+6, h-13)), text[:27], fill=color, stroke_width=2,
                      stroke_fill=(29, 31, 40))
    return np.asarray(canvas)
