"""Generate the white-lamp and warm-lamp layout photographs for testing.

Five pieces are laid out in a row: original fragment, three replacement
samples, one clear blank. The two images share piece positions, only the lamp
spectrum and exposure differ. The sample whose hue is closest to the original
is deliberately different under the two lamps so that the ranking swaps.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


W, H = 880, 660
# Default normalized boxes used by the replacement-match tool.
BOXES = {
    "original": (0.06, 0.32, 0.20, 0.68),
    "sample_1": (0.24, 0.32, 0.38, 0.68),
    "sample_2": (0.42, 0.32, 0.56, 0.68),
    "sample_3": (0.60, 0.32, 0.74, 0.68),
    "blank": (0.78, 0.32, 0.92, 0.68),
}
# Linear-sRGB camera response of each glass under each lamp. The blank is the
# neutral reference; glasses keep distinct per-lamp responses (metamerism), so
# rankings are allowed to change between the two lighting conditions.
GLASS = {
    "white": {
        # Original: cyan-leaning (green/blue dominate, red weakest).
        "original": (0.30, 0.52, 0.58),
        # Sample 1: hue almost identical to the original, modestly lighter.
        "sample_1": (0.34, 0.57, 0.62),
        "sample_2": (0.40, 0.55, 0.57),
        # Sample 3: distinctly darker, hue rotated toward green.
        "sample_3": (0.12, 0.34, 0.30),
        "blank": (0.90, 0.90, 0.90),
    },
    "warm": {
        "original": (0.31, 0.40, 0.27),
        "sample_1": (0.36, 0.43, 0.26),
        "sample_2": (0.35, 0.42, 0.29),
        "sample_3": (0.12, 0.26, 0.14),
        "blank": (0.90, 0.66, 0.42),
    },
}


def srgb_encode(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055)


def make(lamp_name, seed):
    rng = np.random.default_rng(seed)
    canvas = np.full((H, W, 3), 0.07, dtype=float)
    table = GLASS[lamp_name]
    for name, (l, t, r, b) in BOXES.items():
        x0, x1 = round(l * W), round(r * W)
        y0, y1 = round(t * H), round(b * H)
        signal = np.asarray(table[name], dtype=float)
        block = np.tile(signal, (y1 - y0, x1 - x0, 1))
        block *= rng.normal(1.0, 0.012, block.shape)
        canvas[y0:y1, x0:x1] = np.clip(block, 0, 1)
    encoded = np.rint(255 * srgb_encode(canvas)).astype(np.uint8)
    return Image.fromarray(encoded, "RGB")


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, seed in (("white", 11), ("warm", 23)):
        make(name, seed).save(out / f"{name}-lamp.png", format="PNG")
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main(sys.argv[1])
