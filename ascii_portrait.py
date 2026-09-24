#!/usr/bin/env python3
"""Clean, contrasty ASCII portrait from a photo (GitHub-README style).

Why naive ASCII fails on a real photo: the busy background competes with the
subject. The three things that actually matter:
  1) isolate the subject  (rembg person-segmentation if installed, else --crop)
  2) stretch contrast      (autocontrast + gamma on the SUBJECT only)
  3) keep the grid coarse  (~80-100 cols, aspect-corrected for tall glyphs)

Outputs <prefix>.txt (paste into README code block) and <prefix>_preview.png.

Tuning knobs (ASCII art always needs per-image tuning):
  --width N     columns (80-100 reads clean; higher = noisier)
  --gamma G     <1 lifts shadows (hair/dark detail), >1 deepens
  --crop l,t,r,b  manual crop in ORIGINAL pixels when rembg isn't available
  --ramp simple|fine
"""
import argparse, sys
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

RAMPS = {
    # index 0 == darkest (rendered as space for background); last == brightest
    "simple": " .:-=+*#%@",
    "fine":   r""" .'`^,:;Il!i~+_-?][}{1)(|tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@""",
}

def get_subject(img, crop, top):
    """Return (grayscale float array, subject-mask bool array).
    Uses rembg if available for a clean cutout; else crops + treats all as subject."""
    try:
        from rembg import remove
        cut = remove(img.convert("RGBA"))            # person segmentation
        arr = np.asarray(cut)
        alpha = arr[..., 3]
        mask = alpha > 40
        # autocrop to the subject's bounding box so it fills the frame
        ys, xs = np.where(mask)
        if len(xs):
            pad = 8
            x0, x1 = max(xs.min()-pad,0), xs.max()+pad
            y0, y1 = max(ys.min()-pad,0), ys.max()+pad
            y1 = y0 + int((y1 - y0) * top)   # keep only top fraction (head+shoulders)
            gray = np.asarray(Image.fromarray(arr).convert("L"), float)[y0:y1, x0:x1]
            mask = mask[y0:y1, x0:x1]
            return gray, mask
    except ImportError:
        pass
    # fallback: no segmentation -> manual crop, everything is "subject"
    if crop:
        img = img.crop(tuple(crop))
    gray = np.asarray(img.convert("L"), float)
    return gray, np.ones(gray.shape, bool)

def to_ascii(gray, mask, width, gamma, ramp):
    h, w = gray.shape
    new_w = width
    new_h = max(1, int(round(new_w * (h / w) * 0.5)))   # 0.5 = glyph aspect
    g = np.asarray(Image.fromarray(gray.astype("uint8")).resize((new_w, new_h)), float)
    m = np.asarray(Image.fromarray((mask*255).astype("uint8")).resize((new_w, new_h))) > 128
    # contrast-stretch the SUBJECT ONLY, so background doesn't flatten it
    sv = g[m]
    if sv.size:
        lo, hi = np.percentile(sv, 2), np.percentile(sv, 98)
        g = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
    g = g ** gamma
    n = len(ramp)
    # subject pixels map to ramp[1..n-1] (never space -> silhouette stays solid);
    # background pixels forced to space.
    idx = 1 + np.round(g * (n - 2)).astype(int)
    idx = np.clip(idx, 1, n - 1)
    idx[~m] = 0
    return "\n".join("".join(ramp[i] for i in row) for row in idx)

def render_png(text, path):
    lines = text.split("\n")
    try:
        font = ImageFont.truetype("consola.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    cw = font.getbbox("M")[2] or 8
    ch = (font.getbbox("Mg")[3] or 16) + 2
    W = max(len(l) for l in lines) * cw + 24
    H = len(lines) * ch + 24
    im = Image.new("RGB", (W, H), (13, 17, 23))       # GitHub dark
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((12, 12 + i*ch), l, font=font, fill=(201, 209, 217))
    im.save(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--width", type=int, default=90)
    p.add_argument("--gamma", type=float, default=0.85)
    p.add_argument("--top", type=float, default=1.0, help="keep top fraction of subject height (0.55=head+shoulders)")
    p.add_argument("--crop", default=None, help="l,t,r,b in original pixels")
    p.add_argument("--ramp", choices=RAMPS, default="simple")
    p.add_argument("--out", default="ascii_portrait")
    a = p.parse_args()
    crop = [int(x) for x in a.crop.split(",")] if a.crop else None
    img = Image.open(a.input)
    img = ImageOps.exif_transpose(img)                # respect phone rotation
    gray, mask = get_subject(img, crop, a.top)
    art = to_ascii(gray, mask, a.width, a.gamma, RAMPS[a.ramp])
    open(a.out + ".txt", "w").write(art)
    render_png(art, a.out + "_preview.png")
    print(f"wrote {a.out}.txt ({art.count(chr(10))+1} rows) and {a.out}_preview.png")

if __name__ == "__main__":
    main()
