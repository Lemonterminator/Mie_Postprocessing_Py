"""Regenerate cover_en.png (1200x627 og:image) from live demo screenshots.

Pipeline: headless Edge renders metrics_derivations_en.html with a #demoN hash
(which auto-expands the containing section) into a very tall viewport; the demo
boxes are then located by scanning for their background colour (#f6f9fc) and
composed into the cover. Run with the repo .venv python (needs Pillow, numpy).
"""
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(HERE, "metrics_derivations_en.html")
OUT = os.path.join(HERE, "cover_en.png")
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
DEMO_BG = np.array([246, 249, 252])  # .demo background #f6f9fc

# which demos appear on the cover, left to right
COVER_DEMOS = ["demo3", "demo4"]

W, H = 1200, 627
BG = (255, 253, 249)      # page bg #fffdf9
ACCENT = (20, 80, 122)    # #14507a
BORDER = (157, 184, 204)  # #9db8cc


def screenshot(demo, out_png, tmp):
    url = "file:///" + HTML.replace("\\", "/") + "#" + demo
    subprocess.run([
        EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--window-size=1200,16000", "--virtual-time-budget=30000",
        "--run-all-compositor-stages-before-draw",
        "--user-data-dir=" + os.path.join(tmp, "edgeprofile"),
        "--screenshot=" + out_png, url,
    ], check=True, capture_output=True)


def demo_boxes(png):
    """Bounding boxes of .demo panels, found by their background colour.
    A demo's white canvas splits its colour band in two, so nearby bands are merged."""
    img = Image.open(png).convert("RGB")
    a = np.asarray(img)
    mask = np.abs(a.astype(int) - DEMO_BG).max(axis=2) <= 2
    rows = mask.sum(axis=1) > 80
    bands, start = [], None
    for y, v in enumerate(rows):
        if v and start is None:
            start = y
        elif not v and start is not None:
            bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(rows)))
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] < 400:
            merged[-1] = (merged[-1][0], b[1])
        else:
            merged.append(b)
    out = []
    for y0, y1 in (b for b in merged if b[1] - b[0] > 300):
        xs = np.where(mask[y0:y1].any(axis=0))[0]
        out.append(img.crop((int(xs.min()) - 4, y0 - 4, int(xs.max()) + 4, y1 + 4)))
    return out


def main():
    cover = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cover)

    f_title = ImageFont.truetype(r"C:\Windows\Fonts\georgiab.ttf", 44)
    f_arrow = ImageFont.truetype(r"C:\Windows\Fonts\seguisym.ttf", 50)  # georgia lacks U+2192
    f_sub = ImageFont.truetype(r"C:\Windows\Fonts\georgia.ttf", 21)
    f_auth = ImageFont.truetype(r"C:\Windows\Fonts\georgia.ttf", 18)

    x = 36
    for part, fnt, dy in [("Calibration", f_title, 0), (" → ", f_arrow, -4),
                          ("Uncertainty", f_title, 0), (" → ", f_arrow, -4),
                          ("Risk", f_title, 0)]:
        d.text((x, 16 + dy), part, font=fnt, fill=ACCENT)
        x += d.textlength(part, font=fnt)
    d.text((36, 74), "Every metric derived · 5 interactive experiments · RMSE · Coverage · CRPS · PIT · NLL",
           font=f_sub, fill=(70, 70, 70))
    auth = "Linan Jiang"
    d.text((W - 36 - d.textlength(auth, font=f_auth), 30), auth, font=f_auth, fill=(120, 120, 120))
    d.line([(36, 108), (W - 36, 108)], fill=ACCENT, width=2)

    pan_y, pan_h = 122, H - 122 - 14
    pan_w = (W - 3 * 24) // 2
    with tempfile.TemporaryDirectory() as tmp:
        for i, demo in enumerate(COVER_DEMOS):
            shot = os.path.join(tmp, demo + ".png")
            screenshot(demo, shot, tmp)
            boxes = demo_boxes(shot)
            # the hash target is scrolled toward, but all demos of the expanded
            # section are in the shot; pick the box whose id matches by order:
            # sections here contain a single cover demo, so take the last box
            # (demo1's section holds demo2 first, demo1 last — irrelevant for 3/4).
            box = boxes[-1]
            scale = pan_w / box.width
            box = box.resize((pan_w, int(box.height * scale)), Image.LANCZOS)
            if box.height > pan_h:
                box = box.crop((0, 0, pan_w, pan_h))
            px = 24 + i * (pan_w + 24)
            cover.paste(box, (px, pan_y))
            d.rectangle([px - 1, pan_y - 1, px + pan_w, pan_y + min(box.height, pan_h)],
                        outline=BORDER, width=1)
    cover.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
