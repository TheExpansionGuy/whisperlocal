"""Generate WhisperLocal icons — soundprint style."""
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZES = [16, 32, 64, 128, 256, 512, 1024]


def _rounded_square_gradient(size, p):
    """A macOS-style rounded square with a vertical indigo→blue gradient."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grad)
    top = (88, 86, 240)     # indigo
    bot = (56, 150, 255)    # blue
    for y in range(size):
        t = y / size
        r = int(top[0] + (bot[0] - top[0]) * t)
        g = int(top[1] + (bot[1] - top[1]) * t)
        b = int(top[2] + (bot[2] - top[2]) * t)
        gd.line([(0, y), (size, y)], fill=(r, g, b, 255))
    # Rounded-square mask
    mask = Image.new("L", (size, size), 0)
    md = ImageDraw.Draw(mask)
    margin = int(40 * p)
    radius = int(230 * p)
    md.rounded_rectangle([margin, margin, size - margin, size - margin],
                         radius=radius, fill=255)
    img.paste(grad, (0, 0), mask)
    return img


def draw_icon(size: int) -> Image.Image:
    p = size / 1024
    cx = cy = size / 2
    img = _rounded_square_gradient(size, p)
    d = ImageDraw.Draw(img)

    white = (255, 255, 255, 255)

    # --- Clean centered microphone -------------------------------------
    # Capsule body
    mc_w = int(150 * p)
    mc_h = int(300 * p)
    mc_x = cx - mc_w / 2
    mc_y = int(235 * p)
    d.rounded_rectangle([mc_x, mc_y, mc_x + mc_w, mc_y + mc_h],
                        radius=int(75 * p), fill=white)

    # Cradle arc (U-shape around the lower mic)
    cw = int(250 * p)
    ch = int(250 * p)
    cax = cx - cw / 2
    cay = int(360 * p)
    d.arc([cax, cay, cax + cw, cay + ch], start=20, end=160,
          fill=white, width=int(34 * p))

    # Stem + base
    stem_top = int(cay + ch / 2 + 6 * p)
    stem_bot = int(stem_top + 70 * p)
    sw = int(34 * p)
    d.rectangle([cx - sw / 2, stem_top, cx + sw / 2, stem_bot], fill=white)
    bw = int(170 * p)
    bh = int(34 * p)
    d.rounded_rectangle([cx - bw / 2, stem_bot, cx + bw / 2, stem_bot + bh],
                        radius=int(16 * p), fill=int(0))  # placeholder, replaced below
    d.rounded_rectangle([cx - bw / 2, stem_bot, cx + bw / 2, stem_bot + bh],
                        radius=int(16 * p), fill=white)

    # --- Sound waves on either side ------------------------------------
    wave_col = (255, 255, 255, 210)
    for i, r in enumerate([int(120 * p), int(180 * p)]):
        lw = int((20 - i * 4) * p)
        # left
        d.arc([cx - mc_w/2 - r - int(40*p), cy - r,
               cx - mc_w/2 - int(40*p) + r, cy + r],
              start=110, end=250, fill=wave_col, width=lw)
        # right
        d.arc([cx + mc_w/2 + int(40*p) - r, cy - r,
               cx + mc_w/2 + int(40*p) + r, cy + r],
              start=-70, end=70, fill=wave_col, width=lw)

    return img


def draw_menubar_icon(size: int = 44, state: str = "idle") -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = size / 44
    cx = cy = size / 2

    color = (0, 0, 0, 255)  # template image — black on transparent

    # Mic body
    mw = int(12 * p); mh = int(18 * p)
    mx = (size - mw) // 2; my = int(4 * p)
    d.rounded_rectangle([mx, my, mx + mw, my + mh],
                        radius=int(6 * p), fill=color)

    # Stand arc
    sw = int(22 * p); sh = int(14 * p)
    sx = (size - sw) // 2; sy = int(17 * p)
    d.arc([sx, sy, sx + sw, sy + sh], start=0, end=180,
          fill=color, width=max(2, int(2.5 * p)))

    # Pole + base
    lw = max(2, int(2.5 * p))
    pole_top = int(sy + sh // 2); pole_bot = int(pole_top + 6 * p)
    d.rectangle([cx - lw//2, pole_top, cx + lw//2, pole_bot], fill=color)
    bw = int(14 * p)
    d.rectangle([(size - bw)//2, pole_bot, (size + bw)//2,
                 pole_bot + max(2, int(2*p))], fill=color)

    if state == "recording":
        # Small filled circle top-right
        dr = int(4 * p)
        d.ellipse([size - dr*2 - 1, 1, size - 1, dr*2 + 1], fill=color)
    elif state == "processing":
        # Three small dots at bottom
        dr = int(2 * p)
        y = size - dr * 2 - 1
        for i, x in enumerate([int(size*0.3), int(size*0.5), int(size*0.7)]):
            d.ellipse([x-dr, y-dr, x+dr, y+dr], fill=color)

    return img


def build_menubar():
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    for name, state in [("menubar", "idle"), ("menubar_rec", "recording"),
                        ("menubar_proc", "processing")]:
        draw_menubar_icon(22, state).save(assets / f"{name}.png")
        draw_menubar_icon(44, state).save(assets / f"{name}@2x.png")
    print("Menu bar icons written to assets/")


def build():
    assets = Path("assets")
    assets.mkdir(exist_ok=True)

    iconset = assets / "icon.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()

    for size in SIZES:
        img = draw_icon(size)
        img.save(iconset / f"icon_{size}x{size}.png")
        if size <= 512:
            draw_icon(size * 2).save(iconset / f"icon_{size}x{size}@2x.png")

    icns = assets / "icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    shutil.rmtree(iconset)
    print(f"Icon written to {icns}")
    build_menubar()


if __name__ == "__main__":
    build()
