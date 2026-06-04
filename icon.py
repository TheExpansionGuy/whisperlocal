"""Generate WhisperLocal icons — soundprint style."""
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZES = [16, 32, 64, 128, 256, 512, 1024]


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = size / 1024
    cx = cy = size / 2

    # Background circle with deep gradient (draw as layered circles)
    bg_r = int(480 * p)
    for i in range(bg_r, 0, -1):
        t = i / bg_r
        r = int(20 + (45 - 20) * t)
        g = int(10 + (15 - 10) * t)
        b = int(50 + (80 - 50) * t)
        d.ellipse([cx - i, cy - i, cx + i, cy + i], fill=(r, g, b, 255))

    # Soundprint — concentric arcs that look like a fingerprint
    # Each ring is a slightly irregular ellipse arc
    num_rings = 14
    for ring in range(num_rings):
        t = ring / (num_rings - 1)
        base_r = int((60 + 340 * t) * p)

        # Colour: inner purple → outer blue
        cr = int(160 - 80 * t)
        cg = int(100 + 100 * t)
        cb = int(255)
        alpha = int(220 - 60 * t)
        lw = max(1, int((3.5 - 1.5 * t) * p))

        # Draw as segmented arcs with slight irregularities (fingerprint feel)
        # Break each ring into segments and offset them slightly
        segments = 3 + ring % 3
        for seg in range(segments):
            seg_start = (seg / segments) * 360
            seg_end   = ((seg + 0.85) / segments) * 360

            # Slight radial wobble per segment
            wobble = int(8 * p * math.sin(ring * 1.3 + seg * 2.1))
            rx = base_r + wobble
            ry = int(base_r * (0.72 + 0.06 * math.sin(ring * 0.8)))

            box = [cx - rx, cy - ry, cx + rx, cy + ry]
            d.arc(box, start=seg_start + 5, end=seg_end - 5,
                  fill=(cr, cg, cb, alpha), width=lw)

    # Central mic mark — minimal, clean
    mc_w = int(52 * p)
    mc_h = int(72 * p)
    mc_x = cx - mc_w / 2
    mc_y = cy - mc_h / 2 - int(10 * p)
    radius = int(26 * p)
    d.rounded_rectangle([mc_x, mc_y, mc_x + mc_w, mc_y + mc_h],
                        radius=radius, fill=(255, 255, 255, 220))

    # Stand arc
    sw = int(90 * p); sh = int(60 * p)
    sx = cx - sw / 2; sy = cy + int(16 * p)
    d.arc([sx, sy, sx + sw, sy + sh], start=0, end=180,
          fill=(255, 255, 255, 200), width=max(2, int(7 * p)))

    # Pole + base
    pole_x = int(cx); pole_top = int(sy + sh / 2); pole_bot = int(pole_top + 22 * p)
    lw2 = max(2, int(7 * p))
    d.rectangle([pole_x - lw2//2, pole_top, pole_x + lw2//2, pole_bot],
                fill=(255, 255, 255, 200))
    bw = int(54 * p)
    d.rectangle([cx - bw//2, pole_bot, cx + bw//2, pole_bot + max(2, int(7*p))],
                fill=(255, 255, 255, 200))

    # Soft glow
    glow = img.filter(ImageFilter.GaussianBlur(radius=max(1, int(4 * p))))
    return Image.alpha_composite(glow, img)


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
