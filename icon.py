"""Generate assets/icon.icns from scratch using Pillow."""
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SIZES = [16, 32, 64, 128, 256, 512, 1024]


def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = size / 1024  # scale factor

    # Background circle
    pad = int(32 * p)
    d.ellipse([pad, pad, size - pad, size - pad], fill=(30, 30, 35, 255))

    # Mic body (rounded rect)
    mw = int(200 * p)
    mh = int(300 * p)
    mx = (size - mw) // 2
    my = int(180 * p)
    radius = int(100 * p)
    d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=radius, fill=(255, 255, 255, 255))

    # Mic stand arc (bottom half of ellipse)
    sw = int(360 * p)
    sh = int(280 * p)
    sx = (size - sw) // 2
    sy = int(430 * p)
    d.arc([sx, sy, sx + sw, sy + sh], start=0, end=180, fill=(255, 255, 255, 255), width=int(36 * p))

    # Mic stand pole
    pole_x = size // 2
    pole_top = int(sy + sh // 2)
    pole_bot = int(pole_top + 100 * p)
    lw = int(36 * p)
    d.rectangle([pole_x - lw // 2, pole_top, pole_x + lw // 2, pole_bot], fill=(255, 255, 255, 255))

    # Mic stand base
    bw = int(220 * p)
    bh = int(36 * p)
    bx = (size - bw) // 2
    by = pole_bot
    d.rectangle([bx, by, bx + bw, by + bh], fill=(255, 255, 255, 255))

    # Slight glow
    glow = img.filter(ImageFilter.GaussianBlur(radius=int(6 * p)))
    out = Image.alpha_composite(glow, img)
    return out


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
            img2 = draw_icon(size * 2)
            img2.save(iconset / f"icon_{size}x{size}@2x.png")

    icns = assets / "icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    shutil.rmtree(iconset)
    print(f"Icon written to {icns}")


def draw_menubar_icon(size: int = 44) -> Image.Image:
    """Black mic on transparent background — used as a menu bar template image."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    p = size / 44

    # Mic body
    mw = int(12 * p); mh = int(18 * p)
    mx = (size - mw) // 2; my = int(4 * p)
    d.rounded_rectangle([mx, my, mx + mw, my + mh], radius=int(6 * p), fill=(0, 0, 0, 255))

    # Stand arc
    sw = int(22 * p); sh = int(14 * p)
    sx = (size - sw) // 2; sy = int(17 * p)
    d.arc([sx, sy, sx + sw, sy + sh], start=0, end=180, fill=(0, 0, 0, 255), width=max(2, int(2.5 * p)))

    # Pole
    cx = size // 2; pole_top = int(sy + sh // 2); pole_bot = int(pole_top + 6 * p)
    lw = max(2, int(2.5 * p))
    d.rectangle([cx - lw // 2, pole_top, cx + lw // 2, pole_bot], fill=(0, 0, 0, 255))

    # Base
    bw = int(14 * p); bh = max(2, int(2 * p))
    d.rectangle([(size - bw) // 2, pole_bot, (size + bw) // 2, pole_bot + bh], fill=(0, 0, 0, 255))

    return img


def draw_menubar_recording(size: int = 44) -> Image.Image:
    """Mic with a filled dot — recording state."""
    img = draw_menubar_icon(size)
    d = ImageDraw.Draw(img)
    p = size / 44
    r = int(5 * p)
    cx = size - r - int(2 * p)
    cy = r + int(2 * p)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, 255))
    return img


def draw_menubar_processing(size: int = 44) -> Image.Image:
    """Mic with ellipsis dots — transcribing state."""
    img = draw_menubar_icon(size)
    d = ImageDraw.Draw(img)
    p = size / 44
    r = int(2 * p)
    y = int(38 * p)
    for i, x in enumerate([int(15 * p), int(22 * p), int(29 * p)]):
        d.ellipse([x - r, y - r, x + r, y + r], fill=(0, 0, 0, 255))
    return img


def build_menubar():
    assets = Path("assets")
    assets.mkdir(exist_ok=True)
    for name, fn in [
        ("menubar", draw_menubar_icon),
        ("menubar_rec", draw_menubar_recording),
        ("menubar_proc", draw_menubar_processing),
    ]:
        fn(22).save(assets / f"{name}.png")
        fn(44).save(assets / f"{name}@2x.png")
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
            img2 = draw_icon(size * 2)
            img2.save(iconset / f"icon_{size}x{size}@2x.png")

    icns = assets / "icon.icns"
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    shutil.rmtree(iconset)
    print(f"Icon written to {icns}")
    build_menubar()


if __name__ == "__main__":
    build()
