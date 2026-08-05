"""Build a synthetic YOLO dataset for the field localizer by compositing
programmatically-drawn header/table/signature/logo elements onto real
document images (reused from the classifier's data/raw/).

Why synthetic instead of hand-labeled: avoids manual bounding-box
annotation entirely -- since we place each element ourselves, the box
coordinates are exact by construction, no drawing/correcting needed.

Known limitation (disclose in README): elements are drawn shapes/text,
not real headers/tables/signatures/logos, and are placed in fixed zones
(top strip = header/logo, middle band = table, bottom-right = signature)
rather than learned from real layout variety. The backgrounds are real
scans, but the foreground elements are not -- a real-world model trained
here would need fine-tuning on genuine labeled examples before production
use. Good enough for a from-scratch v1 pipeline demo.

Run: python src/localizer/build_synthetic_dataset.py
"""

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "localizer"
N_TRAIN = 400
N_VAL = 80
SEED = 42

CLASSES = ["header", "table", "signature", "logo"]
FONT_PATH = "C:/Windows/Fonts/arial.ttf"

HEADER_TEXTS = [
    "INVOICE #{}", "MEMORANDUM", "Dear Sir or Madam,", "ACME CORPORATION",
    "MONTHLY REPORT", "APPLICATION FORM", "RE: Account Statement",
    "PURCHASE ORDER #{}", "CURRICULUM VITAE", "STATEMENT OF WORK",
]


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def draw_header(draw, w, h):
    x0, y0 = random.randint(int(0.06 * w), int(0.12 * w)), random.randint(int(0.02 * h), int(0.05 * h))
    text = random.choice(HEADER_TEXTS).format(random.randint(1000, 9999))
    f = font(random.randint(20, 30))
    l, t, r, b = draw.textbbox((x0, y0), text, font=f)
    draw.text((x0, y0), text, fill="black", font=f)
    pad = 4
    return "header", (l - pad, t - pad, r + pad, b + pad)


def draw_logo(draw, w, h):
    size = random.randint(50, 90)
    corner = random.choice(["tl", "tr"])
    margin = int(0.05 * w)
    x0 = margin if corner == "tl" else w - margin - size
    y0 = random.randint(int(0.02 * h), int(0.06 * h))
    x1, y1 = x0 + size, y0 + size
    color = random.choice(["#1a3c6e", "#2e2e2e", "#6e1a1a"])
    if random.random() < 0.5:
        draw.ellipse([x0, y0, x1, y1], fill=color)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=color)
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.choice([1, 2])))
    f = font(int(size * 0.4))
    tl, tt, tr, tb = draw.textbbox((0, 0), letters, font=f)
    tw, th = tr - tl, tb - tt
    draw.text((x0 + (size - tw) / 2, y0 + (size - th) / 2 - tt), letters, fill="white", font=f)
    return "logo", (x0, y0, x1, y1)


def draw_table(draw, w, h):
    rows, cols = random.randint(3, 6), random.randint(2, 4)
    x0 = random.randint(int(0.08 * w), int(0.15 * w))
    x1 = random.randint(int(0.75 * w), int(0.92 * w))
    y0 = random.randint(int(0.38 * h), int(0.48 * h))
    row_h = random.randint(int(0.03 * h), int(0.05 * h))
    y1 = y0 + rows * row_h
    draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
    for i in range(1, rows):
        y = y0 + i * row_h
        draw.line([(x0, y), (x1, y)], fill="black", width=1)
    for j in range(1, cols):
        x = x0 + j * (x1 - x0) // cols
        draw.line([(x, y0), (x, y1)], fill="black", width=1)
    return "table", (x0, y0, x1, y1)


def draw_signature(draw, w, h):
    x0 = random.randint(int(0.55 * w), int(0.7 * w))
    y0 = random.randint(int(0.82 * h), int(0.88 * h))
    points = [(x0, y0)]
    for _ in range(random.randint(5, 9)):
        px, py = points[-1]
        points.append((px + random.randint(10, 25), py + random.randint(-15, 15)))
    draw.line(points, fill="black", width=2, joint="curve")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pad = 6
    return "signature", (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


DRAWERS = {"header": draw_header, "table": draw_table, "signature": draw_signature, "logo": draw_logo}


def make_sample(bg_path, out_img_path, out_label_path):
    img = Image.open(bg_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    chosen = random.sample(CLASSES, k=random.randint(1, 4))
    lines = []
    for cls in chosen:
        _, (l, t, r, b) = DRAWERS[cls](draw, w, h)
        l, r = max(0, l), min(w, r)
        t, b = max(0, t), min(h, b)
        xc, yc = (l + r) / 2 / w, (t + b) / 2 / h
        bw, bh = (r - l) / w, (b - t) / h
        lines.append(f"{CLASSES.index(cls)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    img.save(out_img_path, quality=90)
    out_label_path.write_text("\n".join(lines))


def build_split(split, n, raw_split_dir):
    img_dir = OUT_DIR / "images" / split
    label_dir = OUT_DIR / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = list(raw_split_dir.rglob("*.jpg"))
    picks = random.sample(backgrounds, n)
    for i, bg in enumerate(picks):
        make_sample(bg, img_dir / f"{split}_{i:04d}.jpg", label_dir / f"{split}_{i:04d}.txt")


def write_data_yaml():
    yaml_text = (
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_text)


def main():
    random.seed(SEED)
    build_split("train", N_TRAIN, RAW_DIR / "train")
    build_split("val", N_VAL, RAW_DIR / "val")
    write_data_yaml()
    print(f"wrote {N_TRAIN} train / {N_VAL} val synthetic samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
