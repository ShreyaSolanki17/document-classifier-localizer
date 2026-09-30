"""Build a synthetic YOLO dataset for the field localizer by compositing
programmatically-drawn header/table/signature/logo elements onto real
document images (reused from the classifier's data/raw/).

Why synthetic instead of hand-labeled: avoids manual bounding-box
annotation entirely -- since we place each element ourselves, the box
coordinates are exact by construction, no drawing/correcting needed.

Spatial placement (v2): each element is placed by picking a random
horizontal/vertical zone from PLACEMENT_ZONES and a random point inside it,
instead of the old fixed top/middle/bottom-right bands. This removes the
positional shortcut a detector could otherwise learn (e.g. "table = middle
of page") while still keeping placements bounded and non-overlapping via
rejection sampling (see _overlaps). Elements are still drawn shapes/text,
not real headers/tables/signatures/logos -- a real-world model trained here
would need fine-tuning on genuine labeled examples before production use.

Table content (v3): cells are filled with synthetic invoice-like content
(item names, quantities, prices, dates, percentages, reference IDs) instead
of an empty grid, so the table's appearance is closer to a real document
table. Only the TABLE class changed -- header/logo/signature generation is
untouched, and the table's outer YOLO bounding box still represents the
whole table (content is drawn strictly inside it, no extra boxes).

Writes to data/localizer_v3/ so the Phase 2B dataset (data/localizer_v2/)
and the original fixed-band baseline (data/localizer/) are left untouched.

Run: python src/localizer/build_synthetic_dataset.py
"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "localizer_v3"
N_TRAIN = 400
N_VAL = 80
SEED = 42

CLASSES = ["header", "table", "signature", "logo"]
FONT_PATH = "C:/Windows/Fonts/arial.ttf"

DATASET_VERSION = "v3-table-content"
SPATIAL_RANDOMIZATION_VERSION = "v1"
TABLE_CONTENT_VERSION = "v1"
MAX_PLACEMENT_TRIES = 15
OVERLAP_THRESHOLD = 0.05  # max fraction of a new box's area allowed to overlap already-placed boxes

# Each field picks a random (lo, hi) zone -- as a fraction of image w/h --
# then a random point inside it. Multiple zones per axis = varied but
# bounded placement, instead of one fixed narrow band or fully free coords.
PLACEMENT_ZONES = {
    "header": {
        "x": [(0.04, 0.30), (0.28, 0.55), (0.50, 0.80)],
        "y": [(0.02, 0.15), (0.18, 0.35), (0.38, 0.55), (0.58, 0.75)],
    },
    "logo": {
        "x": [(0.04, 0.20), (0.35, 0.55), (0.70, 0.88)],
        "y": [(0.02, 0.15), (0.20, 0.40), (0.45, 0.65), (0.70, 0.85)],
    },
    "table": {
        "x_left": [(0.05, 0.12), (0.20, 0.30)],
        "x_right": [(0.70, 0.80), (0.85, 0.94)],
        "y": [(0.08, 0.18), (0.25, 0.35), (0.42, 0.52), (0.55, 0.62)],
    },
    "signature": {
        "x": [(0.05, 0.25), (0.35, 0.55), (0.60, 0.80)],
        "y": [(0.35, 0.50), (0.55, 0.70), (0.75, 0.88)],
    },
}

HEADER_TEXTS = [
    "INVOICE #{}", "MEMORANDUM", "Dear Sir or Madam,", "ACME CORPORATION",
    "MONTHLY REPORT", "APPLICATION FORM", "RE: Account Statement",
    "PURCHASE ORDER #{}", "CURRICULUM VITAE", "STATEMENT OF WORK",
]

# Each table column is assigned one content type (so a column reads as
# consistently numeric or textual, like a real table) instead of every cell
# being independently random.
TABLE_ITEM_NAMES = [
    "Widget", "Gadget", "Consulting Fee", "License", "Shipping",
    "Materials", "Labor", "Component", "Assembly Kit", "Service Charge",
    "Maintenance", "Subscription", "Hardware Unit", "Software Seat",
]
TABLE_COLUMN_TYPES = ["item", "qty", "price", "date", "percent", "id"]
TABLE_HEADER_LABELS = {
    "item": "Item", "qty": "Qty", "price": "Price",
    "date": "Date", "percent": "Rate", "id": "Ref #",
}
TABLE_NUMERIC_COLUMN_TYPES = {"qty", "price", "percent", "id"}


def _table_cell_text(col_type):
    if col_type == "item":
        return random.choice(TABLE_ITEM_NAMES)
    if col_type == "qty":
        return str(random.randint(1, 500))
    if col_type == "price":
        return f"${random.randint(1, 9999)}.{random.randint(0, 99):02d}"
    if col_type == "date":
        return f"{random.randint(1, 12):02d}/{random.randint(1, 28):02d}/{random.randint(20, 26)}"
    if col_type == "percent":
        return f"{random.randint(0, 100)}%"
    return "#" + "".join(random.choices("0123456789", k=random.randint(4, 6)))  # id


def _fit_font(draw, text, max_width, start_size, min_size=8):
    """Shrink font size until `text` fits max_width, so cell content never
    overflows its column (and therefore never overflows the table box)."""
    size = start_size
    f = font(size)
    while size > min_size and draw.textlength(text, font=f) > max_width:
        size -= 1
        f = font(size)
    return f


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _zone(ranges, dim):
    lo, hi = random.choice(ranges)
    return random.uniform(lo, hi) * dim


def _overlaps(box, placed, threshold=OVERLAP_THRESHOLD):
    l, t, r, b = box
    area = max(0, r - l) * max(0, b - t)
    if area <= 0:
        return True
    for pl, pt, pr, pb in placed:
        il, it = max(l, pl), max(t, pt)
        ir, ib = min(r, pr), min(b, pb)
        inter = max(0, ir - il) * max(0, ib - it)
        if inter / area > threshold:
            return True
    return False


def draw_header(draw, w, h, placed):
    text = random.choice(HEADER_TEXTS).format(random.randint(1000, 9999))
    f = font(random.randint(20, 30))
    pad = 4
    x0 = y0 = 0
    box = (0, 0, 0, 0)
    for _try in range(MAX_PLACEMENT_TRIES):
        x0 = int(_zone(PLACEMENT_ZONES["header"]["x"], w))
        y0 = int(_zone(PLACEMENT_ZONES["header"]["y"], h))
        l, t, r, b = draw.textbbox((x0, y0), text, font=f)
        box = (l - pad, t - pad, r + pad, b + pad)
        if box[2] <= w and box[3] <= h and not _overlaps(box, placed):
            break
    draw.text((x0, y0), text, fill="black", font=f)
    return "header", box


def draw_logo(draw, w, h, placed):
    size = random.randint(50, 90)
    color = random.choice(["#1a3c6e", "#2e2e2e", "#6e1a1a"])
    ellipse = random.random() < 0.5
    box = (0, 0, size, size)
    for _try in range(MAX_PLACEMENT_TRIES):
        x0 = int(_zone(PLACEMENT_ZONES["logo"]["x"], w))
        y0 = int(_zone(PLACEMENT_ZONES["logo"]["y"], h))
        box = (x0, y0, x0 + size, y0 + size)
        if box[2] <= w and box[3] <= h and not _overlaps(box, placed):
            break
    x0, y0, x1, y1 = box
    if ellipse:
        draw.ellipse([x0, y0, x1, y1], fill=color)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=color)
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=random.choice([1, 2])))
    f = font(int(size * 0.4))
    tl, tt, tr, tb = draw.textbbox((0, 0), letters, font=f)
    tw, th = tr - tl, tb - tt
    draw.text((x0 + (size - tw) / 2, y0 + (size - th) / 2 - tt), letters, fill="white", font=f)
    return "logo", box


def draw_table(draw, w, h, placed):
    rows, cols = random.randint(3, 6), random.randint(2, 4)
    row_h = random.randint(int(0.03 * h), int(0.05 * h))
    box = (0, 0, 0, rows * row_h)
    for _try in range(MAX_PLACEMENT_TRIES):
        x0 = int(_zone(PLACEMENT_ZONES["table"]["x_left"], w))
        x1 = int(_zone(PLACEMENT_ZONES["table"]["x_right"], w))
        y0 = int(_zone(PLACEMENT_ZONES["table"]["y"], h))
        y1 = y0 + rows * row_h
        box = (x0, y0, x1, y1)
        if x1 > x0 and y1 <= h and not _overlaps(box, placed):
            break
    x0, y0, x1, y1 = box
    draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
    for i in range(1, rows):
        y = y0 + i * row_h
        draw.line([(x0, y), (x1, y)], fill="black", width=1)
    col_edges = [x0 + j * (x1 - x0) // cols for j in range(cols + 1)]
    for x in col_edges[1:-1]:
        draw.line([(x, y0), (x, y1)], fill="black", width=1)

    # Cell content -- drawn strictly inside the grid computed above, so the
    # table's outer YOLO box (still just `box`) is unaffected.
    col_types = [random.choice(TABLE_COLUMN_TYPES) for _ in range(cols)]
    header_row = random.random() < 0.6
    pad = 4
    for i in range(rows):
        for j in range(cols):
            cx0, cx1 = col_edges[j], col_edges[j + 1]
            cy0 = y0 + i * row_h
            text = TABLE_HEADER_LABELS[col_types[j]] if (header_row and i == 0) else _table_cell_text(col_types[j])
            f = _fit_font(draw, text, max(1, cx1 - cx0 - 2 * pad), max(9, int(row_h * 0.45)))
            tl, tt, tr, tb = draw.textbbox((0, 0), text, font=f)
            tw, th = tr - tl, tb - tt
            tx = cx1 - pad - tw if col_types[j] in TABLE_NUMERIC_COLUMN_TYPES else cx0 + pad
            ty = cy0 + (row_h - th) / 2 - tt
            draw.text((tx, ty), text, fill="black", font=f)
    return "table", box


def draw_signature(draw, w, h, placed):
    pad = 6
    points = [(0, 0)]
    box = (0, 0, 0, 0)
    for _try in range(MAX_PLACEMENT_TRIES):
        x0 = int(_zone(PLACEMENT_ZONES["signature"]["x"], w))
        y0 = int(_zone(PLACEMENT_ZONES["signature"]["y"], h))
        points = [(x0, y0)]
        for _ in range(random.randint(5, 9)):
            px, py = points[-1]
            points.append((px + random.randint(10, 25), py + random.randint(-15, 15)))
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        box = (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)
        if box[2] <= w and box[3] <= h and not _overlaps(box, placed):
            break
    draw.line(points, fill="black", width=2, joint="curve")
    return "signature", box


DRAWERS = {"header": draw_header, "table": draw_table, "signature": draw_signature, "logo": draw_logo}


def make_sample(bg_path, out_img_path, out_label_path):
    img = Image.open(bg_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    chosen = random.sample(CLASSES, k=random.randint(1, 4))
    placed = []
    lines = []
    for cls in chosen:
        _, (l, t, r, b) = DRAWERS[cls](draw, w, h, placed)
        l, r = max(0, l), min(w, r)
        t, b = max(0, t), min(h, b)
        placed.append((l, t, r, b))
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


def write_generation_config():
    config = {
        "dataset_version": DATASET_VERSION,
        "spatial_randomization_version": SPATIAL_RANDOMIZATION_VERSION,
        "table_content_version": TABLE_CONTENT_VERSION,
        "seed": SEED,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "classes": CLASSES,
        "placement_zones": PLACEMENT_ZONES,
        "max_placement_tries": MAX_PLACEMENT_TRIES,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "table_column_types": TABLE_COLUMN_TYPES,
        "table_rows_range": [3, 6],
        "table_cols_range": [2, 4],
    }
    (OUT_DIR / "generation_config.json").write_text(json.dumps(config, indent=2))


def main():
    random.seed(SEED)
    build_split("train", N_TRAIN, RAW_DIR / "train")
    build_split("val", N_VAL, RAW_DIR / "val")
    write_data_yaml()
    write_generation_config()
    print(f"wrote {N_TRAIN} train / {N_VAL} val synthetic samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
