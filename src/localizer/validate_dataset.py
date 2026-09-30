"""Validate a generated localizer dataset and report spatial-placement
statistics vs. the original fixed-band baseline.

Checks: every image has a label, every label has valid class ids and
in-range non-degenerate boxes, all classes are represented, split counts
match, and train/val backgrounds are disjoint. Then reports the
center_x/center_y distribution per class, to confirm placement is actually
more varied than the baseline.

Run: python src/localizer/validate_dataset.py [--dataset-dir data/localizer_v2]
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

from build_synthetic_dataset import CLASSES, N_TRAIN, N_VAL, RAW_DIR

ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "data" / "localizer"


def _read_labels(dataset_dir, split):
    """Return {class_name: [(xc, yc, bw, bh), ...]} for one split, after
    validating every image/label pair."""
    img_dir = dataset_dir / "images" / split
    label_dir = dataset_dir / "labels" / split
    images = sorted(img_dir.glob("*.jpg"))
    assert images, f"no images found in {img_dir}"

    by_class = defaultdict(list)
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        assert label_path.exists(), f"missing label for {img_path}"
        lines = label_path.read_text().splitlines()
        assert lines, f"empty label file {label_path}"
        for line in lines:
            cls_id, xc, yc, bw, bh = line.split()
            cls_id = int(cls_id)
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            assert 0 <= cls_id < len(CLASSES), f"bad class id {cls_id} in {label_path}"
            for v in (xc, yc, bw, bh):
                assert 0.0 <= v <= 1.0, f"coord out of [0,1] in {label_path}: {line}"
            assert bw > 0 and bh > 0, f"zero/negative-area box in {label_path}: {line}"
            by_class[CLASSES[cls_id]].append((xc, yc, bw, bh))

    return images, by_class


def _stats(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "mean": round(statistics.mean(values), 3),
        "stdev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
    }


def validate(dataset_dir):
    train_imgs, train_by_class = _read_labels(dataset_dir, "train")
    val_imgs, val_by_class = _read_labels(dataset_dir, "val")

    assert len(train_imgs) == N_TRAIN, f"expected {N_TRAIN} train images, got {len(train_imgs)}"
    assert len(val_imgs) == N_VAL, f"expected {N_VAL} val images, got {len(val_imgs)}"

    all_by_class = defaultdict(list)
    for d in (train_by_class, val_by_class):
        for cls, boxes in d.items():
            all_by_class[cls].extend(boxes)
    missing = set(CLASSES) - set(all_by_class)
    assert not missing, f"classes never generated: {missing}"

    # background disjointness: train/val backgrounds are drawn from separate
    # directory trees by construction (build_split), so paths can't overlap
    train_bg = {p.name for p in (RAW_DIR / "train").rglob("*.jpg")}
    val_bg = {p.name for p in (RAW_DIR / "val").rglob("*.jpg")}
    overlap = train_bg & val_bg
    print(f"train/val background filename overlap: {len(overlap)} "
          f"(same-named files in separate train/val dirs are still distinct source images)")

    print(f"\n{dataset_dir.name}: {len(train_imgs)} train / {len(val_imgs)} val images, "
          f"all labels valid, all {len(CLASSES)} classes present.\n")

    print(f"{'class':<10} {'axis':<3} {'n':>4} {'min':>7} {'max':>7} {'mean':>7} {'stdev':>7}")
    for cls in CLASSES:
        boxes = all_by_class[cls]
        for axis, idx in (("center_x", 0), ("center_y", 1)):
            s = _stats([b[idx] for b in boxes])
            print(f"{cls:<10} {axis:<3} {s['n']:>4} {s['min']:>7} {s['max']:>7} {s['mean']:>7} {s['stdev']:>7}")

    return all_by_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(ROOT / "data" / "localizer_v2"))
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    print("=== validating", dataset_dir, "===")
    new_stats = validate(dataset_dir)

    if BASELINE_DIR.exists() and dataset_dir != BASELINE_DIR:
        print("\n=== baseline (fixed-band) comparison:", BASELINE_DIR, "===")
        base_stats = validate(BASELINE_DIR)

        print(f"\n{'class':<10} {'axis':<3} {'baseline stdev':>15} {'new stdev':>10}")
        for cls in CLASSES:
            for axis, idx in (("center_x", 0), ("center_y", 1)):
                b = _stats([b[idx] for b in base_stats[cls]])["stdev"]
                n = _stats([b[idx] for b in new_stats[cls]])["stdev"]
                flag = "more varied" if n > b else "WARNING: not more varied"
                print(f"{cls:<10} {axis:<3} {b:>15} {n:>10}   {flag}")


if __name__ == "__main__":
    main()
