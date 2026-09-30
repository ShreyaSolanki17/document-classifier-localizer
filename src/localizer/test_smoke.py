"""Sanity check for synthetic label generation -- catches drawing/bbox bugs
(out-of-range coords, mismatched line count) and validates the spatially
randomized placement (build_synthetic_dataset.py) without needing the real
dataset. Run directly: python src/localizer/test_smoke.py
"""

import random
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from build_synthetic_dataset import CLASSES, make_sample


def _fresh_bg(tmp):
    bg_path = tmp / "bg.jpg"
    Image.new("RGB", (1000, 1300), "white").save(bg_path)
    return bg_path


def test_bboxes_in_range_and_count_matches():
    tmp = Path(tempfile.mkdtemp())
    try:
        bg_path = _fresh_bg(tmp)
        img_path, label_path = tmp / "out.jpg", tmp / "out.txt"

        for _ in range(10):  # random chosen-class count varies per call; run a few
            make_sample(bg_path, img_path, label_path)
            lines = label_path.read_text().splitlines()
            assert 1 <= len(lines) <= len(CLASSES), lines
            for line in lines:
                cls_id, xc, yc, bw, bh = line.split()
                assert 0 <= int(cls_id) < len(CLASSES)
                for v in (xc, yc, bw, bh):
                    assert 0.0 <= float(v) <= 1.0, line
                assert float(bw) > 0 and float(bh) > 0, line
    finally:
        shutil.rmtree(tmp)


def test_same_seed_is_deterministic():
    tmp = Path(tempfile.mkdtemp())
    try:
        bg_path = _fresh_bg(tmp)

        random.seed(123)
        make_sample(bg_path, tmp / "a.jpg", tmp / "a.txt")
        random.seed(123)
        make_sample(bg_path, tmp / "b.jpg", tmp / "b.txt")

        assert (tmp / "a.txt").read_text() == (tmp / "b.txt").read_text()
    finally:
        shutil.rmtree(tmp)


def test_placements_vary_across_images():
    tmp = Path(tempfile.mkdtemp())
    try:
        bg_path = _fresh_bg(tmp)
        header_centers = set()

        for i in range(15):
            img_path, label_path = tmp / f"v{i}.jpg", tmp / f"v{i}.txt"
            make_sample(bg_path, img_path, label_path)
            for line in label_path.read_text().splitlines():
                cls_id, xc, yc, _, _ = line.split()
                if int(cls_id) == CLASSES.index("header"):
                    header_centers.add((round(float(xc), 2), round(float(yc), 2)))

        # old fixed-band generator always placed header in one narrow top
        # corner; randomized placement should land in several distinct spots
        assert len(header_centers) > 3, header_centers
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_bboxes_in_range_and_count_matches()
    test_same_seed_is_deterministic()
    test_placements_vary_across_images()
    print("ok")
