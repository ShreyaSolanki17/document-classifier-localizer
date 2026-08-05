"""Sanity check for synthetic label generation -- catches drawing/bbox bugs
(out-of-range coords, mismatched line count) without needing the real
dataset. Run directly: python src/localizer/test_smoke.py
"""

import shutil
import tempfile
from pathlib import Path

from PIL import Image

from build_synthetic_dataset import CLASSES, make_sample


def test_bboxes_in_range_and_count_matches():
    tmp = Path(tempfile.mkdtemp())
    try:
        bg_path = tmp / "bg.jpg"
        Image.new("RGB", (1000, 1300), "white").save(bg_path)
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
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    test_bboxes_in_range_and_count_matches()
    print("ok")
