"""Build a small labeled subset from the Kaggle mirror of RVL-CDIP's test
split (pdavpoojan/the-rvlcdip-dataset-test, downloaded manually via
`kaggle datasets download`) and lay it out as
data/raw/<train|val>/<class>/*.jpg for torchvision's ImageFolder.

Source is only the RVL-CDIP *test* split (~2400-2500 images/class across all
16 classes), not the full 320k-image training set -- plenty for the
500-1000/class subset this project needs, and it sidesteps having to stream
the full ~40GB dataset.
"""

import io
import random
import zipfile
from pathlib import Path

from PIL import Image

ZIP_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "kaggle_raw"
    / "the-rvlcdip-dataset-test.zip"
)
CLASSES = ["invoice", "form", "letter", "resume", "memo"]
N_TRAIN = 800  # per class
N_VAL = 200  # per class
SEED = 42
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def main():
    for c in CLASSES:
        (OUT_DIR / "train" / c).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "val" / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for cls in CLASSES:
            members = [n for n in z.namelist() if n.startswith(f"test/{cls}/")]
            rng.shuffle(members)
            needed = N_TRAIN + N_VAL
            if len(members) < needed:
                raise ValueError(f"only {len(members)} images for class {cls}, need {needed}")

            for i, member in enumerate(members[:needed]):
                split_name = "train" if i < N_TRAIN else "val"
                idx = i if i < N_TRAIN else i - N_TRAIN
                with z.open(member) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                out_path = OUT_DIR / split_name / cls / f"{idx:04d}.jpg"
                img.save(out_path, "JPEG", quality=90)

            print(f"{cls}: {N_TRAIN} train, {N_VAL} val")

    print("done")


if __name__ == "__main__":
    main()
