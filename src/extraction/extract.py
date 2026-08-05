"""Thin OCR extraction layer: runs the trained YOLO localizer on a document
image, then runs Tesseract on each detected region to pull out its text.

Why crop-then-OCR instead of running Tesseract on the whole page: the
localizer's boxes tell us *what kind* of region we're reading (header vs.
table vs. signature), which a whole-page OCR dump doesn't -- and OCR
accuracy is generally better on a tight, single-purpose crop than a full
busy page.

Known limitation: signature text extraction is close to meaningless --
Tesseract reads handwriting/scrawl as garbage characters, not because the
crop is wrong but because it's not printed text. Signature boxes are kept
for their location (useful for "is this signed?" checks) but the OCR text
from them should not be trusted.

Run: python src/extraction/extract.py <image_path>
"""

import os
import sys
from pathlib import Path

import pytesseract
from PIL import Image
from ultralytics import YOLO

# ponytail: pytesseract needs the Tesseract *binary* path, not just the pip
# wrapper. Default Windows install location, overridable via env var for
# other machines/platforms where it's already on PATH.
pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

MODEL_PATH = Path(__file__).resolve().parents[2] / "runs" / "localizer" / "weights" / "best.pt"
CONF_THRESHOLD = 0.5
# ponytail: Tesseract's default page-segmentation mode assumes a full page
# with columns/paragraphs, and returns empty output on small single-region
# crops like ours (verified: a clearly legible "INVOICE #3423" crop OCR'd
# to '' under the default PSM). --psm 6 ("single uniform block of text")
# is correct for every crop this pipeline produces, not a special case.
OCR_CONFIG = "--psm 6"


def extract_fields(image_path, model=None):
    """Returns a list of {class, confidence, bbox, text} dicts, one per detected field."""
    model = model or YOLO(str(MODEL_PATH))
    image = Image.open(image_path).convert("RGB")
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]

    fields = []
    for box in results.boxes:
        cls_name = results.names[int(box.cls)]
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        crop = image.crop((x0, y0, x1, y1))
        text = pytesseract.image_to_string(crop, config=OCR_CONFIG).strip()
        fields.append(
            {
                "class": cls_name,
                "confidence": float(box.conf),
                "bbox": [x0, y0, x1, y1],
                "text": text,
            }
        )
    return fields


def main():
    if len(sys.argv) != 2:
        print("Usage: python src/extraction/extract.py <image_path>")
        sys.exit(1)
    for field in extract_fields(sys.argv[1]):
        print(f"[{field['class']} {field['confidence']:.2f}] {field['text']!r}")


if __name__ == "__main__":
    main()
