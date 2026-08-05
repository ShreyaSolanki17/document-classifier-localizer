"""Sanity check that Tesseract is wired up correctly (binary path resolves,
OCR reads rendered text) -- catches install/config issues without needing
the trained localizer model. Run directly: python src/extraction/test_smoke.py
"""

import pytesseract
from PIL import Image, ImageDraw

from extract import OCR_CONFIG  # also sets pytesseract.tesseract_cmd as an import side effect


def test_ocr_reads_rendered_text():
    img = Image.new("RGB", (300, 60), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "HELLO WORLD", fill="black")
    text = pytesseract.image_to_string(img, config=OCR_CONFIG)
    assert "HELLO" in text.upper(), text


if __name__ == "__main__":
    test_ocr_reads_rendered_text()
    print("ok")
