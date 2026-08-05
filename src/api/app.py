"""Sync FastAPI endpoint tying the three components together: classify the
document type, locate fields, OCR each field's text.

Why sync, not async/Celery: v1 scope explicitly excludes async task queues
(see README) -- this is a portfolio demo, not a production service handling
concurrent load. A single request blocking on model inference is an
acceptable tradeoff at this scale.

Run: uvicorn src.api.app:app --reload
Then: curl -F "file=@path/to/doc.jpg" http://localhost:8000/classify
"""

import sys
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
# ponytail: classifier/extraction aren't packages (no __init__.py, mirrors
# how their own test_smoke.py scripts import each other) -- reuse their
# code via sys.path rather than duplicating build_model/extract_fields.
sys.path.insert(0, str(ROOT / "src" / "classifier"))
sys.path.insert(0, str(ROOT / "src" / "extraction"))

from extract import extract_fields  # noqa: E402
from train import IMG_SIZE, build_model  # noqa: E402

CLASSIFIER_CKPT = ROOT / "runs" / "classifier" / "best_model.pt"
LOCALIZER_WEIGHTS = ROOT / "runs" / "localizer" / "weights" / "best.pt"

app = FastAPI(title="Document Classifier + Field Localizer")

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ckpt = torch.load(CLASSIFIER_CKPT, map_location=_device)
CLASSES = _ckpt["classes"]
_classifier = build_model(len(CLASSES)).to(_device)
_classifier.load_state_dict(_ckpt["model"])
_classifier.eval()

_localizer = YOLO(str(LOCALIZER_WEIGHTS))

# must match the classifier's val_tf (src/classifier/train.py) exactly
_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def classify(image: Image.Image):
    x = _transform(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        probs = torch.softmax(_classifier(x), dim=1)[0]
    idx = int(probs.argmax())
    return CLASSES[idx], float(probs[idx])


@app.post("/classify")
def classify_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        image = Image.open(tmp_path).convert("RGB")
        doc_type, confidence = classify(image)
        fields = extract_fields(tmp_path, model=_localizer)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "document_type": doc_type,
        "confidence": round(confidence, 4),
        "fields": fields,
    }
