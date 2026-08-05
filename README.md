# Document Classifier + Field Localizer

A two-model PyTorch pipeline for scanned business documents: a CNN
classifies the document type, then a YOLO detector locates key regions
(header, table, signature, logo), and Tesseract OCRs the text out of each
region found.

Built as a from-scratch portfolio piece — every model here is trained, not
called off a pretrained API.

```
document image
      │
      ▼
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  Classifier │ --> │   Localizer   │ --> │  Extraction  │
│  (ResNet18) │     │  (YOLOv8n)    │     │ (Tesseract)  │
│ doc type    │     │ field boxes   │     │ field text   │
└─────────────┘     └───────────────┘     └──────────────┘
```

## Component 1: Document classifier

ResNet18, fine-tuned on a 5-class subset of RVL-CDIP (invoice, form,
letter, resume, memo — the original spec's "receipt/ID card/certificate"
wording doesn't match RVL-CDIP's real 16 classes, so these were chosen
instead: visually distinct layouts, all well-represented in the dataset).
800 train / 200 val images per class. Adam, lr=1e-4, early-stopped at
epoch 10 on val loss.

**Result: 86.2% val accuracy.**

| class   | precision | recall | f1    |
|---------|-----------|--------|-------|
| resume  | 0.945     | 0.940  | 0.942 |
| memo    | 0.926     | 0.870  | 0.897 |
| invoice | 0.800     | 0.880  | 0.838 |
| letter  | 0.838     | 0.800  | 0.818 |
| form    | 0.812     | 0.820  | 0.816 |

resume and memo are near-perfectly separated (bullet/section layout vs.
short text block). The confusion concentrates in form/invoice/letter —
all dense-text or grid layouts that are genuinely hard to tell apart from
a downsampled 224×224 image alone (see `runs/classifier/confusion_matrix.png`).
This is exactly the kind of ambiguity the localizer helps resolve
downstream (e.g. a detected table strongly suggests invoice/form over
letter).

**Known limitations:**
- Single train/val split, no k-fold cross-validation — with 200
  images/class in val, this is a real risk of an optimistic/lucky split.
- Train and val are both random slices of the same Kaggle RVL-CDIP
  test-split source, not independently collected — val accuracy likely
  overstates real-world generalization to other scanners/sources.
- Classes are perfectly balanced by construction, which hides how the
  model would behave under the class imbalance seen in real document
  workflows.

Run: `python src/classifier/build_dataset.py` then `python src/classifier/train.py`

## Component 2: Field localizer

YOLOv8n, fine-tuned to detect 4 field types: header, table, signature,
logo. Trained on a **synthetically labeled** dataset — 400 train / 80 val
images built by programmatically compositing drawn header/table/
signature/logo elements onto real document images (reused from
component 1's data), with bounding boxes that are exact by construction.
This was a deliberate choice to avoid manual annotation entirely; see
`src/localizer/build_synthetic_dataset.py` for the generation logic.

50 epochs, `amp=False` (see hardware note below).

**Result: mAP50 0.979, mAP50-95 0.919** (final epoch, no early stop
triggered — see `runs/localizer/results.png`).

| class     | recall |
|-----------|--------|
| table     | 0.98   |
| logo      | 0.91   |
| header    | 0.90   |
| signature | 0.79   |

signature is the weakest class — it's the thinnest, easiest-to-miss shape
in the set, and the most likely to be confused with background clutter on
a busy real scan (see `runs/localizer/confusion_matrix_normalized.png`).

**Known limitations:**
- Trained entirely on synthetic overlays composited onto real
  backgrounds — val metrics measure how well the model finds *these*
  synthetic shapes, not genuine headers/tables/signatures/logos. Expect a
  real accuracy drop on production scans without fine-tuning on real
  labeled examples.
- Elements are placed in fixed screen zones (top strip, middle band,
  bottom-right) rather than the full layout variety real documents show.

**Hardware note:** training on this repo's dev machine (8GB RAM, GTX 1650)
required `workers=0` (DataLoader worker subprocesses exhausted available
RAM) and `amp=False` (mixed precision produced NaN losses on this GPU).
Both are already the defaults in `train.py`.

Run: `python src/localizer/build_synthetic_dataset.py` then `python src/localizer/train.py`

## Component 3: Extraction

Thin wrapper: runs the localizer on an image, crops each detected field,
and OCRs the crop with Tesseract (`--psm 6` — the default page-segmentation
mode assumes a full page and returns empty output on small single-region
crops, verified against a real "INVOICE #3423" header crop that OCR'd to
`''` under the default before this fix).

```
python src/extraction/extract.py path/to/document.jpg
```

```
[header 0.90] 'INVOICE #3423'
[table 0.98] '...(whatever real text falls in that region)...'
[logo 0.98] 'CR'
```

**Known limitation:** signature OCR text is close to meaningless —
Tesseract reads handwriting/scrawl as garbage characters, not because the
crop is wrong but because it isn't printed text. Signature boxes are
still useful for their *location* (e.g. "is this document signed?") but
the extracted text from them shouldn't be trusted.

## Setup

```
pip install -r requirements.txt
```

Also requires the Tesseract OCR binary installed separately (not just the
`pytesseract` pip package) — see https://github.com/UB-Mannheim/tesseract/wiki
for Windows builds. Point `extract.py` at a non-default install location
via the `TESSERACT_CMD` environment variable.

Kaggle credentials for `build_dataset.py` go in `~/.kaggle/kaggle.json`
(the Kaggle CLI's expected location — not inside this repo).

## Status / what's next

Classifier, localizer, and extraction are all done and wired together
end-to-end. Not yet built: a FastAPI endpoint tying the three components
into one request/response service (sync only for v1 — Celery/Redis/async
queues, multi-format ingestion beyond images, and auth are explicitly out
of scope, future work only).
