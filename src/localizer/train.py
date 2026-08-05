"""Fine-tune YOLOv8n on the synthetic field-localizer dataset.

Why YOLOv8n via ultralytics, not hand-rolled: ultralytics already handles
augmentation, early stopping, checkpointing, and eval plots -- reimplementing
that would just be worse and buggier for no benefit. YOLOv8n (nano) over
larger variants because the dataset is small (480 images) with 4 simple,
visually distinct classes -- a bigger backbone would just overfit faster.

Known limitations (see also src/localizer/build_synthetic_dataset.py):
- Trained entirely on synthetic overlays composited onto real document
  backgrounds -- val metrics measure how well the model finds *these*
  synthetic shapes, not genuine headers/tables/signatures/logos. Expect a
  real accuracy drop on production scans without fine-tuning on real
  labeled examples.

Run: python src/localizer/train.py
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path(__file__).resolve().parents[2] / "data" / "localizer" / "data.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "runs"
EPOCHS = 50
IMG_SIZE = 640
PATIENCE = 10
SEED = 42
# ponytail: this machine has 8GB RAM total (often <2GB free) -- any
# DataLoader worker subprocess has to reload torch/CUDA DLLs and that
# reliably dies (OSError WinError 1455 / silent kill) under this little
# headroom. workers=0 loads data in the main process instead, no spawn.
# Revisit if training on a machine with more RAM.
WORKERS = 0


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        patience=PATIENCE,
        seed=SEED,
        workers=WORKERS,
        # ponytail: GTX 1650 (Turing, no Tensor Cores) produced NaN
        # box/cls/dfl losses from epoch 1 with AMP on, despite the AMP
        # sanity check passing -- a known instability on this GPU class.
        # Full fp32 is slower but numerically stable; dataset is tiny
        # (480 images) so the slowdown doesn't matter here.
        amp=False,
        project=str(OUT_DIR),
        name="localizer",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
