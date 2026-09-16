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

import csv
from pathlib import Path

import mlflow
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

MLFLOW_EXPERIMENT = "document-localizer"

# Dataset metadata (from build_synthetic_dataset.py constants).
N_TRAIN = 400
N_VAL = 80
CLASSES = ["header", "table", "signature", "logo"]


def _log_params():
    """Log all training and dataset parameters to the active MLflow run."""
    mlflow.log_params({
        "model": "yolov8n",
        "task": "detection",
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "batch": 16,          # ultralytics default when not overridden
        "seed": SEED,
        "workers": WORKERS,
        "amp": False,
        "optimizer": "auto",
        "lr0": 0.01,
        "patience": PATIENCE,
        "n_train_images": N_TRAIN,
        "n_val_images": N_VAL,
        "n_classes": len(CLASSES),
        "class_names": ",".join(CLASSES),
        "dataset_config": str(DATA_YAML),
        "output_dir": str(OUT_DIR / "localizer"),
    })


def _log_final_metrics(run_dir: Path):
    """Parse YOLO's results.csv and log the last-epoch validation metrics."""
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found at {results_csv}")

    with results_csv.open() as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    if not rows:
        raise ValueError("results.csv is empty — training may have failed.")

    # YOLO reports the best checkpoint metrics at the epoch that was saved;
    # the last row in results.csv is the final completed training epoch.
    # We log the final-epoch validation metrics to match what YOLO reports.
    last = {k.strip(): v.strip() for k, v in rows[-1].items()}

    mlflow.log_metrics({
        "val/precision":    float(last["metrics/precision(B)"]),
        "val/recall":       float(last["metrics/recall(B)"]),
        "val/mAP50":        float(last["metrics/mAP50(B)"]),
        "val/mAP50-95":     float(last["metrics/mAP50-95(B)"]),
        "train/box_loss":   float(last["train/box_loss"]),
        "train/cls_loss":   float(last["train/cls_loss"]),
        "train/dfl_loss":   float(last["train/dfl_loss"]),
        "val/box_loss":     float(last["val/box_loss"]),
        "val/cls_loss":     float(last["val/cls_loss"]),
        "val/dfl_loss":     float(last["val/dfl_loss"]),
        "epoch":            float(last["epoch"]),
    })


def _log_artifacts(run_dir: Path):
    """Upload the YOLO-generated output files to the active MLflow run."""
    # Core model checkpoints.
    for ckpt in ["weights/best.pt", "weights/last.pt"]:
        p = run_dir / ckpt
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="weights")

    # Training metrics CSV.
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        mlflow.log_artifact(str(csv_path))

    # Standard YOLO plots (training curves, confusion matrix, PR curves, etc.).
    plot_patterns = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
        "labels.jpg",
        "labels_correlogram.jpg",
        "val_batch0_labels.jpg",
        "val_batch0_pred.jpg",
        "val_batch1_labels.jpg",
        "val_batch1_pred.jpg",
        "val_batch2_labels.jpg",
        "val_batch2_pred.jpg",
        "train_batch0.jpg",
        "train_batch1.jpg",
        "train_batch2.jpg",
        "args.yaml",
    ]
    for name in plot_patterns:
        p = run_dir / name
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="plots")


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="baseline-yolov8n"):
        _log_params()

        # ------------------------------------------------------------------ #
        # YOLO training — nothing below this line changes the training config #
        # ------------------------------------------------------------------ #
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
        # ------------------------------------------------------------------ #

        run_dir = OUT_DIR / "localizer"
        _log_final_metrics(run_dir)
        _log_artifacts(run_dir)

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"Experiment:    {MLFLOW_EXPERIMENT}")
        print(f"Artifacts logged from: {run_dir}")


if __name__ == "__main__":
    main()
