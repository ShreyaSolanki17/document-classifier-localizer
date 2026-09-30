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

Run: python src/localizer/train.py             (baseline, fixed-band dataset)
Run: python src/localizer/train.py --dataset v2 (spatial-randomization experiment)
Run: python src/localizer/train.py --dataset v3 (+ realistic table content)
"""

import argparse
import csv
import json
from pathlib import Path

import mlflow
from ultralytics import YOLO
from ultralytics import settings as yolo_settings

# Ultralytics has its own built-in MLflow auto-logging, which (a) would
# create a second MLflow run for the same experiment (explicitly disallowed
# by this project's tracking convention) and (b) calls
# mlflow.set_tracking_uri() to its own file-store default, clobbering the
# tracking URI this script already set and breaking our own later
# mlflow.log_metrics() calls. We do our own logging below, so disable it.
yolo_settings.update({"mlflow": False})

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs"
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

# --dataset selects which generated dataset/output/run-name to use; training
# config (epochs/imgsz/batch/optimizer/etc.) is identical across both so the
# only variable in the spatial-randomization experiment is the data itself.
DATASETS = {
    "baseline": {
        "data_yaml": ROOT / "data" / "localizer" / "data.yaml",
        "run_name": "baseline-yolov8n",
        "project_name": "localizer",
    },
    "v2": {
        "data_yaml": ROOT / "data" / "localizer_v2" / "data.yaml",
        "run_name": "localizer-spatial-randomization-v1",
        "project_name": "localizer_v2",
    },
    "v3": {
        "data_yaml": ROOT / "data" / "localizer_v3" / "data.yaml",
        "run_name": "localizer-table-content-v1",
        "project_name": "localizer_v3",
    },
}


def _log_params(dataset_key, data_yaml, output_dir):
    """Log all training and dataset parameters to the active MLflow run."""
    params = {
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
        "dataset_config": str(data_yaml),
        "output_dir": str(output_dir),
        "dataset_key": dataset_key,
    }
    # generation_config.json (written by build_synthetic_dataset.py) carries
    # the actual placement/versioning config -- reuse it instead of
    # duplicating those constants here.
    gen_config_path = data_yaml.parent / "generation_config.json"
    if gen_config_path.exists():
        gen_config = json.loads(gen_config_path.read_text())
        params["dataset_version"] = gen_config["dataset_version"]
        params["spatial_randomization_version"] = gen_config["spatial_randomization_version"]
        params["dataset_seed"] = gen_config["seed"]
        if "table_content_version" in gen_config:
            params["table_content_version"] = gen_config["table_content_version"]
            params["table_column_types"] = ",".join(gen_config["table_column_types"])
            params["table_rows_range"] = str(gen_config["table_rows_range"])
            params["table_cols_range"] = str(gen_config["table_cols_range"])
    mlflow.log_params(params)


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


def _log_artifacts(run_dir: Path, dataset_dir: Path):
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

    # Dataset generation/validation metadata, if this dataset variant has them.
    for name in ["generation_config.json", "validation_report.txt"]:
        p = dataset_dir / name
        if p.exists():
            mlflow.log_artifact(str(p))

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


def _log_cross_eval(model, baseline_yaml):
    """Evaluate the trained model on the original baseline val set too, so
    the spatial-randomization experiment can be compared on both the
    original benchmark and its own (harder) val set, without opening a
    second MLflow run for the same experiment."""
    metrics = model.val(data=str(baseline_yaml), split="val")
    mlflow.log_metrics({
        "val_baseline_dataset/precision": float(metrics.box.mp),
        "val_baseline_dataset/recall":    float(metrics.box.mr),
        "val_baseline_dataset/mAP50":     float(metrics.box.map50),
        "val_baseline_dataset/mAP50-95":  float(metrics.box.map),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="baseline")
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    data_yaml = cfg["data_yaml"]
    run_dir = OUT_DIR / cfg["project_name"]

    # Pin the store explicitly (CWD-dependent default otherwise), so the
    # experiment lands in the same place regardless of where this is run from.
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=cfg["run_name"]):
        _log_params(args.dataset, data_yaml, run_dir)

        # ------------------------------------------------------------------ #
        # YOLO training — nothing below this line changes the training config #
        # ------------------------------------------------------------------ #
        model = YOLO("yolov8n.pt")
        model.train(
            data=str(data_yaml),
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
            name=cfg["project_name"],
            exist_ok=True,
        )
        # ------------------------------------------------------------------ #

        _log_final_metrics(run_dir)
        _log_artifacts(run_dir, data_yaml.parent)

        if args.dataset != "baseline":
            _log_cross_eval(model, DATASETS["baseline"]["data_yaml"])

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")
        print(f"Experiment:    {MLFLOW_EXPERIMENT}")
        print(f"Artifacts logged from: {run_dir}")


if __name__ == "__main__":
    main()
