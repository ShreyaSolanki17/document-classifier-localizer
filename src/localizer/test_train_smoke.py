"""Sanity check for YOLO model loading/inference plumbing -- catches a
broken install or wrong model name before committing to a full training
run. Run directly: python src/localizer/test_train_smoke.py
"""

import numpy as np
from ultralytics import YOLO


def test_model_loads_and_predicts():
    model = YOLO("yolov8n.pt")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy, verbose=False)
    assert len(results) == 1


if __name__ == "__main__":
    test_model_loads_and_predicts()
    print("ok")
