"""Sanity check that the /classify endpoint runs end-to-end against a real
sample image and returns well-formed JSON. Loads real trained weights (no
mocking) -- this is the one test in the repo that exercises the full
pipeline together. Run directly: python src/api/test_smoke.py
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app import CLASSES, app

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "val"


def test_classify_endpoint_returns_well_formed_response():
    sample = next(SAMPLE_DIR.rglob("*.jpg"))
    client = TestClient(app)
    with open(sample, "rb") as f:
        resp = client.post("/classify", files={"file": (sample.name, f, "image/jpeg")})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["document_type"] in CLASSES, body
    assert 0 <= body["confidence"] <= 1, body
    assert isinstance(body["fields"], list), body
    for field in body["fields"]:
        assert {"class", "confidence", "bbox", "text"} <= field.keys(), field


if __name__ == "__main__":
    test_classify_endpoint_returns_well_formed_response()
    print("ok")
