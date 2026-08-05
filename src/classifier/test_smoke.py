"""Sanity check for the classifier's model/transform plumbing -- catches
wiring bugs (wrong num_classes, bad tensor shapes) without needing the
dataset or a GPU. Run directly: python src/classifier/test_smoke.py
"""

import torch
import torchvision.transforms as transforms
from PIL import Image

from train import IMG_SIZE, build_model


def test_model_forward_shape():
    model = build_model(num_classes=5)
    model.eval()
    dummy = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 5), out.shape


def test_val_transform_shape():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])
    img = Image.new("RGB", (300, 400))
    tensor = tf(img)
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), tensor.shape


if __name__ == "__main__":
    test_model_forward_shape()
    test_val_transform_shape()
    print("ok")
