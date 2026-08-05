"""Fine-tune ResNet18 on a 5-class RVL-CDIP subset (invoice, form, letter,
resume, memo).

Training choices, and why:
- Full fine-tune, not a frozen backbone: document scans are visually far
  from ImageNet's natural images, so letting the early conv layers adapt
  helps more than it risks at this dataset size.
- Adam, lr=1e-4, single param group for the whole network: the dataset is
  small enough that per-layer learning rates aren't worth the complexity.
- Light augmentation only (mild resize-crop, small rotation, no color
  jitter): these are document layouts, not natural scenes, so aggressive
  geometric/color augmentation would distort structural cues (table
  alignment, header position) more than it would help generalization.
- Early stopping on val loss (patience=5): 4000 train images total overfits
  within ~15-20 epochs on a ResNet18.

Known limitations (see also README):
- Single train/val split, no k-fold cross-validation -- with 1000
  images/class per class this is a real risk of an optimistic/lucky split.
- Train and val are both random slices of the same Kaggle RVL-CDIP
  test-split source, not independently collected -- val accuracy likely
  overstates real-world generalization to other scanners/sources.
- Classes are perfectly balanced by construction (800/200 each), which
  hides how the model would behave under the class imbalance seen in real
  document workflows.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "runs" / "classifier"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
PATIENCE = 5
SEED = 42


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_loaders():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)
    assert train_ds.classes == val_ds.classes
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader, train_ds.classes


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def plot_curves(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "curves.png")


def evaluate(model, val_loader, classes, device):
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)
    (OUT_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png")


def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, classes = get_loaders()
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch:02d}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "classes": classes}, OUT_DIR / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch}")
                break

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_curves(history)
    evaluate(model, val_loader, classes, device)


if __name__ == "__main__":
    main()
