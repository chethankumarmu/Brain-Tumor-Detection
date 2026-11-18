"""
Brain tumor MRI classifier (4 classes): Glioma, Meningioma, No_Tumor, Pituitary.

Uses a pretrained EfficientNet-B0 and fine-tunes it on your dataset.

Expected folder layout:
    <data_dir>/train/Glioma, Meningioma, Pituitary, No_Tumor
    <data_dir>/test/Glioma,  Meningioma, Pituitary, No_Tumor

Run (Command Prompt, inside the project folder):
    python train_brain_tumor.py --data_dir "D:\\chethan\\Projects\\Brain Tumor Detection\\Epic and CSCR hospital Dataset"

Outputs (saved in --out_dir, default: the folder this script is in):
    model_best.pt         best model weights (used by predict.py and the web page)
    results.txt           per-class precision / recall / F1 and the confusion matrix
    confusion_matrix.png  confusion matrix picture
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")                      # save plots to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(img_size, train):
    """Training images get small random changes (augmentation); test images do not."""
    steps = [transforms.Resize((img_size, img_size))]          # any size -> img_size x img_size
    if train:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def build_model(num_classes, pretrained):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)   # 4 outputs instead of 1000
    return model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_pred, all_true = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_true.append(labels.numpy())
    return np.concatenate(all_true), np.concatenate(all_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="folder that contains train/ and test/")
    parser.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6, help="stop if val loss does not improve for this many epochs")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_pretrained", action="store_true", help="skip downloading pretrained weights (testing only)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("WARNING: no GPU found, training will be slow.")

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    # Two views of the same training folder: one with augmentation, one without (for validation)
    train_aug = datasets.ImageFolder(train_dir, transform=make_transforms(args.img_size, train=True))
    train_plain = datasets.ImageFolder(train_dir, transform=make_transforms(args.img_size, train=False))
    test_set = datasets.ImageFolder(test_dir, transform=make_transforms(args.img_size, train=False))
    classes = train_aug.classes
    assert classes == test_set.classes, "train and test folders must have the same class folders"
    print("Classes:", classes)

    # Split the training folder into train and validation, keeping class proportions the same
    indices = np.arange(len(train_aug))
    labels = np.array(train_aug.targets)
    idx_train, idx_val = train_test_split(indices, test_size=args.val_fraction,
                                          random_state=42, stratify=labels)
    print(f"Train: {len(idx_train)}  Validation: {len(idx_val)}  Test: {len(test_set)}")

    train_loader = DataLoader(Subset(train_aug, idx_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(Subset(train_plain, idx_val), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # Class weights: rarer classes count more in the loss, so the model doesn't ignore them
    counts = np.bincount(labels[idx_train], minlength=len(classes)).astype(np.float32)
    class_weights = counts.sum() / (len(classes) * counts)
    print("Training images per class:", {c: int(n) for c, n in zip(classes, counts)})
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device))

    model = build_model(len(classes), pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []   # one entry per round, saved to train_history.json
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    best_path = os.path.join(args.out_dir, "model_best.pt")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        train_loss_sum, train_correct, train_total = 0.0, 0, 0
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += loss.item() * images.size(0)
            train_correct += (logits.argmax(1) == labels_batch).sum().item()
            train_total += images.size(0)

        # ---- validate ----
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=use_amp):
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                logits = model(images)
                val_loss_sum += criterion(logits, labels_batch).item() * images.size(0)
                val_correct += (logits.argmax(1) == labels_batch).sum().item()
                val_total += images.size(0)

        train_loss = train_loss_sum / train_total
        val_loss = val_loss_sum / val_total
        scheduler.step(val_loss)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {train_loss:.4f} acc {train_correct/train_total:.3f} "
              f"| val loss {val_loss:.4f} acc {val_correct/val_total:.3f} | {time.time()-start:.0f}s")

        # ---- keep the best model (lowest validation loss) ----
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save({"state_dict": model.state_dict(), "classes": classes,
                        "img_size": args.img_size}, best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    import json
    with open(os.path.join(args.out_dir, 'train_history.json'), 'w') as f:
        json.dump({'history': history, 'best_epoch': best_epoch}, f, indent=1)

    print(f"\nTraining finished. Best epoch: {best_epoch} (val loss {best_val_loss:.4f}). "
          f"Total time: {(time.time()-start)/60:.1f} min")

    # ---- final test: load best weights, score once ----
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    y_true, y_pred = predict_all(model, test_loader, device)

    report = classification_report(y_true, y_pred, target_names=classes, digits=3, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    test_acc = (y_true == y_pred).mean()

    text = [f"TEST ACCURACY: {test_acc:.3f}", "", report, "Confusion matrix (rows = true, columns = predicted):",
            "classes: " + ", ".join(classes), str(cm)]
    with open(os.path.join(args.out_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("\n".join(text))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (accuracy {test_acc:.3f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrix.png"), dpi=150)
    print("\nSaved: model_best.pt, results.txt, confusion_matrix.png in", args.out_dir)


if __name__ == "__main__":
    main()
