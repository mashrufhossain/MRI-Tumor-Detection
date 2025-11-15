import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src import config


def _make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr, weight_decay=1e-4)


def _evaluate(model, loader, device):
    """Run validation and return average loss and accuracy."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, running_loss = 0, 0, 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc


def train_model(model, train_loader, test_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()

    # ---- Phase 1: train head only ----
    optimizer = _make_optimizer(model, config.LR_HEAD)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_acc = 0.0  # persists through epochs

    # ---- Set up logging ----
    os.makedirs("outputs/logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"outputs/logs/train_{timestamp}.txt"

    log_file = None
    try:
        log_file = open(log_path, "w")
        log_file.write(f"Training started: {timestamp}\n")
        log_file.write(f"Device: {device}\n")
        log_file.write(f"Epochs: {epochs}\n")
        log_file.write(f"LR_HEAD: {getattr(config, 'LR_HEAD', None)}\n")
        log_file.write(f"LR_FINE_TUNE: {getattr(config, 'LR_FINE_TUNE', None)}\n")
        log_file.write(f"UNFREEZE_EPOCH: {getattr(config, 'UNFREEZE_EPOCH', None)}\n")
        log_file.write("\n")
    except Exception as e:
        print(f"⚠️ Could not open training log file: {e}")
        log_file = None

    for epoch in range(1, epochs + 1):
        # ---- Unfreeze backbone at chosen epoch ----
        if epoch == config.UNFREEZE_EPOCH and hasattr(model, "set_backbone_requires_grad"):
            model.set_backbone_requires_grad(True)
            optimizer = _make_optimizer(model, config.LR_FINE_TUNE)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            if log_file:
                log_file.write(f"\n[Epoch {epoch}] Unfroze backbone, switched to fine-tune LR.\n")

        # ---- Training Loop ----
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            running_total += labels.size(0)
            running_correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * running_correct / running_total

        # ---- Validation ----
        val_loss, val_acc = _evaluate(model, test_loader, device)
        scheduler.step(val_loss)

        msg = (
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )
        print(msg)

        if log_file:
            log_file.write(msg + "\n")

        # ---- Save best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_model.pth")
            best_msg = f"✅ Saved new best model at epoch {epoch} (Val Acc: {val_acc:.2f}%)"
            print(best_msg)
            if log_file:
                log_file.write(best_msg + "\n")

    print("✅ Training complete.")

    if log_file:
        log_file.write(f"\nBest Val Acc: {best_acc:.2f}%\n")
        log_file.write("Training complete.\n")
        log_file.close()
        print(f"📝 Training log saved to: {log_path}")

    return model
