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

    for epoch in range(1, epochs + 1):
        # ---- Unfreeze backbone at chosen epoch ----
        if epoch == config.UNFREEZE_EPOCH and hasattr(model, "set_backbone_requires_grad"):
            model.set_backbone_requires_grad(True)
            optimizer = _make_optimizer(model, config.LR_FINE_TUNE)
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

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

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # ---- Save best model ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✅ Saved new best model at epoch {epoch} (Val Acc: {val_acc:.2f}%)")

    print("✅ Training complete.")
    return model
