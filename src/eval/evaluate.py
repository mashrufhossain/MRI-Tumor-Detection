import os
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader

from src.data.dataloader import get_loaders
from src.models.resnet import BrainTumorResNet
from src import config


CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(device):
    model = BrainTumorResNet(num_classes=len(CLASS_NAMES))
    checkpoint_path = "weights/best_model.pth"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def print_confusion_matrix(cm, class_names):
    max_len = max(len(cls) for cls in class_names)
    pad = max_len + 2

    print(" " * (pad + 10) + "Predicted →")
    print(" " * pad + " ".join(f"{cls:>{pad}}" for cls in class_names))
    print("Actual ↓")

    for i, actual_class in enumerate(class_names):
        row = f"{actual_class:<{pad}}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>{pad}}"
        print(row)


def save_eval_log(log_path, dataset_name, accuracy, cm, class_names, report_str):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")

        f.write("Confusion matrix (rows = actual, columns = predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n\n")

        f.write("Classification report:\n")
        f.write(report_str)
        f.write("\n")


def evaluate():
    device = get_device()
    print(f"🖥️ Using device: {device}")

    _, test_loader, class_names = get_loaders()
    print(f"Class names from dataloader: {class_names}")

    model = load_model(device)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean() * 100.0
    print(f"\n✅ Overall Accuracy on test set: {accuracy:.2f}%\n")

    cm = confusion_matrix(y_true, y_pred)
    print("\nNeatly formatted confusion matrix:")
    print_confusion_matrix(cm, class_names)

    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print("\nDetailed classification report:")
    print(report_str)

    # ---- Logging ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"outputs/logs/eval_test_{timestamp}.txt"
    save_eval_log(
        log_path=log_path,
        dataset_name="default_test_set",
        accuracy=accuracy,
        cm=cm,
        class_names=class_names,
        report_str=report_str,
    )
    print(f"\n📝 Saved evaluation log to: {log_path}")

    # ---- Save confusion matrix PNG ----
    os.makedirs("outputs/confusion_matrices", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    max_val = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > max_val / 2 else "black",
            )

    fig.tight_layout()
    out_path = "outputs/confusion_matrices/confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n📊 Saved confusion matrix image to: {out_path}")


def evaluate_on_folder(path):
    if not os.path.exists(path):
        print(f"❌ Path not found: {path}")
        return

    print(f"📁 Evaluating model on folder: {path}")

    dataset = datasets.ImageFolder(root=path, transform=config.transform_test)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    class_names = dataset.classes
    print(f"Detected classes: {class_names}")

    device = get_device()
    model = load_model(device)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = (y_true == y_pred).mean() * 100.0
    print(f"\n✅ Accuracy: {accuracy:.2f}%\n")

    cm = confusion_matrix(y_true, y_pred)
    print("\nNeatly formatted confusion matrix:")
    print_confusion_matrix(cm, class_names)

    report_str = classification_report(y_true, y_pred, target_names=class_names)
    print("\nDetailed classification report:")
    print(report_str)

    # ---- Logging ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dataset_name = os.path.basename(path.rstrip("/")) or path
    log_path = f"outputs/logs/eval_{dataset_name}_{timestamp}.txt"

    save_eval_log(
        log_path=log_path,
        dataset_name=dataset_name,
        accuracy=accuracy,
        cm=cm,
        class_names=class_names,
        report_str=report_str,
    )

    print(f"\n📝 Saved evaluation log to: {log_path}")

    # ---- Save confusion matrix PNG ----
    os.makedirs("outputs/confusion_matrices", exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {dataset_name}")

    max_val = cm.max() if cm.size > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > max_val / 2 else "black",
            )

    fig.tight_layout()
    save_path = f"outputs/confusion_matrices/cm_{dataset_name}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 Saved confusion matrix image to: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir:
        evaluate_on_folder(args.data_dir)
    else:
        evaluate()
