import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.data.dataloader import get_loaders
from src.models.resnet import BrainTumorResNet
from src import config


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_model(device):
    model = BrainTumorResNet(num_classes=len(CLASS_NAMES))
    checkpoint_path = "best_model.pth"  # adjust path if needed
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate():
    device = get_device()
    print(f"🖥️ Using device: {device}")

    # get_loaders returns (train_loader, test_loader, classes)
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

    # ---- Overall accuracy ----
    accuracy = (y_true == y_pred).mean() * 100.0
    print(f"\n✅ Overall Accuracy on test set: {accuracy:.2f}%\n")

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (raw counts):")
    print(cm)

    # ---- Classification report ----
    print("\nDetailed classification report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ---- Plot confusion matrix ----
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    # annotate cells
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
    out_path = "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\n📊 Saved confusion matrix plot to: {out_path}")


if __name__ == "__main__":
    evaluate()
