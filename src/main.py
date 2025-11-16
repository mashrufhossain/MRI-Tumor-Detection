import torch
from src.data.dataloader import get_loaders
from src.models.resnet import BrainTumorResNet
import src.config as config
from src.train.trainer import train_model


def main():
    # ---- Device Selection (MPS first, then CUDA, then CPU) ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥️ Using device: {device}")

    # ---- Load Data ----
    train_loader, test_loader, classes = get_loaders()

    # ---- Initialize Model ----
    model = BrainTumorResNet(num_classes=len(classes), freeze_backbone=True).to(device)

    # ---- Try loading best checkpoint ----
    checkpoint_path = "weights/best_model.pth"
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ Loaded best model from {checkpoint_path}")
    except FileNotFoundError:
        print("⚠️ No saved model found — starting fresh.")

    # ---- Train ----
    train_model(model, train_loader, test_loader, config.EPOCHS, device)


if __name__ == "__main__":
    main()
