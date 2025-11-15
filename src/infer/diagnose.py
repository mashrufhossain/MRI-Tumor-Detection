import argparse
import os
import csv

import torch
from PIL import Image

from src.models.resnet import BrainTumorResNet
from src import config


CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
transform = config.transform_test


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


device = get_device()
print(f"🖥️ Using device: {device}")


def load_model():
    model = BrainTumorResNet(num_classes=len(CLASS_NAMES))
    checkpoint_path = "best_model.pth"  # adjust if path differs
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


model = load_model()


def predict_image(img_path: str):
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    pred_idx = probs.argmax()
    diagnosis = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx] * 100.0)
    return diagnosis, confidence, probs


def save_rows_to_csv(rows, csv_path: str):
    if not rows:
        print("No rows to save, skipping CSV.")
        return

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_on_path(path: str, csv_path: str | None = None):
    rows = []

    if os.path.isdir(path):
        print(f"📁 Running diagnosis on all images under folder (recursive): {path}")

        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, path)

                diagnosis, confidence, probs = predict_image(full_path)
                print(f"🧠 {rel_path}: {diagnosis} ({confidence:.2f}% confidence)")

                rows.append(
                    {
                        "relative_path": rel_path,
                        "filename": fname,
                        "prediction": diagnosis,
                        "confidence": confidence,
                        **{
                            f"prob_{cls}": float(p * 100.0)
                            for cls, p in zip(CLASS_NAMES, probs)
                        },
                    }
                )
    else:
        print(f"🖼️ Running diagnosis on single image: {path}")
        diagnosis, confidence, probs = predict_image(path)

        print(
            f"🧠 {os.path.basename(path)}: {diagnosis} "
            f"({confidence:.2f}% confidence)"
        )

        rows.append(
            {
                "relative_path": os.path.basename(path),
                "filename": os.path.basename(path),
                "prediction": diagnosis,
                "confidence": confidence,
                **{
                    f"prob_{cls}": float(p * 100.0)
                    for cls, p in zip(CLASS_NAMES, probs)
                },
            }
        )

    # If --csv was provided, save immediately
    if csv_path and rows:
        save_rows_to_csv(rows, csv_path)
        print(f"\n📄 Saved predictions to: {csv_path}")

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose brain MRI images using trained ResNet model."
    )
    parser.add_argument(
        "path",
        help="Path to an image file OR a folder (or folder-of-folders) of images.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Optional path to save predictions as CSV immediately.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"❌ Path not found: {args.path}")
        return

    rows = run_on_path(args.path, args.csv_path)

    # If user didn't pass --csv, ask interactively
    if args.csv_path is None and rows:
        choice = input(
            "\n❓ Save results to a CSV file? (y/n): "
        ).strip().lower()

        if choice == "y":
            default_name = "predictions.csv"
            user_filename = input(
                f"📄 Enter filename (press Enter for '{default_name}'): "
            ).strip()
            csv_path = user_filename if user_filename else default_name
            save_rows_to_csv(rows, csv_path)
            print(f"\n✅ Saved predictions to: {csv_path}\n")
        else:
            print("\n👍 Okay, not saving to CSV.\n")


if __name__ == "__main__":
    main()
