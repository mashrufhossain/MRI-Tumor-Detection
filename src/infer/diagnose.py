import torch
from torchvision import transforms
from PIL import Image
import os
from src.models.resnet import BrainTumorResNet

# --- setup ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BrainTumorResNet(num_classes=4)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# class names must match your training order
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]

# preprocessing pipeline same as during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# --- loop through images ---
folder_path = "path/to/your/new_mri_folder"

for img_name in os.listdir(folder_path):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        diagnosis = CLASS_NAMES[pred]
        confidence = probs[0][pred].item() * 100

        print(f"🧠 {img_name}: {diagnosis} ({confidence:.2f}% confidence)")
